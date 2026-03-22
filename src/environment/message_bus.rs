//! Message Bus for Inter-Agent Communication

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageBusConfig {
    pub bus_type: BusType,
    pub max_queue_size: usize,
    pub message_timeout_ms: u64,
    pub enable_routing: bool,
}

impl Default for MessageBusConfig {
    fn default() -> Self {
        Self {
            bus_type: BusType::InMemory,
            max_queue_size: 1000,
            message_timeout_ms: 5000,
            enable_routing: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BusType {
    InMemory,
    Redis,
    Kafka,
    RabbitMQ,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BusMessage {
    pub id: String,
    pub topic: String,
    pub sender: String,
    pub payload: Vec<u8>,
    pub timestamp: u64,
    pub priority: Priority,
    pub ttl_ms: u64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum Priority {
    Critical = 0,
    High = 1,
    Normal = 2,
    Low = 3,
}

impl Default for Priority {
    fn default() -> Self {
        Priority::Normal
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Subscription {
    pub topic_pattern: String,
    pub queue_size: usize,
    pub callback: Option<String>,
}

pub struct MessageBus {
    config: MessageBusConfig,
    queues: Arc<RwLock<HashMap<String, VecDeque<BusMessage>>>>,
    subscribers: Arc<RwLock<HashMap<String, Vec<String>>>>,
    routing_table: Arc<RwLock<HashMap<String, Vec<String>>>>,
    dead_letter_queue: Arc<RwLock<VecDeque<BusMessage>>>,
}

impl MessageBus {
    pub fn new(config: MessageBusConfig) -> Self {
        Self {
            config,
            queues: Arc::new(RwLock::new(HashMap::new())),
            subscribers: Arc::new(RwLock::new(HashMap::new())),
            routing_table: Arc::new(RwLock::new(HashMap::new())),
            dead_letter_queue: Arc::new(RwLock::new(VecDeque::new())),
        }
    }

    pub fn publish(&self, message: BusMessage) -> Result<(), String> {
        let topic = &message.topic;

        {
            let mut queues = self.queues.write();
            let queue = queues.entry(topic.clone()).or_insert_with(VecDeque::new);

            if queue.len() >= self.config.max_queue_size {
                queue.pop_front();
            }

            queue.push_back(message.clone());
        }

        if self.config.enable_routing {
            self.route_message(&message)?;
        }

        Ok(())
    }

    pub fn subscribe(&self, agent_id: &str, topic: &str) -> Result<(), String> {
        let mut subscribers = self.subscribers.write();
        let agents = subscribers
            .entry(topic.to_string())
            .or_insert_with(Vec::new);

        if !agents.contains(&agent_id.to_string()) {
            agents.push(agent_id.to_string());
        }

        Ok(())
    }

    pub fn unsubscribe(&self, agent_id: &str, topic: &str) -> Result<(), String> {
        let mut subscribers = self.subscribers.write();
        if let Some(agents) = subscribers.get_mut(topic) {
            agents.retain(|a| a != agent_id);
        }
        Ok(())
    }

    pub fn receive(&self, agent_id: &str, topic: &str, timeout_ms: u64) -> Option<BusMessage> {
        let queue = self.queues.read();

        if let Some(messages) = queue.get(topic) {
            for message in messages.iter() {
                if message.sender != agent_id {
                    return Some(message.clone());
                }
            }
        }

        None
    }

    pub fn receive_batch(&self, agent_id: &str, topic: &str, batch_size: usize) -> Vec<BusMessage> {
        let mut messages = Vec::new();

        let queue = self.queues.read();
        if let Some(topic_queue) = queue.get(topic) {
            for message in topic_queue.iter().take(batch_size) {
                if message.sender != agent_id {
                    messages.push(message.clone());
                }
            }
        }

        messages
    }

    fn route_message(&self, message: &BusMessage) -> Result<(), String> {
        let routing_table = self.routing_table.read();

        for (pattern, destinations) in routing_table.iter() {
            if self.matches_pattern(&message.topic, pattern) {
                for dest in destinations {
                    let mut queues = self.queues.write();
                    let queue = queues.entry(dest.clone()).or_insert_with(VecDeque::new);

                    if queue.len() < self.config.max_queue_size {
                        queue.push_back(message.clone());
                    }
                }
            }
        }

        Ok(())
    }

    fn matches_pattern(&self, topic: &str, pattern: &str) -> bool {
        if pattern == "*" {
            return true;
        }

        if pattern.ends_with("#") {
            let prefix = &pattern[..pattern.len() - 1];
            return topic.starts_with(prefix);
        }

        let pattern_parts: Vec<&str> = pattern.split('/').collect();
        let topic_parts: Vec<&str> = topic.split('/').collect();

        for (p, t) in pattern_parts.iter().zip(topic_parts.iter()) {
            if *p == "+" {
                continue;
            }
            if p != t {
                return false;
            }
        }

        true
    }

    pub fn add_route(&self, pattern: String, destination: String) -> Result<(), String> {
        let mut routing_table = self.routing_table.write();
        let destinations = routing_table.entry(pattern).or_insert_with(Vec::new);

        if !destinations.contains(&destination) {
            destinations.push(destination);
        }

        Ok(())
    }

    pub fn remove_route(&self, pattern: &str, destination: &str) -> Result<(), String> {
        let mut routing_table = self.routing_table.write();
        if let Some(destinations) = routing_table.get_mut(pattern) {
            destinations.retain(|d| d != destination);
        }
        Ok(())
    }

    pub fn get_queue_size(&self, topic: &str) -> usize {
        let queues = self.queues.read();
        queues.get(topic).map(|q| q.len()).unwrap_or(0)
    }

    pub fn get_total_messages(&self) -> usize {
        let queues = self.queues.read();
        queues.values().map(|q| q.len()).sum()
    }

    pub fn clear_queue(&self, topic: &str) {
        let mut queues = self.queues.write();
        if let Some(queue) = queues.get_mut(topic) {
            queue.clear();
        }
    }

    pub fn move_to_dead_letter(&self, message: BusMessage) {
        let mut dlq = self.dead_letter_queue.write();
        dlq.push_back(message);
    }

    pub fn get_dead_letter_messages(&self) -> Vec<BusMessage> {
        self.dead_letter_queue.read().iter().cloned().collect()
    }

    pub fn drain_dead_letter(&self) -> Vec<BusMessage> {
        let mut dlq = self.dead_letter_queue.write();
        let messages: Vec<_> = dlq.drain(..).collect();
        messages
    }
}
