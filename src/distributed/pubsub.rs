//! Pub/Sub for Agent Communication

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PubSubConfig {
    pub max_subscribers: usize,
    pub message_ttl_ms: u64,
    pub enable_retained: bool,
}

impl Default for PubSubConfig {
    fn default() -> Self {
        Self {
            max_subscribers: 100,
            message_ttl_ms: 60000,
            enable_retained: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PubSubMessage {
    pub topic: String,
    pub payload: Vec<u8>,
    pub sender: String,
    pub timestamp: u64,
    pub retained: bool,
    pub qos: QoS,
}

#[derive(Debug, Clone, Serialize, Deserialize, Copy)]
pub enum QoS {
    AtMostOnce,
    AtLeastOnce,
    ExactlyOnce,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopicSubscription {
    pub topic: String,
    pub subscriber: String,
    pub qos: QoS,
}

pub struct PubSub {
    config: PubSubConfig,
    subscribers: HashMap<String, Vec<Subscriber>>,
    retained_messages: HashMap<String, PubSubMessage>,
    message_history: VecDeque<PubSubMessage>,
    max_history: usize,
}

struct Subscriber {
    id: String,
    qos: QoS,
    callback: Option<Box<dyn Fn(&PubSubMessage) + Send>>,
}

impl PubSub {
    pub fn new(config: PubSubConfig) -> Self {
        Self {
            config,
            subscribers: HashMap::new(),
            retained_messages: HashMap::new(),
            message_history: VecDeque::new(),
            max_history: 1000,
        }
    }

    pub fn subscribe(&mut self, subscriber_id: &str, topic: &str) -> Result<(), String> {
        let subscribers = self
            .subscribers
            .entry(topic.to_string())
            .or_insert_with(Vec::new);

        if subscribers.len() >= self.config.max_subscribers {
            return Err("Maximum subscribers reached".to_string());
        }

        if !subscribers.iter().any(|s| s.id == subscriber_id) {
            subscribers.push(Subscriber {
                id: subscriber_id.to_string(),
                qos: QoS::AtLeastOnce,
                callback: None,
            });
        }

        Ok(())
    }

    pub fn unsubscribe(&mut self, subscriber_id: &str, topic: &str) -> Result<(), String> {
        if let Some(subscribers) = self.subscribers.get_mut(topic) {
            subscribers.retain(|s| s.id != subscriber_id);
        }
        Ok(())
    }

    pub fn publish(&mut self, message: PubSubMessage) -> Result<usize, String> {
        let mut delivered = 0;

        self.message_history.push_back(message.clone());
        if self.message_history.len() > self.max_history {
            self.message_history.pop_front();
        }

        if message.retained {
            self.retained_messages
                .insert(message.topic.clone(), message.clone());
        }

        for (topic, subscribers) in &self.subscribers {
            if self.topic_matches(&message.topic, topic) {
                for subscriber in subscribers {
                    if subscriber.id != message.sender {
                        delivered += 1;
                    }
                }
            }
        }

        Ok(delivered)
    }

    fn topic_matches(&self, published: &str, subscribed: &str) -> bool {
        let published_parts: Vec<&str> = published.split('/').collect();
        let subscribed_parts: Vec<&str> = subscribed.split('/').collect();

        let max_len = published_parts.len().max(subscribed_parts.len());

        for i in 0..max_len {
            let p = published_parts.get(i).unwrap_or(&"");
            let s = subscribed_parts.get(i).unwrap_or(&"");

            match *s {
                "#" => return true,
                "+" => continue,
                _ if p == s => continue,
                _ => return false,
            }
        }

        true
    }

    pub fn get_retained(&self, topic: &str) -> Option<&PubSubMessage> {
        self.retained_messages.get(topic)
    }

    pub fn get_history(&self, topic: &str, count: usize) -> Vec<&PubSubMessage> {
        self.message_history
            .iter()
            .filter(|m| self.topic_matches(&m.topic, topic))
            .rev()
            .take(count)
            .collect()
    }

    pub fn clear_retained(&mut self, topic: &str) {
        self.retained_messages.remove(topic);
    }

    pub fn get_subscribers(&self, topic: &str) -> Vec<&str> {
        self.subscribers
            .get(topic)
            .map(|subs| subs.iter().map(|s| s.id.as_str()).collect())
            .unwrap_or_default()
    }

    pub fn topic_exists(&self, topic: &str) -> bool {
        self.subscribers.contains_key(topic)
    }

    pub fn get_all_topics(&self) -> Vec<&String> {
        self.subscribers.keys().collect()
    }
}
