//! Gossip Protocol for Distributed Learning

use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::{Hash, Hasher};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GossipConfig {
    pub fanout: usize,
    pub interval_ms: u64,
    pub convergence_threshold: f64,
    pub max_peers: usize,
    pub use_push_pull: bool,
}

impl Default for GossipConfig {
    fn default() -> Self {
        Self {
            fanout: 3,
            interval_ms: 100,
            convergence_threshold: 0.95,
            max_peers: 10,
            use_push_pull: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GossipMessage {
    pub message_id: String,
    pub sender: String,
    pub vector_clock: VectorClock,
    pub payload: GossipPayload,
    pub timestamp: u64,
    pub ttl: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorClock {
    pub clock: HashMap<String, u64>,
}

impl VectorClock {
    pub fn new() -> Self {
        Self {
            clock: HashMap::new(),
        }
    }

    pub fn increment(&mut self, node_id: &str) {
        *self.clock.entry(node_id.to_string()).or_insert(0) += 1;
    }

    pub fn merge(&mut self, other: &VectorClock) {
        for (node, &version) in &other.clock {
            let current = *self.clock.get(node).unwrap_or(&0);
            if version > current {
                self.clock.insert(node.clone(), version);
            }
        }
    }

    pub fn happened_before(&self, other: &VectorClock) -> bool {
        let mut dominated = false;

        for (node, &v1) in &self.clock {
            let v2 = *other.clock.get(node).unwrap_or(&0);
            if v1 > v2 {
                return false;
            }
            if v1 < v2 {
                dominated = true;
            }
        }

        dominated
    }
}

impl Default for VectorClock {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GossipPayload {
    ModelUpdate {
        round: u64,
        parameters: Vec<f64>,
        loss: f64,
    },
    PeerInfo {
        id: String,
        address: String,
        trust: f64,
    },
    Heartbeat {
        timestamp: u64,
    },
    Termination {
        reason: String,
    },
}

pub struct GossipProtocol {
    config: GossipConfig,
    node_id: String,
    peers: HashSet<String>,
    message_buffer: VecDeque<GossipMessage>,
    received_messages: HashSet<String>,
    convergence_score: f64,
}

impl GossipProtocol {
    pub fn new(node_id: String, config: GossipConfig) -> Self {
        Self {
            config,
            node_id,
            peers: HashSet::new(),
            message_buffer: VecDeque::new(),
            received_messages: HashSet::new(),
            convergence_score: 0.0,
        }
    }

    pub fn add_peer(&mut self, peer_id: &str) {
        if self.peers.len() < self.config.max_peers {
            self.peers.insert(peer_id.to_string());
        }
    }

    pub fn remove_peer(&mut self, peer_id: &str) {
        self.peers.remove(peer_id);
    }

    pub fn get_peers(&self) -> Vec<&String> {
        self.peers.iter().collect()
    }

    pub fn select_gossip_targets(&self, exclude: Option<&str>) -> Vec<String> {
        let mut available: Vec<String> = self
            .peers
            .iter()
            .filter(|p| Some(p.as_str()) != exclude)
            .cloned()
            .collect();

        Self::shuffle_slice(&mut available);
        available.truncate(self.config.fanout);
        available
    }

    pub fn create_message(&self, payload: GossipPayload) -> GossipMessage {
        GossipMessage {
            message_id: format!("{}_{}", self.node_id, rand::random::<u64>()),
            sender: self.node_id.clone(),
            vector_clock: VectorClock::new(),
            payload,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
            ttl: 5,
        }
    }

    pub fn receive_message(&mut self, message: GossipMessage) -> Option<GossipPayload> {
        if self.received_messages.contains(&message.message_id) {
            return None;
        }

        self.received_messages.insert(message.message_id);

        if message.ttl > 0 && message.sender != self.node_id {
            return Some(message.payload);
        }

        None
    }

    pub fn disseminate(&mut self, payload: GossipPayload) -> Vec<GossipMessage> {
        let targets = self.select_gossip_targets(None);
        let mut messages = Vec::new();

        for target in targets {
            let mut message = self.create_message(payload.clone());
            message.ttl -= 1;
            messages.push(message);
        }

        messages
    }

    pub fn update_convergence(&mut self, total_nodes: usize) {
        let unique_senders = self.received_messages.len();
        self.convergence_score = (unique_senders as f64 / total_nodes as f64).min(1.0);
    }

    pub fn is_converged(&self) -> bool {
        self.convergence_score >= self.config.convergence_threshold
    }

    pub fn get_convergence(&self) -> f64 {
        self.convergence_score
    }

    pub fn prune_old_messages(&mut self, max_age_ms: u64) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        while let Some(msg) = self.message_buffer.pop_front() {
            if now - msg.timestamp < max_age_ms {
                self.message_buffer.push_front(msg);
                break;
            }
        }
    }
}

impl GossipProtocol {
    fn shuffle_slice<T>(slice: &mut [T]) {
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();
        slice.shuffle(&mut rng);
    }
}
