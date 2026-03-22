//! Distributed Learning - Main Learner Implementation

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedConfig {
    pub num_peers: usize,
    pub aggregation: AggregationMethod,
    pub sync_interval_ms: u64,
    pub min_peers_for_sync: usize,
}

impl Default for DistributedConfig {
    fn default() -> Self {
        Self {
            num_peers: 5,
            aggregation: AggregationMethod::FedAvg,
            sync_interval_ms: 1000,
            min_peers_for_sync: 2,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum AggregationMethod {
    FedAvg,
    FedProx,
    FedMA,
    SCAFFOLD,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelUpdate {
    pub peer_id: String,
    pub round: u64,
    pub parameters: Vec<f64>,
    pub num_samples: usize,
    pub loss: f64,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerInfo {
    pub id: String,
    pub address: String,
    pub trust_score: f64,
    pub round_last_seen: u64,
    pub is_active: bool,
}

pub struct DistributedLearner {
    config: DistributedConfig,
    peer_id: String,
    parameters: Vec<f64>,
    peers: HashMap<String, PeerInfo>,
    pending_updates: Vec<ModelUpdate>,
    round: u64,
    local_history: Vec<Vec<f64>>,
}

impl DistributedLearner {
    pub fn new(peer_id: String, config: DistributedConfig) -> Self {
        Self {
            config,
            peer_id,
            parameters: Vec::new(),
            peers: HashMap::new(),
            pending_updates: Vec::new(),
            round: 0,
            local_history: Vec::new(),
        }
    }

    pub fn initialize_parameters(&mut self, dims: usize) {
        self.parameters = vec![0.0; dims];
    }

    pub fn set_parameters(&mut self, params: Vec<f64>) {
        self.parameters = params;
    }

    pub fn get_parameters(&self) -> &Vec<f64> {
        &self.parameters
    }

    pub fn add_peer(&mut self, peer: PeerInfo) {
        self.peers.insert(peer.id.clone(), peer);
    }

    pub fn remove_peer(&mut self, peer_id: &str) {
        self.peers.remove(peer_id);
    }

    pub fn get_active_peers(&self) -> Vec<&PeerInfo> {
        self.peers
            .values()
            .filter(|p| p.is_active && p.round_last_seen >= self.round.saturating_sub(3))
            .collect()
    }

    pub fn create_update(&self, gradient: &[f64]) -> ModelUpdate {
        ModelUpdate {
            peer_id: self.peer_id.clone(),
            round: self.round,
            parameters: gradient.to_vec(),
            num_samples: 1,
            loss: 0.0,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
        }
    }

    pub fn receive_update(&mut self, update: ModelUpdate) {
        if update.round >= self.round {
            self.pending_updates.push(update);
        }
    }

    pub fn aggregate(&mut self) -> Result<Vec<f64>, String> {
        let active_peers = self.get_active_peers();
        let total_peers = active_peers.len() + 1;

        if total_peers < self.config.min_peers_for_sync {
            return Err("Not enough peers for synchronization".to_string());
        }

        let self_weight = 1.0;
        let peer_weight_sum: f64 = active_peers.iter().map(|p| p.trust_score).sum();
        let total_weight = self_weight + peer_weight_sum;

        let mut aggregated = self.parameters.clone();
        let mut total_samples = 1;

        for update in &self.pending_updates {
            let weight = if let Some(peer) = self.peers.get(&update.peer_id) {
                peer.trust_score
            } else {
                1.0
            };

            for (i, param) in update.parameters.iter().enumerate() {
                if i < aggregated.len() {
                    aggregated[i] += (weight / total_weight) * param;
                }
            }
            total_samples += update.num_samples;
        }

        let self_contribution = self_weight / total_weight;
        for (i, param) in aggregated.iter_mut().enumerate() {
            *param = self_contribution * self.parameters[i]
                + ((1.0 - self_contribution) / (total_peers as f64)) * *param;
        }

        self.pending_updates.clear();
        self.round += 1;

        Ok(aggregated)
    }

    pub fn apply_update(&mut self, new_params: Vec<f64>) {
        self.local_history.push(self.parameters.clone());
        if self.local_history.len() > 100 {
            self.local_history.remove(0);
        }
        self.parameters = new_params;
    }

    pub fn get_round(&self) -> u64 {
        self.round
    }

    pub fn advance_round(&mut self) {
        self.round += 1;
    }
}

pub struct FederatedLearningCoordinator {
    learners: HashMap<String, Arc<RwLock<DistributedLearner>>>,
    global_parameters: Vec<f64>,
    current_round: u64,
}

impl FederatedLearningCoordinator {
    pub fn new(initial_params: Vec<f64>) -> Self {
        Self {
            learners: HashMap::new(),
            global_parameters: initial_params,
            current_round: 0,
        }
    }

    pub fn register_learner(&mut self, id: String, learner: Arc<RwLock<DistributedLearner>>) {
        if let Ok(mut l) = learner.write() {
            l.set_parameters(self.global_parameters.clone());
        }
        self.learners.insert(id, learner);
    }

    pub fn broadcast_global_model(&self) {
        for (_, learner) in &self.learners {
            if let Ok(mut l) = learner.write() {
                l.set_parameters(self.global_parameters.clone());
            }
        }
    }

    pub fn collect_updates(&self) -> Vec<ModelUpdate> {
        let mut updates = Vec::new();
        for (_, learner) in &self.learners {
            if let Ok(l) = learner.read() {
                let params = l.get_parameters().clone();
                updates.push(ModelUpdate {
                    peer_id: "coordinator".to_string(),
                    round: self.current_round,
                    parameters: params,
                    num_samples: 1,
                    loss: 0.0,
                    timestamp: 0,
                });
            }
        }
        updates
    }

    pub fn update_global_model(&mut self, aggregated: Vec<f64>) {
        self.global_parameters = aggregated;
        self.current_round += 1;
        self.broadcast_global_model();
    }
}
