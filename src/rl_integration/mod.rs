//! Advanced RL Integration - Deep Reinforcement Learning with Active Inference

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RLConfig {
    pub algorithm: RLAlgorithm,
    pub gamma: f64,
    pub tau: f64,
    pub alpha: f64,
    pub beta: f64,
    pub use_her: bool,
    pub use_per: bool,
    pub use_rnd: bool,
}

impl Default for RLConfig {
    fn default() -> Self {
        Self {
            algorithm: RLAlgorithm::SAC,
            gamma: 0.99,
            tau: 0.005,
            alpha: 0.2,
            beta: 1.0,
            use_her: true,
            use_per: true,
            use_rnd: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RLAlgorithm {
    DQN,
    DDQN,
    A2C,
    PPO,
    SAC,
    TD3,
    Rainbow,
    IQN,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RLTransition {
    pub state: Vec<f64>,
    pub action: Vec<f64>,
    pub reward: f64,
    pub next_state: Vec<f64>,
    pub done: bool,
    pub priority: f64,
    pub rnd_intrinsic_reward: f64,
}

impl RLTransition {
    pub fn new(
        state: Vec<f64>,
        action: Vec<f64>,
        reward: f64,
        next_state: Vec<f64>,
        done: bool,
    ) -> Self {
        Self {
            state,
            action,
            reward,
            next_state,
            done,
            priority: 1.0,
            rnd_intrinsic_reward: 0.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveInferenceReward {
    pub extrinsic: f64,
    pub intrinsic: f64,
    pub entropy: f64,
    pub information_gain: f64,
    pub total: f64,
}

impl ActiveInferenceReward {
    pub fn compute(
        env_reward: f64,
        prediction_error: f64,
        uncertainty: f64,
        entropy: f64,
        prev_entropy: f64,
    ) -> Self {
        let information_gain = prev_entropy - entropy;

        let intrinsic = -prediction_error + uncertainty;

        let total = env_reward + intrinsic + information_gain;

        Self {
            extrinsic: env_reward,
            intrinsic,
            entropy,
            information_gain,
            total,
        }
    }
}

pub struct RLAgent {
    config: RLConfig,
    q_network: QNetwork,
    target_network: QNetwork,
    policy_network: PolicyNetwork,
    replay_buffer: PrioritizedReplayBuffer,
    rnd_predictor: RNDNetwork,
    steps: u64,
    learning_starts: u64,
    target_update_interval: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QNetwork {
    pub layers: Vec<Layer>,
    pub learning_rate: f64,
    pub target_update_rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Layer {
    pub size: usize,
    pub weights: Vec<Vec<f64>>,
    pub biases: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyNetwork {
    pub layers: Vec<Layer>,
    pub log_alpha: f64,
    pub target_entropy: f64,
}

pub struct PrioritizedReplayBuffer {
    buffer: Vec<RLTransition>,
    capacities: usize,
    alpha: f64,
    beta_start: f64,
    beta_frames: u64,
    frame: u64,
    priority_sum: f64,
    priority_tree: Vec<f64>,
}

impl PrioritizedReplayBuffer {
    pub fn new(capacity: usize, alpha: f64) -> Self {
        Self {
            buffer: Vec::with_capacity(capacity),
            capacities: capacity,
            alpha,
            beta_start: 0.4,
            beta_frames: 100000,
            frame: 0,
            priority_sum: 0.0,
            priority_tree: vec![0.0; capacity * 4],
        }
    }

    pub fn add(&mut self, transition: RLTransition) {
        let priority = self.priority_sum.max(1e-6);

        if self.buffer.len() >= self.capacities {
            self.buffer.remove(0);
        }

        self.buffer.push(transition);
    }

    pub fn sample(&self, batch_size: usize, beta: f64) -> Vec<(RLTransition, f64, usize)> {
        let mut samples = Vec::new();

        for _ in 0..batch_size.min(self.buffer.len()) {
            let idx = rand::random::<usize>() % self.buffer.len();
            let transition = &self.buffer[idx];
            let weight = (self.buffer.len() as f64 * transition.priority)
                .recip()
                .powf(beta);
            samples.push((transition.clone(), weight, idx));
        }

        samples
    }

    pub fn update_priorities(&mut self, indices: &[usize], priorities: &[f64]) {
        for (&idx, &priority) in indices.iter().zip(priorities.iter()) {
            if idx < self.buffer.len() {
                self.buffer[idx].priority = priority.powf(self.alpha);
            }
        }
    }
}

pub struct RNDNetwork {
    target_network: Vec<Vec<f64>>,
    predictor_network: Vec<Vec<f64>>,
    learning_rate: f64,
}

impl RNDNetwork {
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        let mut rng_seed = 42u64;

        let mut make_weights = |size: usize| -> Vec<Vec<f64>> {
            (0..size)
                .map(|_| {
                    rng_seed = rng_seed.wrapping_mul(1103515245).wrapping_add(12345);
                    (0..hidden_size)
                        .map(|_| {
                            rng_seed = rng_seed.wrapping_mul(1103515245).wrapping_add(12345);
                            Self::xorshift_f64(rng_seed) * 0.01
                        })
                        .collect()
                })
                .collect()
        };

        Self {
            target_network: make_weights(input_size),
            predictor_network: make_weights(input_size),
            learning_rate: 0.001,
        }
    }

    fn xorshift_f64(seed: u64) -> f64 {
        let mut x = seed;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        (x as i64).abs() as f64 / (i64::MAX as f64) * 2.0 - 1.0
    }

    pub fn compute_intrinsic_reward(&self, state: &[f64]) -> f64 {
        let target_features: Vec<f64> = self
            .target_network
            .iter()
            .map(|w| w.iter().zip(state.iter()).map(|(w, s)| w * s).sum())
            .collect();

        let pred_features: Vec<f64> = self
            .predictor_network
            .iter()
            .map(|w| w.iter().zip(state.iter()).map(|(w, s)| w * s).sum())
            .collect();

        target_features
            .iter()
            .zip(pred_features.iter())
            .map(|(t, p)| (t - p).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    pub fn update(&mut self, state: &[f64], target: &[f64]) {
        let pred: Vec<f64> = self
            .predictor_network
            .iter()
            .map(|w| w.iter().zip(state.iter()).map(|(w, s)| w * s).sum())
            .collect();

        for (i, layer) in self.predictor_network.iter_mut().enumerate() {
            for (j, weight) in layer.iter_mut().enumerate() {
                if i < target.len() && j < pred.len() {
                    let error = target[i] - pred[j];
                    *weight +=
                        self.learning_rate * error * state.get(i % state.len()).unwrap_or(&0.0);
                }
            }
        }
    }
}

impl RLAgent {
    pub fn new(config: RLConfig, state_dim: usize, action_dim: usize) -> Self {
        let alpha = config.alpha;
        let learning_starts = 1000;
        let target_update_interval = match config.algorithm {
            RLAlgorithm::DQN | RLAlgorithm::DDQN => 500,
            _ => 1,
        };

        Self {
            config,
            q_network: QNetwork {
                layers: vec![],
                learning_rate: 0.001,
                target_update_rate: 0.001,
            },
            target_network: QNetwork {
                layers: vec![],
                learning_rate: 0.001,
                target_update_rate: 0.001,
            },
            policy_network: PolicyNetwork {
                layers: vec![],
                log_alpha: 0.0,
                target_entropy: -(action_dim as f64) * 0.98,
            },
            replay_buffer: PrioritizedReplayBuffer::new(100000, alpha),
            rnd_predictor: RNDNetwork::new(state_dim, 64, state_dim),
            steps: 0,
            learning_starts,
            target_update_interval,
        }
    }

    pub fn select_action(&mut self, state: &[f64], deterministic: bool) -> Vec<f64> {
        if deterministic || rand::random::<f64>() < 0.01 {
            vec![0.5; 4]
        } else {
            vec![rand::random::<f64>() * 2.0 - 1.0; 4]
        }
    }

    pub fn store_transition(&mut self, transition: RLTransition) {
        self.replay_buffer.add(transition);
    }

    pub fn update(&mut self, batch: &[RLTransition]) -> f64 {
        self.steps += 1;

        let beta = self.replay_buffer.beta_start
            + (1.0 - self.replay_buffer.beta_start)
                * (self.steps as f64 / self.replay_buffer.beta_frames as f64).min(1.0);

        let loss: f64 = batch
            .iter()
            .map(|t| {
                let intrinsic = if self.config.use_rnd {
                    self.rnd_predictor.compute_intrinsic_reward(&t.state)
                } else {
                    0.0
                };
                (t.reward + intrinsic).powi(2)
            })
            .sum::<f64>()
            / batch.len() as f64;

        if self.steps % self.target_update_interval == 0 {
            self.update_target_network();
        }

        loss
    }

    fn update_target_network(&mut self) {}

    pub fn compute_active_inference_reward(
        &self,
        env_reward: f64,
        prediction_error: f64,
        state: &[f64],
    ) -> f64 {
        let uncertainty = if self.config.use_rnd {
            self.rnd_predictor.compute_intrinsic_reward(state)
        } else {
            0.0
        };

        let ai_reward =
            ActiveInferenceReward::compute(env_reward, prediction_error, uncertainty, 0.0, 0.0);

        ai_reward.total
    }
}

pub struct QLearning {
    pub q_table: HashMap<String, Vec<f64>>,
    pub learning_rate: f64,
    pub discount_factor: f64,
    pub epsilon: f64,
    pub epsilon_decay: f64,
    pub min_epsilon: f64,
}

impl QLearning {
    pub fn new(action_count: usize) -> Self {
        Self {
            q_table: HashMap::new(),
            learning_rate: 0.1,
            discount_factor: 0.99,
            epsilon: 1.0,
            epsilon_decay: 0.995,
            min_epsilon: 0.01,
        }
    }

    pub fn select_action(&mut self, state: &[f64]) -> usize {
        let state_key = Self::state_to_key(state);

        if rand::random::<f64>() < self.epsilon {
            return rand::random::<usize>() % 4;
        }

        self.q_table
            .get(&state_key)
            .map(|q| {
                q.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            })
            .unwrap_or(0)
    }

    pub fn update(
        &mut self,
        state: &[f64],
        action: usize,
        reward: f64,
        next_state: &[f64],
        done: bool,
    ) {
        let state_key = Self::state_to_key(state);
        let next_key = Self::state_to_key(next_state);

        let qsa = *self
            .q_table
            .entry(state_key.clone())
            .or_insert_with(|| vec![0.0; 4])
            .get(action)
            .unwrap_or(&0.0);

        let max_next_q = self
            .q_table
            .get(&next_key)
            .map(|q| q.iter().cloned().fold(f64::MIN, f64::max))
            .unwrap_or(0.0);

        let target = if done {
            reward
        } else {
            reward + self.discount_factor * max_next_q
        };

        let entry = self
            .q_table
            .entry(state_key.clone())
            .or_insert_with(|| vec![0.0; 4]);
        if action < entry.len() {
            entry[action] += self.learning_rate * (target - qsa);
        }

        self.epsilon = (self.epsilon * self.epsilon_decay).max(self.min_epsilon);
    }

    fn state_to_key(state: &[f64]) -> String {
        state
            .iter()
            .map(|&v| if v > 0.5 { "1" } else { "0" })
            .collect::<String>()
    }
}
