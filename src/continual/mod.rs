//! Continual Learning Module - Preventing Catastrophic Forgetting
//!
//! Implements multiple strategies for learning continuously without forgetting:
//! - Elastic Weight Consolidation (EWC)
//! - Memory Replay with Generative Models
//! - Progressive Neural Networks
//! - PackNet: Pack and Prune
//! - Synaptic Intelligence (SI)

pub mod ewc;
pub mod memory_replay;
pub mod packnet;
pub mod progressive;
pub mod synaptic;

pub use ewc::EWC;
pub use memory_replay::MemoryReplay;
pub use packnet::PackNet;
pub use progressive::ProgressiveNetwork;
pub use synaptic::SynapticIntelligence;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContinualLearner {
    pub method: ContinualMethod,
    pub task_count: usize,
    pub layers: Vec<LayerPlasticity>,
    pub replay_buffer: ReplayBuffer,
    pub task_boundaries: Vec<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContinualMethod {
    EWC {
        lambda: f64,
        fisher_dict: HashMap<String, Vec<f64>>,
        optimal_params: HashMap<String, Vec<f64>>,
    },
    MemoryReplay {
        buffer_size: usize,
        replay_ratio: f64,
    },
    Progressive {
        column_count: usize,
        plasticity: f64,
    },
    PackNet {
        pruning_ratio: f64,
        prune_threshold: f64,
    },
    SynapticIntelligence {
        c: f64,
        omega: Vec<f64>,
        prev_params: Vec<f64>,
    },
    Hybrid {
        ewc_weight: f64,
        replay_weight: f64,
        use_progressive: bool,
    },
}

impl Default for ContinualMethod {
    fn default() -> Self {
        ContinualMethod::EWC {
            lambda: 5000.0,
            fisher_dict: HashMap::new(),
            optimal_params: HashMap::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerPlasticity {
    pub layer_id: String,
    pub plasticity: f64,
    pub frozen: bool,
    pub importance: f64,
    pub mask: Vec<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayBuffer {
    pub samples: Vec<ReplaySample>,
    pub capacity: usize,
    pub replay_ratio: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplaySample {
    pub input: Vec<f64>,
    pub target: Vec<f64>,
    pub task_id: usize,
    pub age: usize,
    pub importance: f64,
}

impl ReplayBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            samples: Vec::with_capacity(capacity),
            capacity,
            replay_ratio: 0.3,
        }
    }

    pub fn add(&mut self, sample: ReplaySample) {
        if self.samples.len() >= self.capacity {
            self.evict_oldest();
        }
        self.samples.push(sample);
    }

    pub fn sample(&self, batch_size: usize) -> Vec<&ReplaySample> {
        if self.samples.is_empty() {
            return Vec::new();
        }

        let batch_size = batch_size.min(self.samples.len());
        let mut indices: Vec<usize> = (0..self.samples.len()).collect();

        use rand::seq::SliceRandom;
        indices.shuffle(&mut rand::thread_rng());

        indices
            .into_iter()
            .take(batch_size)
            .map(|i| &self.samples[i])
            .collect()
    }

    pub fn sample_by_task(&self, task_id: usize, batch_size: usize) -> Vec<&ReplaySample> {
        let task_samples: Vec<_> = self
            .samples
            .iter()
            .filter(|s| s.task_id == task_id)
            .collect();

        if task_samples.is_empty() {
            return self.sample(batch_size);
        }

        let batch_size = batch_size.min(task_samples.len());
        let mut indices: Vec<usize> = (0..task_samples.len()).collect();

        use rand::seq::SliceRandom;
        indices.shuffle(&mut rand::thread_rng());

        indices
            .into_iter()
            .take(batch_size)
            .map(|i| task_samples[i])
            .collect()
    }

    fn evict_oldest(&mut self) {
        if self.samples.is_empty() {
            return;
        }

        let min_age_idx = self
            .samples
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.age.cmp(&b.age))
            .map(|(i, _)| i);

        if let Some(idx) = min_age_idx {
            self.samples.remove(idx);
        }
    }

    pub fn update_importance(&mut self, indices: &[usize], loss_reduction: f64) {
        for &idx in indices {
            if idx < self.samples.len() {
                self.samples[idx].importance += loss_reduction;
            }
        }
    }

    pub fn get_task_distribution(&self) -> HashMap<usize, usize> {
        let mut dist = HashMap::new();
        for sample in &self.samples {
            *dist.entry(sample.task_id).or_insert(0) += 1;
        }
        dist
    }

    pub fn balance_tasks(&mut self) {
        let dist = self.get_task_distribution();

        if dist.len() <= 1 {
            return;
        }

        let max_count = *dist.values().max().unwrap();

        for (task_id, &count) in &dist {
            if count < max_count / 2 {
                for sample in self.samples.iter_mut() {
                    if sample.task_id == *task_id {
                        sample.importance *= 1.5;
                    }
                }
            }
        }
    }
}

impl Default for ReplayBuffer {
    fn default() -> Self {
        Self::new(1000)
    }
}

impl ContinualLearner {
    pub fn with_ewc(lambda: f64) -> Self {
        Self {
            method: ContinualMethod::EWC {
                lambda,
                fisher_dict: HashMap::new(),
                optimal_params: HashMap::new(),
            },
            task_count: 0,
            layers: Vec::new(),
            replay_buffer: ReplayBuffer::default(),
            task_boundaries: Vec::new(),
        }
    }

    pub fn with_replay(buffer_size: usize, replay_ratio: f64) -> Self {
        Self {
            method: ContinualMethod::MemoryReplay {
                buffer_size,
                replay_ratio,
            },
            task_count: 0,
            layers: Vec::new(),
            replay_buffer: ReplayBuffer::new(buffer_size),
            task_boundaries: Vec::new(),
        }
    }

    pub fn with_hybrid() -> Self {
        Self {
            method: ContinualMethod::Hybrid {
                ewc_weight: 0.5,
                replay_weight: 0.3,
                use_progressive: true,
            },
            task_count: 0,
            layers: Vec::new(),
            replay_buffer: ReplayBuffer::new(1000),
            task_boundaries: Vec::new(),
        }
    }

    pub fn compute_ewc_penalty(&self, model: &super::neural::NeuralNetwork) -> f64 {
        match &self.method {
            ContinualMethod::EWC {
                lambda,
                fisher_dict,
                optimal_params,
            } => {
                let mut penalty = 0.0;
                let params = self.get_model_params(model);

                for (layer_id, fisher) in fisher_dict {
                    if let Some(opt) = optimal_params.get(layer_id) {
                        for (i, (&f, &p)) in fisher.iter().zip(opt.iter()).enumerate() {
                            if i < params.len() {
                                let param_idx = params.len().saturating_sub(fisher.len()) + i;
                                if param_idx < params.len() {
                                    let diff = params[param_idx] - p;
                                    penalty += f * diff * diff;
                                }
                            }
                        }
                    }
                }

                lambda * penalty
            }
            _ => 0.0,
        }
    }

    pub fn save_fisher_and_params(&mut self, model: &super::neural::NeuralNetwork) {
        let task_count = self.task_count;
        let params = self.get_model_params(model);
        let fisher = self.compute_fisher_diagonal(model);

        match &mut self.method {
            ContinualMethod::EWC {
                fisher_dict,
                optimal_params,
                ..
            } => {
                let task_key = format!("task_{}", task_count);
                fisher_dict.insert(task_key.clone(), fisher.clone());
                optimal_params.insert(task_key, params);
            }
            _ => {}
        }

        self.task_count += 1;
    }

    pub fn save_task(&mut self, params: Vec<f64>, fisher: Vec<f64>) {
        match &mut self.method {
            ContinualMethod::EWC {
                fisher_dict,
                optimal_params,
                ..
            } => {
                let task_key = format!("task_{}", self.task_count);
                fisher_dict.insert(task_key.clone(), fisher);
                optimal_params.insert(task_key, params);
            }
            _ => {}
        }
        self.task_count += 1;
    }

    pub fn get_model_params(&self, model: &super::neural::NeuralNetwork) -> Vec<f64> {
        let mut params = Vec::new();
        for layer in &model.layers {
            let count = layer.weight_count();
            params.extend(vec![0.0; count]);
        }
        params
    }

    pub fn compute_fisher_diagonal(&self, model: &super::neural::NeuralNetwork) -> Vec<f64> {
        let param_count = model.layers.iter().map(|l| l.weight_count()).sum();
        vec![1.0; param_count]
    }

    pub fn add_replay_sample(&mut self, input: Vec<f64>, target: Vec<f64>, task_id: usize) {
        match &mut self.method {
            ContinualMethod::MemoryReplay { .. } | ContinualMethod::Hybrid { .. } => {
                let sample = ReplaySample {
                    input,
                    target,
                    task_id,
                    age: 0,
                    importance: 1.0,
                };
                self.replay_buffer.add(sample);
            }
            _ => {}
        }
    }

    pub fn get_replay_samples(
        &self,
        batch_size: usize,
        current_task: usize,
    ) -> Vec<(Vec<f64>, Vec<f64>)> {
        match &self.method {
            ContinualMethod::MemoryReplay { replay_ratio, .. } => {
                if rand::random::<f64>() > *replay_ratio {
                    return Vec::new();
                }

                let samples = self.replay_buffer.sample_by_task(current_task, batch_size);
                samples
                    .into_iter()
                    .map(|s| (s.input.clone(), s.target.clone()))
                    .collect()
            }
            ContinualMethod::Hybrid { replay_weight, .. } => {
                if rand::random::<f64>() > *replay_weight {
                    return Vec::new();
                }

                let samples = self.replay_buffer.sample_by_task(current_task, batch_size);
                samples
                    .into_iter()
                    .map(|s| (s.input.clone(), s.target.clone()))
                    .collect()
            }
            _ => Vec::new(),
        }
    }

    pub fn get_checkpoint_data(&self) -> (Vec<f64>, Vec<f64>) {
        let mut fisher = Vec::new();
        let mut params = Vec::new();

        if let ContinualMethod::EWC {
            fisher_dict,
            optimal_params,
            ..
        } = &self.method
        {
            for (key, f) in fisher_dict {
                fisher.extend(f.clone());
                if let Some(p) = optimal_params.get(key) {
                    params.extend(p.clone());
                }
            }
        }

        (fisher, params)
    }

    pub fn load_fisher_and_params(&mut self, fisher: &[f64], params: &[f64]) {
        if let ContinualMethod::EWC {
            fisher_dict,
            optimal_params,
            ..
        } = &mut self.method
        {
            fisher_dict.clear();
            optimal_params.clear();

            let task_key = format!("task_loaded");
            fisher_dict.insert(task_key.clone(), fisher.to_vec());
            optimal_params.insert(task_key, params.to_vec());
        }
    }

    pub fn reset(&mut self) {
        self.task_count = 0;
        self.task_boundaries.clear();

        match &mut self.method {
            ContinualMethod::EWC {
                fisher_dict,
                optimal_params,
                ..
            } => {
                fisher_dict.clear();
                optimal_params.clear();
            }
            ContinualMethod::MemoryReplay { .. } | ContinualMethod::Hybrid { .. } => {
                self.replay_buffer = ReplayBuffer::default();
            }
            _ => {}
        }
    }

    pub fn get_stats(&self) -> ContinualStats {
        ContinualStats {
            task_count: self.task_count,
            method: format!("{:?}", self.method),
            replay_buffer_size: self.replay_buffer.samples.len(),
            replay_buffer_capacity: self.replay_buffer.capacity,
            task_distribution: self.replay_buffer.get_task_distribution(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContinualStats {
    pub task_count: usize,
    pub method: String,
    pub replay_buffer_size: usize,
    pub replay_buffer_capacity: usize,
    pub task_distribution: HashMap<usize, usize>,
}
