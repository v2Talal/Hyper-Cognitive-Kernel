//! Real-time Learning Controller
//!
//! Core module for LIVE ONLINE LEARNING - no batch processing, no resource-heavy training.
//! Learns immediately from each sample with adaptive parameters.

pub mod adaptive_lr;
pub mod online_trainer;

pub use adaptive_lr::AdaptiveLearningRate;
pub use online_trainer::OnlineTrainer;

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealTimeLearningConfig {
    pub enable_adaptive_lr: bool,
    pub enable_gradient_clipping: bool,
    pub enable_momentum: bool,
    pub enable_elastic_weights: bool,
    pub gradient_clip_value: f64,
    pub max_grad_norm: f64,
    pub plasticity_decay: f64,
    pub memory_compression: bool,
    pub online_batch_size: usize,
    pub update_frequency_ms: u64,
}

impl Default for RealTimeLearningConfig {
    fn default() -> Self {
        Self {
            enable_adaptive_lr: true,
            enable_gradient_clipping: true,
            enable_momentum: true,
            enable_elastic_weights: true,
            gradient_clip_value: 5.0,
            max_grad_norm: 1.0,
            plasticity_decay: 0.99,
            memory_compression: true,
            online_batch_size: 1,
            update_frequency_ms: 0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealTimeStats {
    pub samples_processed: u64,
    pub total_updates: u64,
    pub successful_updates: u64,
    pub failed_updates: u64,
    pub average_loss: f64,
    pub recent_losses: Vec<f64>,
    pub learning_rate: f64,
    pub gradient_norm: f64,
    pub memory_usage_bytes: usize,
    pub throughput_samples_per_sec: f64,
    pub last_update_time_ms: u64,
}

impl Default for RealTimeStats {
    fn default() -> Self {
        Self {
            samples_processed: 0,
            total_updates: 0,
            successful_updates: 0,
            failed_updates: 0,
            average_loss: 0.0,
            recent_losses: Vec::with_capacity(1000),
            learning_rate: 0.001,
            gradient_norm: 0.0,
            memory_usage_bytes: 0,
            throughput_samples_per_sec: 0.0,
            last_update_time_ms: 0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sample {
    pub input: Vec<f64>,
    pub target: Vec<f64>,
    pub weight: f64,
    pub timestamp: u64,
    pub metadata: Option<String>,
}

impl Sample {
    pub fn new(input: Vec<f64>, target: Vec<f64>) -> Self {
        Self {
            input,
            target,
            weight: 1.0,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
            metadata: None,
        }
    }

    pub fn with_weight(mut self, weight: f64) -> Self {
        self.weight = weight;
        self
    }

    pub fn with_metadata(mut self, meta: String) -> Self {
        self.metadata = Some(meta);
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LearningDecision {
    Update,
    Skip,
    Compress,
    Batch(Vec<Sample>),
}

pub struct RealTimeLearningController {
    config: RealTimeLearningConfig,
    stats: RealTimeStats,
    adaptive_lr: AdaptiveLearningRate,
    sample_buffer: Vec<Sample>,
    last_batch_time: u64,
    processing_start: std::time::Instant,
}

impl RealTimeLearningController {
    pub fn new() -> Self {
        Self {
            config: RealTimeLearningConfig::default(),
            stats: RealTimeStats::default(),
            adaptive_lr: AdaptiveLearningRate::new(),
            sample_buffer: Vec::with_capacity(100),
            last_batch_time: 0,
            processing_start: std::time::Instant::now(),
        }
    }

    pub fn with_config(config: RealTimeLearningConfig) -> Self {
        Self {
            config: config.clone(),
            stats: RealTimeStats::default(),
            adaptive_lr: AdaptiveLearningRate::new(),
            sample_buffer: Vec::with_capacity(100),
            last_batch_time: 0,
            processing_start: std::time::Instant::now(),
        }
    }

    pub fn process_sample(&mut self, sample: Sample) -> LearningDecision {
        self.stats.samples_processed += 1;

        if self.stats.recent_losses.len() >= 1000 {
            self.stats.recent_losses.remove(0);
        }

        let current_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        if self.config.online_batch_size == 1 {
            self.stats.total_updates += 1;
            self.stats.successful_updates += 1;
            self.last_batch_time = current_time;
            return LearningDecision::Update;
        }

        self.sample_buffer.push(sample);

        if self.sample_buffer.len() >= self.config.online_batch_size {
            let batch = std::mem::take(&mut self.sample_buffer);
            self.stats.total_updates += 1;
            self.stats.successful_updates += 1;
            self.last_batch_time = current_time;
            return LearningDecision::Batch(batch);
        }

        LearningDecision::Skip
    }

    pub fn compute_adaptive_gradient(&mut self, gradients: &[f64]) -> Vec<f64> {
        let grad_norm = gradients.iter().map(|g| g * g).sum::<f64>().sqrt();

        self.stats.gradient_norm = grad_norm;

        let mut clipped_gradients = gradients.to_vec();

        if self.config.enable_gradient_clipping {
            let clip_value = self.config.gradient_clip_value;
            for grad in &mut clipped_gradients {
                *grad = grad.clamp(-clip_value, clip_value);
            }
        }

        if self.config.max_grad_norm > 0.0 && grad_norm > self.config.max_grad_norm {
            let scale = self.config.max_grad_norm / grad_norm;
            for grad in &mut clipped_gradients {
                *grad *= scale;
            }
        }

        clipped_gradients
    }

    pub fn should_update(&self, current_loss: f64) -> bool {
        if self.stats.recent_losses.is_empty() {
            return true;
        }

        let recent_avg = self.stats.recent_losses.iter().rev().take(10).sum::<f64>()
            / 10.0_f64.min(self.stats.recent_losses.len() as f64);

        if current_loss > recent_avg * 10.0 {
            return false;
        }

        true
    }

    pub fn update_learning_rate(&mut self, loss: f64) -> f64 {
        self.stats.recent_losses.push(loss);

        self.stats.average_loss =
            self.stats.recent_losses.iter().sum::<f64>() / self.stats.recent_losses.len() as f64;

        if self.config.enable_adaptive_lr {
            self.stats.learning_rate = self.adaptive_lr.compute(loss);
        }

        self.stats.learning_rate
    }

    pub fn get_stats(&self) -> RealTimeStats {
        let elapsed = self.processing_start.elapsed().as_secs_f64();
        if elapsed > 0.0 {
            let mut stats = self.stats.clone();
            stats.throughput_samples_per_sec = stats.samples_processed as f64 / elapsed;
            stats
        } else {
            self.stats.clone()
        }
    }

    pub fn reset(&mut self) {
        self.stats = RealTimeStats::default();
        self.adaptive_lr.reset();
        self.sample_buffer.clear();
        self.processing_start = std::time::Instant::now();
    }

    pub fn get_memory_usage(&self) -> usize {
        let sample_size = std::mem::size_of::<Sample>() * self.sample_buffer.capacity();
        let buffer_size = self.sample_buffer.capacity()
            * (self
                .sample_buffer
                .first()
                .map(|s| s.input.capacity() + s.target.capacity())
                .unwrap_or(0)
                * std::mem::size_of::<f64>());
        sample_size + buffer_size
    }
}

impl Default for RealTimeLearningController {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningPolicy {
    pub min_samples_for_update: usize,
    pub loss_threshold: f64,
    pub gradient_threshold: f64,
    pub skip_probability: f64,
    pub compression_ratio: f64,
}

impl Default for LearningPolicy {
    fn default() -> Self {
        Self {
            min_samples_for_update: 1,
            loss_threshold: 100.0,
            gradient_threshold: 100.0,
            skip_probability: 0.0,
            compression_ratio: 0.5,
        }
    }
}
