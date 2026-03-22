//! Neural Network Module for Online Deep Learning
//!
//! Implements efficient neural networks designed for REAL-TIME online learning.
//! Key features:
//! - No batch processing - learns from single samples immediately
//! - Adaptive learning rates per layer
//! - Elastic Weight Consolidation for catastrophic forgetting prevention
//! - Reservoir computing for temporal patterns
//! - Sparse connectivity for efficiency

pub mod activations;
pub mod layers;
pub mod optimizer;

pub use activations::ActivationFunction;
pub use layers::{Attention, ConvolutionalLayer, DenseLayer, Layer, RecurrentLayer, LSTM};
pub use optimizer::{AdamConfig, OnlineOptimizer, RMSpropConfig, SGDConfig};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralNetwork {
    pub layers: Vec<Layer>,
    pub input_size: usize,
    pub output_size: usize,
    pub learning_stats: LearningStats,
    config: NetworkConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    pub online_learning: bool,
    pub adaptive_lr: bool,
    pub gradient_clip: f64,
    pub max_grad_norm: f64,
    pub plasticity: f64,
    pub memory_efficient: bool,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            online_learning: true,
            adaptive_lr: true,
            gradient_clip: 5.0,
            max_grad_norm: 1.0,
            plasticity: 0.01,
            memory_efficient: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningStats {
    pub total_updates: u64,
    pub prediction_count: u64,
    pub correct_predictions: u64,
    pub total_loss: f64,
    pub recent_loss: Vec<f64>,
    pub gradient_magnitude: f64,
    pub learning_rate: f64,
    pub online_accuracy: f64,
}

impl Default for LearningStats {
    fn default() -> Self {
        Self {
            total_updates: 0,
            prediction_count: 0,
            correct_predictions: 0,
            total_loss: 0.0,
            recent_loss: Vec::with_capacity(100),
            gradient_magnitude: 0.0,
            learning_rate: 0.001,
            online_accuracy: 0.0,
        }
    }
}

impl NeuralNetwork {
    pub fn new(input_size: usize, output_size: usize) -> Self {
        Self {
            layers: Vec::new(),
            input_size,
            output_size,
            learning_stats: LearningStats::default(),
            config: NetworkConfig::default(),
        }
    }

    pub fn with_config(input_size: usize, output_size: usize, config: NetworkConfig) -> Self {
        Self {
            layers: Vec::new(),
            input_size,
            output_size,
            learning_stats: LearningStats::default(),
            config,
        }
    }

    pub fn add_dense(&mut self, size: usize, activation: ActivationFunction) -> &mut Self {
        let input = if self.layers.is_empty() {
            self.input_size
        } else {
            self.layers.last().unwrap().output_size()
        };

        self.layers
            .push(Layer::Dense(DenseLayer::new(input, size, activation)));
        self
    }

    pub fn add_lstm(&mut self, hidden_size: usize) -> &mut Self {
        let input = if self.layers.is_empty() {
            self.input_size
        } else {
            self.layers.last().unwrap().output_size()
        };

        self.layers.push(Layer::LSTM(LSTM::new(input, hidden_size)));
        self
    }

    pub fn add_attention(&mut self, heads: usize) -> &mut Self {
        let size = if self.layers.is_empty() {
            self.input_size
        } else {
            self.layers.last().unwrap().output_size()
        };

        self.layers
            .push(Layer::Attention(Attention::new(size, heads)));
        self
    }

    pub fn add_reservoir(&mut self, size: usize, leak_rate: f64) -> &mut Self {
        let input = if self.layers.is_empty() {
            self.input_size
        } else {
            self.layers.last().unwrap().output_size()
        };

        self.layers.push(Layer::Reservoir(ReservoirLayer::new(
            input, size, leak_rate,
        )));
        self
    }

    pub fn forward(&mut self, input: &[f64]) -> Vec<f64> {
        let mut current = input.to_vec();

        for layer in &mut self.layers {
            current = layer.forward(&current);
        }

        self.learning_stats.prediction_count += 1;
        current
    }

    pub fn online_train(&mut self, input: &[f64], target: &[f64]) -> f64 {
        if !self.config.online_learning {
            return 0.0;
        }

        let output = self.forward(input);
        let loss = self.compute_loss(&output, target);

        self.learning_stats.total_updates += 1;
        self.learning_stats.total_loss += loss;

        self.learning_stats.recent_loss.push(loss);
        if self.learning_stats.recent_loss.len() > 100 {
            self.learning_stats.recent_loss.remove(0);
        }

        let gradient = self.compute_gradient(input, target);
        self.apply_gradient(&gradient);

        if self.learning_stats.total_updates % 100 == 0 {
            self.update_learning_rate();
        }

        loss
    }

    fn compute_loss(&self, output: &[f64], target: &[f64]) -> f64 {
        output
            .iter()
            .zip(target.iter())
            .map(|(o, t)| {
                let diff = o - t;
                diff * diff
            })
            .sum::<f64>()
            / output.len() as f64
    }

    fn compute_gradient(&mut self, input: &[f64], target: &[f64]) -> Vec<f64> {
        let mut gradient = Vec::new();

        for (i, layer) in self.layers.iter_mut().enumerate() {
            let layer_grad = layer.compute_gradients(input, target);
            gradient.extend(layer_grad);
        }

        gradient
    }

    fn apply_gradient(&mut self, gradient: &[f64]) {
        let mut grad_idx = 0;

        for layer in &mut self.layers {
            let layer_size = layer.weight_count();
            if grad_idx + layer_size <= gradient.len() {
                layer.apply_gradients(&gradient[grad_idx..grad_idx + layer_size]);
                grad_idx += layer_size;
            }
        }

        let grad_norm = gradient.iter().map(|g| g * g).sum::<f64>().sqrt();

        self.learning_stats.gradient_magnitude = grad_norm;
    }

    fn update_learning_rate(&mut self) {
        if !self.config.adaptive_lr {
            return;
        }

        let recent_loss_avg = if !self.learning_stats.recent_loss.is_empty() {
            self.learning_stats.recent_loss.iter().sum::<f64>()
                / self.learning_stats.recent_loss.len() as f64
        } else {
            0.0
        };

        let current_lr = self.learning_stats.learning_rate;

        let new_lr = if recent_loss_avg < 0.01 {
            current_lr * 0.99
        } else if recent_loss_avg > 0.1 {
            current_lr * 1.05
        } else {
            current_lr
        };

        self.learning_stats.learning_rate = new_lr.clamp(0.0001, 0.1);
    }

    pub fn predict(&mut self, input: &[f64]) -> Vec<f64> {
        self.forward(input)
    }

    pub fn predict_class(&mut self, input: &[f64]) -> usize {
        let output = self.forward(input);
        output
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    pub fn get_accuracy(&self) -> f64 {
        if self.learning_stats.prediction_count == 0 {
            return 0.0;
        }
        self.learning_stats.correct_predictions as f64 / self.learning_stats.prediction_count as f64
    }

    pub fn get_average_loss(&self) -> f64 {
        if self.learning_stats.total_updates == 0 {
            return 0.0;
        }
        self.learning_stats.total_loss / self.learning_stats.total_updates as f64
    }

    pub fn get_stats(&self) -> NeuralStats {
        NeuralStats {
            layer_count: self.layers.len(),
            total_parameters: self.layers.iter().map(|l| l.weight_count()).sum(),
            total_updates: self.learning_stats.total_updates,
            prediction_count: self.learning_stats.prediction_count,
            average_loss: self.get_average_loss(),
            current_lr: self.learning_stats.learning_rate,
            gradient_magnitude: self.learning_stats.gradient_magnitude,
        }
    }

    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            layer.reset();
        }
        self.learning_stats = LearningStats::default();
    }

    pub fn summary(&self) -> String {
        let mut s = format!("NeuralNetwork [{} -> ", self.input_size);
        for layer in &self.layers {
            s.push_str(&format!("{} -> ", layer.output_size()));
        }
        s.push_str(&format!("{}]", self.output_size));
        s
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralStats {
    pub layer_count: usize,
    pub total_parameters: usize,
    pub total_updates: u64,
    pub prediction_count: u64,
    pub average_loss: f64,
    pub current_lr: f64,
    pub gradient_magnitude: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReservoirLayer {
    pub input_size: usize,
    pub size: usize,
    pub leak_rate: f64,
    pub weights_input: Vec<Vec<f64>>,
    pub weights_reservoir: Vec<Vec<f64>>,
    pub state: Vec<f64>,
    pub bias: Vec<f64>,
    rng_seed: u64,
}

impl ReservoirLayer {
    pub fn new(input_size: usize, size: usize, leak_rate: f64) -> Self {
        let mut rng_seed = (input_size as u64).wrapping_mul(size as u64);

        let weights_input: Vec<Vec<f64>> = (0..size)
            .map(|_| {
                rng_seed = rng_seed.wrapping_mul(1103515245).wrapping_add(12345);
                (0..input_size)
                    .map(|_| {
                        rng_seed = rng_seed.wrapping_mul(1103515245).wrapping_add(12345);
                        Self::xorshift_f64(rng_seed) * 0.1
                    })
                    .collect()
            })
            .collect();

        let weights_reservoir: Vec<Vec<f64>> = (0..size)
            .map(|_| {
                rng_seed = rng_seed.wrapping_mul(1103515245).wrapping_add(12345);
                (0..size)
                    .map(|_| {
                        rng_seed = rng_seed.wrapping_mul(1103515245).wrapping_add(12345);
                        Self::xorshift_f64(rng_seed) * 0.5
                    })
                    .collect()
            })
            .collect();

        Self {
            input_size,
            size,
            leak_rate,
            weights_input,
            weights_reservoir,
            state: vec![0.0; size],
            bias: vec![0.0; size],
            rng_seed,
        }
    }

    fn xorshift_f64(seed: u64) -> f64 {
        let mut x = seed;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        (x as i64).abs() as f64 / (i64::MAX as f64) * 2.0 - 1.0
    }

    pub fn forward(&mut self, input: &[f64]) -> Vec<f64> {
        let new_state: Vec<f64> = (0..self.size)
            .map(|i| {
                let mut pre_activation = self.bias[i];

                for (j, &w) in self.weights_input[i].iter().enumerate() {
                    if j < input.len() {
                        pre_activation += w * input[j];
                    }
                }

                for (j, &w) in self.weights_reservoir[i].iter().enumerate() {
                    pre_activation += w * self.state[j];
                }

                (1.0 - self.leak_rate) * self.state[i] + self.leak_rate * pre_activation.tanh()
            })
            .collect();

        self.state = new_state.clone();
        self.state.clone()
    }

    pub fn reset(&mut self) {
        self.state.fill(0.0);
    }

    pub fn output_size(&self) -> usize {
        self.size
    }

    pub fn weight_count(&self) -> usize {
        self.input_size * self.size + self.size * self.size + self.size
    }

    pub fn compute_gradients(&mut self, _input: &[f64], _target: &[f64]) -> Vec<f64> {
        vec![0.0; self.weight_count()]
    }

    pub fn apply_gradients(&mut self, _gradients: &[f64]) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_creation() {
        let mut net = NeuralNetwork::new(4, 2);
        net.add_dense(8, ActivationFunction::ReLU);
        net.add_dense(2, ActivationFunction::Softmax);

        assert_eq!(net.layers.len(), 2);
    }

    #[test]
    fn test_forward_pass() {
        let mut net = NeuralNetwork::new(4, 2);
        net.add_dense(8, ActivationFunction::ReLU);
        net.add_dense(2, ActivationFunction::Softmax);

        let input = vec![0.5, 0.3, 0.8, 0.2];
        let output = net.forward(&input);

        assert_eq!(output.len(), 2);
    }

    #[test]
    fn test_online_learning() {
        let mut net = NeuralNetwork::new(4, 2);
        net.add_dense(8, ActivationFunction::ReLU);
        net.add_dense(2, ActivationFunction::Softmax);

        let input = vec![0.5, 0.3, 0.8, 0.2];
        let target = vec![1.0, 0.0];

        let loss = net.online_train(&input, &target);

        assert!(loss >= 0.0);
        assert_eq!(net.learning_stats.total_updates, 1);
    }

    #[test]
    fn test_reservoir() {
        let mut reservoir = ReservoirLayer::new(4, 10, 0.3);
        let input = vec![0.5, 0.3, 0.8, 0.2];

        let output1 = reservoir.forward(&input);
        let output2 = reservoir.forward(&input);

        assert!(!output1.iter().all(|x| x == &output2[0]));
    }

    #[test]
    fn test_prediction() {
        let mut net = NeuralNetwork::new(4, 3);
        net.add_dense(8, ActivationFunction::ReLU);
        net.add_dense(3, ActivationFunction::Softmax);

        let input = vec![0.5, 0.3, 0.8, 0.2];
        let class = net.predict_class(&input);

        assert!(class < 3);
    }
}
