//! Hierarchical Predictive Coding Module
//!
//! Implements a multi-layer predictive coding network where each layer makes
//! predictions about the layer below it. Prediction errors are propagated upward
//! and used to update predictions at each level.
//!
//! ## Theory
//!
//! Predictive coding is based on the principle that the brain is constantly
//! generating predictions about sensory inputs and only processes the "prediction error"
//! - the difference between expected and actual input.

use crate::utils::MathUtils;
use serde::{Deserialize, Serialize};

/// A single prediction layer in the hierarchy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionLayer {
    /// Neuron activations
    pub activations: Vec<f64>,

    /// Predictions for the layer below
    pub predictions: Vec<f64>,

    /// Prediction errors at this layer
    pub errors: Vec<f64>,

    /// Connection weights to layer below
    pub weights: Vec<Vec<f64>>,

    /// Bias values
    pub biases: Vec<f64>,

    /// Layer index in hierarchy
    pub level: usize,
}

impl PredictionLayer {
    /// Create a new prediction layer
    ///
    /// # Arguments
    /// * `size` - Number of neurons in this layer
    /// * `input_size` - Size of input from layer below
    /// * `level` - Hierarchy level (0 = bottom)
    pub fn new(size: usize, input_size: usize, level: usize) -> Self {
        let mut rng_seed: u64 = (level as u64 * 1000) + size as u64;

        let weights: Vec<Vec<f64>> = (0..size)
            .map(|_| {
                rng_seed = rng_seed.wrapping_mul(1103515245).wrapping_add(12345);
                (0..input_size)
                    .map(|_| {
                        rng_seed = rng_seed.wrapping_mul(1103515245).wrapping_add(12345);
                        MathUtils::xorshift_f64(rng_seed) * 0.5
                    })
                    .collect()
            })
            .collect();

        Self {
            activations: vec![0.0; size],
            predictions: vec![0.0; size],
            errors: vec![0.0; size],
            weights,
            biases: vec![0.0; size],
            level,
        }
    }

    /// Compute predictions from bottom-up input
    pub fn predict(&mut self, input: &[f64]) {
        for (i, activation) in self.activations.iter_mut().enumerate() {
            let mut sum = self.biases[i];

            for (j, &weight) in self.weights[i].iter().enumerate() {
                if j < input.len() {
                    sum += weight * input[j];
                }
            }

            *activation = MathUtils::sigmoid(sum);
            self.predictions[i] = *activation;
        }
    }

    /// Compute prediction errors
    ///
    /// # Arguments
    /// * `target` - Actual values to compare predictions against
    pub fn compute_errors(&mut self, target: &[f64]) {
        for (i, error) in self.errors.iter_mut().enumerate() {
            let predicted = self.predictions.get(i).copied().unwrap_or(0.0);
            let actual = target.get(i).copied().unwrap_or(0.0);
            *error = actual - predicted;
        }
    }

    /// Update weights based on prediction errors
    ///
    /// # Arguments
    /// * `learning_rate` - Speed of weight adjustment
    /// * `target` - Target values for bottom layer, predictions for others
    /// * `top_down_errors` - Error signal from layer above (if any)
    pub fn learn(&mut self, learning_rate: f64, target: &[f64], top_down_errors: Option<&[f64]>) {
        for (i, error) in self.errors.iter_mut().enumerate() {
            let predicted = self.predictions[i];
            let actual = target.get(i).copied().unwrap_or(0.0);
            *error = actual - predicted;

            let error_signal = if let Some(top_errors) = top_down_errors {
                let top_error = top_errors.get(i).copied().unwrap_or(0.0);
                *error + top_error * 0.5
            } else {
                *error
            };

            for (j, weight) in self.weights[i].iter_mut().enumerate() {
                if j < target.len() {
                    let input_val = target[j];
                    *weight += learning_rate * error_signal * input_val;
                    *weight = weight.clamp(-2.0, 2.0);
                }
            }

            self.biases[i] += learning_rate * error_signal;
            self.biases[i] = self.biases[i].clamp(-1.0, 1.0);
        }
    }

    /// Reset layer state
    pub fn reset(&mut self) {
        self.activations.fill(0.0);
        self.predictions.fill(0.0);
        self.errors.fill(0.0);
    }
}

/// Hierarchical Predictive Coding Network
///
/// This implements a multi-layer network where:
/// - Each layer predicts the activity of the layer below it
/// - Prediction errors are computed and propagated upward
/// - Top-down predictions provide context
/// - Bottom-up inputs provide sensory data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictiveCoder {
    /// The prediction layers, ordered from bottom (sensory) to top
    pub layers: Vec<PredictionLayer>,

    /// Number of layers in the hierarchy
    pub depth: usize,

    /// Total number of predictions made
    pub total_predictions: u64,

    /// Number of accurate predictions (error below threshold)
    pub accurate_predictions: u64,

    /// Size of input layer
    pub input_size: usize,

    /// Learning rate
    pub learning_rate: f64,

    /// Error threshold for counting as accurate
    pub accuracy_threshold: f64,
}

impl PredictiveCoder {
    /// Create a new predictive coder
    ///
    /// # Arguments
    /// * `depth` - Number of layers in the hierarchy
    /// * `input_size` - Size of sensory input
    pub fn new(depth: usize, input_size: usize) -> Self {
        let base_size = input_size;
        let size_reduction: u32 = 2;

        let mut layers = Vec::with_capacity(depth);

        for level in 0..depth {
            let size = (base_size / (size_reduction.pow(level as u32) as usize)).max(4);
            let input_sz = if level == 0 {
                input_size
            } else {
                (base_size / (size_reduction.pow((level - 1) as u32) as usize)).max(4)
            };
            layers.push(PredictionLayer::new(size, input_sz, level));
        }

        Self {
            layers,
            depth,
            total_predictions: 0,
            accurate_predictions: 0,
            input_size,
            learning_rate: 0.01,
            accuracy_threshold: 0.1,
        }
    }

    /// Generate predictions for the given input
    ///
    /// # Arguments
    /// * `input` - Sensory input vector
    ///
    /// # Returns
    /// Predictions for the input (from bottom layer)
    pub fn predict(&mut self, input: &[f64]) -> Vec<f64> {
        self.total_predictions += 1;

        if self.layers.is_empty() {
            return input.to_vec();
        }

        let mut current_input = input.to_vec();

        for layer in &mut self.layers {
            layer.predict(&current_input);
            current_input = layer.predictions.clone();
        }

        self.layers[0].predictions.clone()
    }

    /// Learn from prediction errors
    ///
    /// # Arguments
    /// * `actual` - Actual sensory values
    /// * `actions` - Actions taken (used for motor learning)
    /// * `surprise` - Prediction error magnitude
    pub fn learn(&mut self, actual: &[f64], _actions: &[f64], surprise: f64) {
        let adaptive_lr = self.learning_rate * (1.0 + surprise);

        let layer_count = self.layers.len();

        for i in 0..layer_count {
            let target: Vec<f64> = if i == 0 {
                actual.to_vec()
            } else {
                self.layers[i - 1].predictions.clone()
            };

            let top_down_errors: Option<Vec<f64>> = if i + 1 < layer_count {
                Some(self.layers[i + 1].errors.clone())
            } else {
                None
            };

            self.layers[i].learn(adaptive_lr, &target, top_down_errors.as_deref());
        }

        let avg_error: f64 = self.layers[0].errors.iter().map(|e| e.abs()).sum::<f64>()
            / self.layers[0].errors.len().max(1) as f64;

        if avg_error < self.accuracy_threshold {
            self.accurate_predictions += 1;
        }
    }

    /// Compute accuracy as ratio of accurate predictions
    pub fn accuracy(&self) -> f64 {
        if self.total_predictions == 0 {
            return 0.0;
        }
        self.accurate_predictions as f64 / self.total_predictions as f64
    }

    /// Get the current prediction error
    pub fn current_error(&self) -> f64 {
        if self.layers.is_empty() {
            return 0.0;
        }

        self.layers[0].errors.iter().map(|e| e.abs()).sum::<f64>()
            / self.layers[0].errors.len().max(1) as f64
    }

    /// Get prediction for a specific layer
    pub fn get_layer_predictions(&self, layer: usize) -> Option<&[f64]> {
        self.layers.get(layer).map(|l| l.predictions.as_slice())
    }

    /// Get errors for a specific layer
    pub fn get_layer_errors(&self, layer: usize) -> Option<&[f64]> {
        self.layers.get(layer).map(|l| l.errors.as_slice())
    }

    /// Reset all layers
    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            layer.reset();
        }
        self.total_predictions = 0;
        self.accurate_predictions = 0;
    }

    /// Update learning rate
    pub fn set_learning_rate(&mut self, lr: f64) {
        self.learning_rate = lr.clamp(0.001, 0.5);
    }

    /// Get the number of parameters (weights + biases)
    pub fn parameter_count(&self) -> usize {
        self.layers
            .iter()
            .map(|l| l.weights.iter().map(|w| w.len()).sum::<usize>() + l.biases.len())
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_predictive_coder_creation() {
        let coder = PredictiveCoder::new(3, 8);
        assert_eq!(coder.depth, 3);
        assert_eq!(coder.input_size, 8);
    }

    #[test]
    fn test_prediction() {
        let mut coder = PredictiveCoder::new(2, 4);
        let input = vec![0.5, 0.3, 0.8, 0.2];
        let predictions = coder.predict(&input);
        assert_eq!(predictions.len(), 4);
    }

    #[test]
    fn test_learning() {
        let mut coder = PredictiveCoder::new(2, 4);
        let input = vec![0.5, 0.3, 0.8, 0.2];
        let actions = vec![0.1, 0.2, 0.3, 0.4];

        coder.predict(&input);
        coder.learn(&input, &actions, 0.5);

        assert_eq!(coder.total_predictions, 1);
    }

    #[test]
    fn test_accuracy() {
        let mut coder = PredictiveCoder::new(2, 4);
        let input = vec![0.5; 4];

        for _ in 0..10 {
            coder.predict(&input);
            coder.learn(&input, &input, 0.1);
        }

        assert!(coder.total_predictions > 0);
    }

    #[test]
    fn test_reset() {
        let mut coder = PredictiveCoder::new(2, 4);
        let input = vec![0.5; 4];

        coder.predict(&input);
        coder.learn(&input, &input, 0.1);

        coder.reset();

        assert_eq!(coder.total_predictions, 0);
        assert_eq!(coder.accurate_predictions, 0);
    }
}
