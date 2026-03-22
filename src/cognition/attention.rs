//! Attention System Module
//!
//! Implements selective attention mechanisms that focus cognitive resources
//! on the most relevant aspects of the sensory input. This improves efficiency
//! by reducing processing of irrelevant information.
//!
//! ## Attention Mechanisms
//!
//! - **Bottom-up attention**: Triggered by salient stimuli
//! - **Top-down attention**: Guided by current goals and drives
//! - **Feature-based attention**: Selects specific features for processing

use crate::homeostasis::DriveSystem;
use serde::{Deserialize, Serialize};

/// Attention System for selective focus
///
/// Implements a multi-factor attention mechanism that considers:
/// - Bottom-up saliency (inherent stimulus properties)
/// - Top-down goal alignment (current drives and objectives)
/// - Feature importance (learned relevance of input dimensions)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionSystem {
    /// Attention weights for each input dimension
    weights: Vec<f64>,

    /// Default attention weights
    default_weights: Vec<f64>,

    /// Current focus strength (0.0 to 2.0)
    focus_strength: f64,

    /// Input dimension size
    input_size: usize,

    /// Minimum attention weight
    min_weight: f64,

    /// Maximum attention weight
    max_weight: f64,

    /// Adaptation rate for weight updates
    adaptation_rate: f64,

    /// History of attention decisions
    decision_history: Vec<AttentionDecision>,

    /// Maximum history size
    history_capacity: usize,

    /// Global attention strength parameter
    strength: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AttentionDecision {
    input_pattern: Vec<f64>,
    weights_before: Vec<f64>,
    weights_after: Vec<f64>,
    drive_state: (f64, f64, f64),
}

impl AttentionSystem {
    /// Create a new attention system
    ///
    /// # Arguments
    /// * `input_size` - Number of input dimensions to attend to
    /// * `strength` - Global attention strength (0.0 to 2.0)
    pub fn new(input_size: usize, strength: f64) -> Self {
        let weights = vec![1.0; input_size];
        let default_weights = weights.clone();

        Self {
            weights,
            default_weights,
            focus_strength: 1.0,
            input_size,
            min_weight: 0.1,
            max_weight: 2.0,
            adaptation_rate: 0.01,
            decision_history: Vec::with_capacity(50),
            history_capacity: 50,
            strength: strength.clamp(0.1, 2.0),
        }
    }

    /// Focus attention on important aspects of the input
    ///
    /// Combines bottom-up saliency with top-down drive modulation
    /// to select the most relevant information for processing.
    ///
    /// # Arguments
    /// * `input` - Raw sensory input
    /// * `drives` - Current drive levels
    ///
    /// # Returns
    /// Attended (focused) input
    pub fn focus(&mut self, input: &[f64], drives: &DriveSystem) -> Vec<f64> {
        if input.is_empty() {
            return vec![];
        }

        let weights_before = self.weights.clone();

        let mut attended = Vec::with_capacity(input.len());

        for (i, &value) in input.iter().enumerate() {
            let weight_idx = i % self.weights.len();
            let base_weight = self.weights[weight_idx];

            let bottom_up_saliency = self.compute_saliency(value);

            let top_down_modulation = drives.get_attention_boost(i);

            let drive_influence = if i % 2 == 0 {
                drives.survival
            } else {
                drives.curiosity
            };

            let effective_weight = base_weight * bottom_up_saliency * top_down_modulation;

            let final_weight = effective_weight * (1.0 + drive_influence * 0.2);

            attended.push(value * final_weight * self.strength);
        }

        self.adapt_weights(input, &attended);

        let decision = AttentionDecision {
            input_pattern: input.to_vec(),
            weights_before,
            weights_after: self.weights.clone(),
            drive_state: (drives.survival, drives.curiosity, drives.efficiency),
        };

        self.decision_history.push(decision);
        if self.decision_history.len() > self.history_capacity {
            self.decision_history.remove(0);
        }

        attended
    }

    /// Compute bottom-up saliency of a value
    ///
    /// Values that deviate from the mean are considered more salient
    fn compute_saliency(&self, value: f64) -> f64 {
        let deviation = (value - 0.5).abs();
        1.0 + deviation
    }

    /// Adapt attention weights based on usefulness
    ///
    /// Weights are increased for attended values that lead to good outcomes
    /// and decreased for those that don't.
    fn adapt_weights(&mut self, original: &[f64], attended: &[f64]) {
        for i in 0..original.len().min(attended.len()) {
            let weight_idx = i % self.weights.len();

            let original_abs = original[i].abs();
            let attended_abs = attended[i].abs();

            if original_abs > 0.5 && attended_abs > original_abs * self.weights[weight_idx] {
                self.weights[weight_idx] *= 1.0 + self.adaptation_rate;
            } else if original_abs < 0.3 {
                self.weights[weight_idx] *= 1.0 - self.adaptation_rate * 0.5;
            }

            self.weights[weight_idx] =
                self.weights[weight_idx].clamp(self.min_weight, self.max_weight);
        }
    }

    /// Filter actions to suppress weak responses
    ///
    /// Actions with magnitude below threshold are set to zero
    /// to reduce noise in motor output.
    ///
    /// # Arguments
    /// * `actions` - Raw action values to filter
    pub fn filter_actions(&self, actions: &mut [f64]) {
        let threshold = 0.1 / self.strength;

        for action in actions.iter_mut() {
            if action.abs() < threshold {
                *action = 0.0;
            }
        }
    }

    /// Get current attention weights
    pub fn get_weights(&self) -> &[f64] {
        &self.weights
    }

    /// Set attention weights
    pub fn set_weights(&mut self, weights: Vec<f64>) {
        for (i, w) in weights.iter().enumerate().take(self.weights.len()) {
            self.weights[i] = w.clamp(self.min_weight, self.max_weight);
        }
    }

    /// Reset weights to default
    pub fn reset_weights(&mut self) {
        self.weights = self.default_weights.clone();
    }

    /// Reset the entire attention system
    pub fn reset(&mut self) {
        self.weights = self.default_weights.clone();
        self.focus_strength = 1.0;
        self.decision_history.clear();
    }

    /// Set focus strength
    ///
    /// Higher values mean more selective (narrower) attention
    pub fn set_focus_strength(&mut self, strength: f64) {
        self.focus_strength = strength.clamp(0.1, 2.0);
    }

    /// Get focus strength
    pub fn get_focus_strength(&self) -> f64 {
        self.focus_strength
    }

    /// Compute attention entropy (measure of spread)
    ///
    /// High entropy means diffuse attention, low entropy means focused
    pub fn attention_entropy(&self) -> f64 {
        let sum: f64 = self.weights.iter().map(|w| w.abs()).sum();
        if sum == 0.0 {
            return 0.0;
        }

        let probs: Vec<f64> = self.weights.iter().map(|w| w.abs() / sum).collect();

        -probs
            .iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| p * p.log2())
            .sum::<f64>()
    }

    /// Get the most attended input dimension
    pub fn get_top_attended(&self, n: usize) -> Vec<usize> {
        let mut indexed: Vec<(usize, f64)> = self
            .weights
            .iter()
            .enumerate()
            .map(|(i, &w)| (i, w))
            .collect();

        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        indexed.into_iter().take(n).map(|(i, _)| i).collect()
    }

    /// Boost attention to specific dimensions
    pub fn boost_attention(&mut self, dimensions: &[usize], factor: f64) {
        for &dim in dimensions {
            if dim < self.weights.len() {
                self.weights[dim] *= factor;
                self.weights[dim] = self.weights[dim].min(self.max_weight);
            }
        }
    }

    /// Suppress attention to specific dimensions
    pub fn suppress_attention(&mut self, dimensions: &[usize], factor: f64) {
        for &dim in dimensions {
            if dim < self.weights.len() {
                self.weights[dim] *= factor;
                self.weights[dim] = self.weights[dim].max(self.min_weight);
            }
        }
    }

    /// Get statistics about attention system
    pub fn get_stats(&self) -> AttentionStats {
        let weight_sum: f64 = self.weights.iter().sum();
        let weight_mean = weight_sum / self.weights.len().max(1) as f64;
        let weight_variance: f64 = self
            .weights
            .iter()
            .map(|&w| (w - weight_mean).powi(2))
            .sum::<f64>()
            / self.weights.len().max(1) as f64;

        AttentionStats {
            focus_strength: self.focus_strength,
            entropy: self.attention_entropy(),
            weight_mean,
            weight_std: weight_variance.sqrt(),
            decision_count: self.decision_history.len() as u64,
        }
    }
}

/// Attention system statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionStats {
    pub focus_strength: f64,
    pub entropy: f64,
    pub weight_mean: f64,
    pub weight_std: f64,
    pub decision_count: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attention_creation() {
        let attention = AttentionSystem::new(8, 1.0);
        assert_eq!(attention.input_size, 8);
    }

    #[test]
    fn test_focus() {
        let mut attention = AttentionSystem::new(8, 1.0);
        let drives = DriveSystem::new(1.0, 0.5, 0.3);
        let input = vec![0.5, 0.3, 0.8, 0.2, 0.6, 0.4, 0.7, 0.1];

        let attended = attention.focus(&input, &drives);

        assert_eq!(attended.len(), 8);
    }

    #[test]
    fn test_filter_actions() {
        let attention = AttentionSystem::new(4, 1.0);
        let mut actions = vec![0.05, 0.5, 0.08, 0.9];

        attention.filter_actions(&mut actions);

        assert_eq!(actions[0], 0.0);
        assert_eq!(actions[2], 0.0);
        assert!(actions[1] > 0.0);
        assert!(actions[3] > 0.0);
    }

    #[test]
    fn test_entropy() {
        let attention = AttentionSystem::new(4, 1.0);
        let entropy = attention.attention_entropy();

        assert!(entropy >= 0.0);
    }

    #[test]
    fn test_reset() {
        let mut attention = AttentionSystem::new(8, 1.5);
        let drives = DriveSystem::new(1.0, 0.5, 0.3);

        attention.focus(&vec![0.5; 8], &drives);
        attention.reset();

        assert_eq!(attention.focus_strength, 1.0);
    }

    #[test]
    fn test_get_top_attended() {
        let attention = AttentionSystem::new(8, 1.0);
        let top = attention.get_top_attended(3);

        assert_eq!(top.len(), 3);
    }
}
