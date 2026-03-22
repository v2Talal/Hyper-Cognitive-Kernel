//! World Model Module
//!
//! Implements an internal model of the environment that simulates consequences
//! of actions before they are executed. This allows the agent to plan and
//! predict outcomes without costly real-world experimentation.
//!
//! ## Components
//!
//! - **Transition Model**: Learns how states evolve with actions
//! - **Reward Model**: Predicts expected rewards for state-action pairs
//! - **Planning**: Uses learned models to select optimal actions

use crate::homeostasis::DriveSystem;
use serde::{Deserialize, Serialize};

/// Internal model of the environment
///
/// Learns the transition dynamics and reward structure of the environment
/// through experience.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldModel {
    /// Transition probability matrix: transition[i][j] = P(next_state=j | current_state=i, action)
    /// Simplified as deterministic with noise for efficiency
    transition_matrix: Vec<Vec<f64>>,

    /// Reward predictions for state-action pairs
    reward_model: Vec<f64>,

    /// State representation size
    state_size: usize,

    /// Action space size
    action_size: usize,

    /// Learning rate for model updates
    learning_rate: f64,

    /// Number of updates performed
    pub update_count: u64,

    /// Model uncertainty (epistemic uncertainty)
    uncertainty: f64,

    /// Prediction history for uncertainty estimation
    prediction_history: Vec<f64>,

    /// Maximum history length
    history_capacity: usize,
}

impl WorldModel {
    /// Create a new world model
    ///
    /// # Arguments
    /// * `state_size` - Dimension of state representation
    /// * `action_size` - Number of possible actions
    /// * `learning_rate` - Rate at which the model learns
    pub fn new(state_size: usize, action_size: usize, learning_rate: f64) -> Self {
        let total_size = state_size * action_size;

        let mut rng_seed: u64 = (state_size as u64)
            .wrapping_mul(action_size as u64)
            .wrapping_add(42);

        let transition_matrix: Vec<Vec<f64>> = (0..state_size)
            .map(|i| {
                let mut row = Vec::with_capacity(action_size);
                let base = i as f64 / state_size as f64;

                for _ in 0..action_size {
                    rng_seed = rng_seed.wrapping_mul(1103515245).wrapping_add(12345);
                    let val = base + MathUtils::xorshift_f64(rng_seed) * 0.1;
                    row.push(val);
                }

                let sum: f64 = row.iter().sum();
                for val in row.iter_mut() {
                    *val /= sum;
                }

                row
            })
            .collect();

        Self {
            transition_matrix,
            reward_model: vec![0.0; total_size],
            state_size,
            action_size,
            learning_rate,
            update_count: 0,
            uncertainty: 0.5,
            prediction_history: Vec::with_capacity(100),
            history_capacity: 100,
        }
    }

    /// Update the world model based on observed experience
    ///
    /// # Arguments
    /// * `state` - Current state observation
    /// * `errors` - Prediction errors from sensory processing
    /// * `surprise` - Magnitude of surprise (prediction error)
    pub fn update(&mut self, state: &[f64], errors: &[f64], surprise: f64) {
        if state.is_empty() || errors.is_empty() {
            return;
        }

        let adaptive_lr = self.learning_rate * (1.0 + surprise);

        for (i, state_val) in state.iter().enumerate().take(self.state_size) {
            let idx_i = i % self.state_size;

            for j in 0..self.action_size {
                if i < self.transition_matrix.len() && j < self.transition_matrix[i].len() {
                    let error_contribution = errors.get(i % errors.len()).copied().unwrap_or(0.0);

                    self.transition_matrix[idx_i][j] +=
                        adaptive_lr * error_contribution * state_val;

                    let max_val = 1.0;
                    let min_val = -1.0;
                    self.transition_matrix[idx_i][j] =
                        self.transition_matrix[idx_i][j].clamp(min_val, max_val);
                }
            }
        }

        self.update_count += 1;

        let avg_error = errors.iter().map(|e| e.abs()).sum::<f64>() / errors.len().max(1) as f64;
        self.prediction_history.push(avg_error);

        if self.prediction_history.len() > self.history_capacity {
            self.prediction_history.remove(0);
        }

        self.uncertainty = self.estimate_uncertainty();
    }

    /// Estimate model uncertainty based on prediction history
    fn estimate_uncertainty(&self) -> f64 {
        if self.prediction_history.len() < 2 {
            return 0.5;
        }

        let mean: f64 =
            self.prediction_history.iter().sum::<f64>() / self.prediction_history.len() as f64;
        let variance: f64 = self
            .prediction_history
            .iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>()
            / self.prediction_history.len() as f64;

        variance.sqrt().min(1.0)
    }

    /// Plan actions to minimize expected free energy
    ///
    /// Uses the learned model to evaluate potential actions and selects
    /// the one expected to lead to the best outcomes.
    ///
    /// # Arguments
    /// * `state` - Current state
    /// * `drives` - Current drive levels for motivation
    ///
    /// # Returns
    /// Selected action values
    pub fn plan(&self, state: &[f64], drives: &DriveSystem) -> Vec<f64> {
        let mut actions = vec![0.0; self.action_size];

        for (i, action) in actions.iter_mut().enumerate() {
            let mut expected_value = 0.0;
            let mut expected_free_energy = 0.0;

            for (j, &state_val) in state.iter().enumerate().take(self.state_size) {
                if j < self.transition_matrix.len() && i < self.transition_matrix[j].len() {
                    let transition = self.transition_matrix[j][i];
                    expected_value += state_val * transition;

                    let reward_idx = j * self.action_size + i;
                    let predicted_reward =
                        self.reward_model.get(reward_idx).copied().unwrap_or(0.0);
                    expected_free_energy -= predicted_reward;
                }
            }

            let drive_modulation = drives.get_modulation(i);
            *action = expected_value * drive_modulation;

            let uncertainty_bonus = self.uncertainty * 0.1;
            if uncertainty_bonus > 0.05 {
                *action += uncertainty_bonus * drives.curiosity;
            }
        }

        let sum: f64 = actions.iter().map(|a| a.abs()).sum();
        if sum > 0.0 {
            for action in actions.iter_mut() {
                *action /= sum;
            }
        }

        actions
    }

    /// Update reward model based on observed rewards
    ///
    /// # Arguments
    /// * `state_idx` - Current state index
    /// * `action_idx` - Action index
    /// * `reward` - Observed reward
    pub fn update_reward(&mut self, state_idx: usize, action_idx: usize, reward: f64) {
        if state_idx >= self.state_size || action_idx >= self.action_size {
            return;
        }

        let idx = state_idx * self.action_size + action_idx;

        let count = (self.update_count as f64).min(100.0) + 1.0;
        let alpha = 1.0 / count;

        self.reward_model[idx] += alpha * (reward - self.reward_model[idx]);
    }

    /// Predict next state given current state and action
    ///
    /// # Arguments
    /// * `state` - Current state
    /// * `action` - Action index
    ///
    /// # Returns
    /// Predicted next state
    pub fn predict_next_state(&self, state: &[f64], action: usize) -> Vec<f64> {
        let mut next_state = vec![0.0; self.state_size];

        for (i, &state_val) in state.iter().enumerate().take(self.state_size) {
            if i < self.transition_matrix.len() && action < self.transition_matrix[i].len() {
                let transition = self.transition_matrix[i][action];
                next_state[i] = state_val * transition;
            }
        }

        next_state
    }

    /// Get expected reward for a state-action pair
    pub fn get_expected_reward(&self, state_idx: usize, action_idx: usize) -> f64 {
        if state_idx >= self.state_size || action_idx >= self.action_size {
            return 0.0;
        }

        let idx = state_idx * self.action_size + action_idx;
        self.reward_model.get(idx).copied().unwrap_or(0.0)
    }

    /// Get current model uncertainty
    pub fn get_uncertainty(&self) -> f64 {
        self.uncertainty
    }

    /// Reset the model
    pub fn reset(&mut self) {
        for row in &mut self.transition_matrix {
            for val in row.iter_mut() {
                *val = 0.0;
            }
        }
        self.reward_model.fill(0.0);
        self.update_count = 0;
        self.uncertainty = 0.5;
        self.prediction_history.clear();
    }

    /// Get model statistics
    pub fn get_stats(&self) -> WorldModelStats {
        WorldModelStats {
            update_count: self.update_count,
            uncertainty: self.uncertainty,
            transition_mean: self
                .transition_matrix
                .iter()
                .flat_map(|r| r.iter())
                .sum::<f64>()
                / (self.state_size * self.action_size).max(1) as f64,
            reward_mean: self.reward_model.iter().sum::<f64>()
                / self.reward_model.len().max(1) as f64,
        }
    }
}

/// Statistics about the world model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldModelStats {
    pub update_count: u64,
    pub uncertainty: f64,
    pub transition_mean: f64,
    pub reward_mean: f64,
}

/// Math utilities used by world model
mod MathUtils {
    /// Sigmoid activation function
    pub fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    /// XORSHIFT random number generator for f64
    pub fn xorshift_f64(seed: u64) -> f64 {
        let mut x = seed;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;

        let normalized = (x as i64).abs() as f64 / (i64::MAX as f64);
        normalized * 2.0 - 1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_world_model() -> WorldModel {
        WorldModel::new(8, 4, 0.05)
    }

    #[test]
    fn test_world_model_creation() {
        let model = create_test_world_model();
        assert_eq!(model.state_size, 8);
        assert_eq!(model.action_size, 4);
    }

    #[test]
    fn test_update() {
        let mut model = create_test_world_model();
        let state = vec![0.5; 8];
        let errors = vec![0.1; 8];

        model.update(&state, &errors, 0.3);

        assert_eq!(model.update_count, 1);
    }

    #[test]
    fn test_planning() {
        let model = create_test_world_model();
        let state = vec![0.5; 8];
        let drives = DriveSystem::new(1.0, 0.5, 0.3);

        let actions = model.plan(&state, &drives);

        assert_eq!(actions.len(), 4);
    }

    #[test]
    fn test_uncertainty() {
        let mut model = create_test_world_model();

        assert_eq!(model.get_uncertainty(), 0.5);

        let state = vec![0.5; 8];
        let errors = vec![0.2; 8];

        for _ in 0..20 {
            model.update(&state, &errors, 0.5);
        }

        assert!(model.get_uncertainty() >= 0.0);
    }

    #[test]
    fn test_reset() {
        let mut model = create_test_world_model();
        let state = vec![0.5; 8];
        let errors = vec![0.1; 8];

        model.update(&state, &errors, 0.3);
        model.reset();

        assert_eq!(model.update_count, 0);
    }
}
