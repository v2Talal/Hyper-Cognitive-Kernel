//! Agent Configuration Module
//!
//! Defines the configuration parameters that control the behavior of the cognitive agent.
//! All parameters are tunable and can be adjusted at runtime for different scenarios.

use serde::{Deserialize, Serialize};

/// Configuration for the Cognitive Agent
///
/// This struct contains all tunable parameters that control the agent's behavior,
/// learning, memory, and decision-making processes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    /// Base learning rate for weight updates
    /// Range: 0.001 to 0.1
    /// Default: 0.01
    pub base_learning_rate: f64,

    /// Working memory decay rate
    /// Controls how quickly short-term memories fade
    /// Range: 0.0 to 0.1
    /// Default: 0.001
    pub working_memory_decay: f64,

    /// Depth of hierarchical predictions
    /// Higher values allow for more complex predictions but increase computation
    /// Range: 1 to 10
    /// Default: 3
    pub prediction_depth: usize,

    /// Maximum capacity of episodic memory
    /// Number of experiences that can be stored before forgetting
    /// Default: 1000
    pub episodic_capacity: usize,

    /// Surprise threshold for triggering exploration
    /// When prediction error exceeds this, the agent explores more
    /// Range: 0.0 to 1.0
    /// Default: 0.5
    pub surprise_threshold: f64,

    /// Weight of the survival drive in decision making
    /// Range: 0.0 to 2.0
    /// Default: 1.0
    pub survival_drive_weight: f64,

    /// Weight of the curiosity drive in decision making
    /// Range: 0.0 to 2.0
    /// Default: 0.5
    pub curiosity_drive_weight: f64,

    /// Weight of the efficiency drive in decision making
    /// Range: 0.0 to 2.0
    /// Default: 0.3
    pub efficiency_drive_weight: f64,

    /// Maximum number of working memory items
    /// Default: 100
    pub working_memory_capacity: usize,

    /// Exploration noise amplitude
    /// Range: 0.0 to 1.0
    /// Default: 0.1
    pub exploration_noise: f64,

    /// Attention mechanism strength
    /// Range: 0.0 to 2.0
    /// Default: 1.0
    pub attention_strength: f64,

    /// World model update learning rate
    /// Range: 0.0 to 0.5
    /// Default: 0.05
    pub world_model_lr: f64,

    /// Minimum prediction error to trigger learning
    /// Range: 0.0 to 1.0
    /// Default: 0.1
    pub prediction_error_threshold: f64,

    /// Enable meta-learning adaptations
    /// Default: true
    pub enable_meta_learning: bool,

    /// Enable self-reflection reports
    /// Default: true
    pub enable_self_reflection: bool,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            base_learning_rate: 0.01,
            working_memory_decay: 0.001,
            prediction_depth: 3,
            episodic_capacity: 1000,
            surprise_threshold: 0.5,
            survival_drive_weight: 1.0,
            curiosity_drive_weight: 0.5,
            efficiency_drive_weight: 0.3,
            working_memory_capacity: 100,
            exploration_noise: 0.1,
            attention_strength: 1.0,
            world_model_lr: 0.05,
            prediction_error_threshold: 0.1,
            enable_meta_learning: true,
            enable_self_reflection: true,
        }
    }
}

impl AgentConfig {
    /// Create a new configuration with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the base learning rate
    ///
    /// # Arguments
    /// * `rate` - Learning rate value (clamped to 0.001-0.1)
    pub fn with_learning_rate(mut self, rate: f64) -> Self {
        self.base_learning_rate = rate.clamp(0.001, 0.1);
        self
    }

    /// Set the prediction depth
    ///
    /// # Arguments
    /// * `depth` - Number of hierarchical layers (clamped to 1-10)
    pub fn with_prediction_depth(mut self, depth: usize) -> Self {
        self.prediction_depth = depth.clamp(1, 10);
        self
    }

    /// Set the surprise threshold
    ///
    /// # Arguments
    /// * `threshold` - Threshold value (clamped to 0.0-1.0)
    pub fn with_surprise_threshold(mut self, threshold: f64) -> Self {
        self.surprise_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Set the episodic memory capacity
    ///
    /// # Arguments
    /// * `capacity` - Maximum number of stored episodes
    pub fn with_episodic_capacity(mut self, capacity: usize) -> Self {
        self.episodic_capacity = capacity.max(10);
        self
    }

    /// Set drive weights for all drives
    ///
    /// # Arguments
    /// * `survival` - Survival drive weight
    /// * `curiosity` - Curiosity drive weight
    /// * `efficiency` - Efficiency drive weight
    pub fn with_drive_weights(mut self, survival: f64, curiosity: f64, efficiency: f64) -> Self {
        self.survival_drive_weight = survival.clamp(0.0, 2.0);
        self.curiosity_drive_weight = curiosity.clamp(0.0, 2.0);
        self.efficiency_drive_weight = efficiency.clamp(0.0, 2.0);
        self
    }

    /// Enable or disable meta-learning
    pub fn with_meta_learning(mut self, enabled: bool) -> Self {
        self.enable_meta_learning = enabled;
        self
    }

    /// Set the exploration noise level
    pub fn with_exploration_noise(mut self, noise: f64) -> Self {
        self.exploration_noise = noise.clamp(0.0, 1.0);
        self
    }

    /// Validate the configuration
    ///
    /// Returns an error message if any parameter is invalid
    pub fn validate(&self) -> Result<(), String> {
        if !(0.001..=0.1).contains(&self.base_learning_rate) {
            return Err("base_learning_rate must be between 0.001 and 0.1".to_string());
        }
        if !(1..=10).contains(&self.prediction_depth) {
            return Err("prediction_depth must be between 1 and 10".to_string());
        }
        if self.episodic_capacity < 10 {
            return Err("episodic_capacity must be at least 10".to_string());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = AgentConfig::default();
        assert_eq!(config.base_learning_rate, 0.01);
        assert_eq!(config.prediction_depth, 3);
        assert_eq!(config.episodic_capacity, 1000);
    }

    #[test]
    fn test_with_learning_rate() {
        let config = AgentConfig::new().with_learning_rate(0.05);
        assert_eq!(config.base_learning_rate, 0.05);

        let config = AgentConfig::new().with_learning_rate(5.0);
        assert_eq!(config.base_learning_rate, 0.1);

        let config = AgentConfig::new().with_learning_rate(0.0);
        assert_eq!(config.base_learning_rate, 0.001);
    }

    #[test]
    fn test_with_prediction_depth() {
        let config = AgentConfig::new().with_prediction_depth(5);
        assert_eq!(config.prediction_depth, 5);

        let config = AgentConfig::new().with_prediction_depth(0);
        assert_eq!(config.prediction_depth, 1);

        let config = AgentConfig::new().with_prediction_depth(20);
        assert_eq!(config.prediction_depth, 10);
    }

    #[test]
    fn test_validation() {
        let config = AgentConfig::default();
        assert!(config.validate().is_ok());
    }
}
