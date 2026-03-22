//! Meta-Learning Controller Module
//!
//! Implements self-modifying learning that adjusts the agent's learning
//! parameters based on performance feedback. This enables the agent to
//! learn how to learn more effectively.
//!
//! ## Adaptation Strategies
//!
//! - **Learning Rate**: Adjust based on error magnitude
//! - **Exploration**: Modify based on uncertainty
//! - **Memory**: Adapt based on recall success
//! - **Attention**: Adjust focus based on task demands

use crate::core::config::AgentConfig;
use crate::homeostasis::DriveSystem;
use serde::{Deserialize, Serialize};

/// Meta-Learning Controller
///
/// Analyzes agent performance and adapts learning parameters to optimize
/// learning efficiency over time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaLearner {
    /// Base learning rate
    pub base_rate: f64,

    /// Current effective learning rate
    current_rate: f64,

    /// History of surprise values for trend analysis
    surprise_history: Vec<f64>,

    /// History of learning rate adaptations
    adaptation_history: Vec<f64>,

    /// Performance trend (positive = improving)
    performance_trend: f64,

    /// Number of adaptations performed
    pub adaptation_count: u64,

    /// Maximum history length
    history_capacity: usize,

    /// Learning rate bounds
    min_lr: f64,
    max_lr: f64,

    /// Momentum for adaptation smoothing
    adaptation_momentum: f64,

    /// Current adaptation direction
    current_adaptation: f64,
}

impl MetaLearner {
    /// Create a new meta-learner
    ///
    /// # Arguments
    /// * `base_rate` - Initial learning rate
    pub fn new(base_rate: f64) -> Self {
        Self {
            base_rate,
            current_rate: base_rate,
            surprise_history: Vec::with_capacity(100),
            adaptation_history: Vec::with_capacity(100),
            performance_trend: 0.0,
            adaptation_count: 0,
            history_capacity: 100,
            min_lr: 0.001,
            max_lr: 0.1,
            adaptation_momentum: 0.9,
            current_adaptation: 0.0,
        }
    }

    /// Adapt learning parameters based on current state
    ///
    /// This is the main meta-learning function that analyzes performance
    /// and adjusts learning parameters accordingly.
    ///
    /// # Arguments
    /// * `surprise` - Current prediction error (surprise level)
    /// * `drives` - Current drive states
    /// * `config` - Agent configuration to modify
    pub fn adapt(&mut self, surprise: f64, drives: &DriveSystem, config: &mut AgentConfig) {
        self.surprise_history.push(surprise);

        if self.surprise_history.len() > self.history_capacity {
            self.surprise_history.remove(0);
        }

        if self.surprise_history.len() >= 20 {
            self.update_performance_trend();
        }

        self.adapt_learning_rate(surprise, config, drives);

        self.adapt_surprise_threshold(drives);

        self.adapt_exploration(surprise, config);

        self.adaptation_count += 1;
    }

    /// Update the performance trend based on recent history
    fn update_performance_trend(&mut self) {
        if self.surprise_history.len() < 20 {
            return;
        }

        let recent: Vec<_> = self.surprise_history.iter().rev().take(10).collect();
        let older: Vec<_> = self
            .surprise_history
            .iter()
            .rev()
            .skip(10)
            .take(10)
            .collect();

        if recent.is_empty() || older.is_empty() {
            return;
        }

        let recent_avg: f64 = recent.iter().map(|&&v| v).sum::<f64>() / recent.len() as f64;
        let older_avg: f64 = older.iter().map(|&&v| v).sum::<f64>() / older.len() as f64;

        self.performance_trend = older_avg - recent_avg;
    }

    /// Adapt the learning rate based on surprise and performance
    fn adapt_learning_rate(
        &mut self,
        surprise: f64,
        config: &mut AgentConfig,
        drives: &DriveSystem,
    ) {
        let mut adaptation: f64;

        if surprise > 0.7 {
            adaptation = 1.1;
        } else if surprise > 0.5 {
            adaptation = 1.05;
        } else if self.performance_trend > 0.1 {
            adaptation = 0.99;
        } else if self.performance_trend < -0.1 {
            adaptation = 1.05;
        } else {
            adaptation = 1.0;
        }

        if drives.get_primary_drive() < 0.3 {
            adaptation *= 1.2;
        }

        self.current_adaptation = self.adaptation_momentum * self.current_adaptation
            + (1.0 - self.adaptation_momentum) * (adaptation - 1.0);

        let new_rate = config.base_learning_rate * (1.0 + self.current_adaptation);
        config.base_learning_rate = new_rate.clamp(self.min_lr, self.max_lr);

        self.adaptation_history.push(self.current_adaptation);

        if self.adaptation_history.len() > self.history_capacity {
            self.adaptation_history.remove(0);
        }
    }

    /// Adapt surprise threshold based on drive states
    fn adapt_surprise_threshold(&mut self, drives: &DriveSystem) {
        self.min_lr = if drives.curiosity > 0.7 { 0.005 } else { 0.001 };
    }

    /// Adapt exploration parameters
    fn adapt_exploration(&mut self, surprise: f64, config: &mut AgentConfig) {
        if surprise > 0.6 {
            config.exploration_noise = (config.exploration_noise * 1.1).min(1.0);
        } else if surprise < 0.2 && self.performance_trend > 0.05 {
            config.exploration_noise = (config.exploration_noise * 0.95).max(0.01);
        }
    }

    /// Get current effective learning rate
    pub fn get_current_rate(&self) -> f64 {
        self.current_rate
    }

    /// Get performance trend
    pub fn get_performance_trend(&self) -> f64 {
        self.performance_trend
    }

    /// Get average surprise over recent history
    pub fn get_average_surprise(&self, window: usize) -> f64 {
        let window = window.min(self.surprise_history.len());
        if window == 0 {
            return 0.0;
        }

        self.surprise_history.iter().rev().take(window).sum::<f64>() / window as f64
    }

    /// Check if learning is improving
    pub fn is_improving(&self) -> bool {
        self.performance_trend > 0.05
    }

    /// Check if learning is degrading
    pub fn is_degrading(&self) -> bool {
        self.performance_trend < -0.05
    }

    /// Get adaptation statistics
    pub fn get_stats(&self) -> MetaStats {
        MetaStats {
            current_lr: self.current_rate,
            performance_trend: self.performance_trend,
            adaptation_count: self.adaptation_count,
            avg_surprise: self.get_average_surprise(50),
            is_improving: self.is_improving(),
            is_degrading: self.is_degrading(),
            recent_adaptations: self.adaptation_history.len(),
        }
    }

    /// Reset the meta-learner
    pub fn reset(&mut self) {
        self.surprise_history.clear();
        self.adaptation_history.clear();
        self.performance_trend = 0.0;
        self.adaptation_count = 0;
        self.current_adaptation = 0.0;
    }

    /// Set learning rate bounds
    pub fn set_lr_bounds(&mut self, min: f64, max: f64) {
        self.min_lr = min.clamp(0.0001, 0.01);
        self.max_lr = max.clamp(0.05, 0.5);
    }
}

/// Meta-learner statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaStats {
    pub current_lr: f64,
    pub performance_trend: f64,
    pub adaptation_count: u64,
    pub avg_surprise: f64,
    pub is_improving: bool,
    pub is_degrading: bool,
    pub recent_adaptations: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_meta_learner() -> MetaLearner {
        MetaLearner::new(0.01)
    }

    fn create_test_drives() -> DriveSystem {
        DriveSystem::new(1.0, 0.5, 0.3)
    }

    #[test]
    fn test_meta_learner_creation() {
        let ml = create_test_meta_learner();
        assert_eq!(ml.base_rate, 0.01);
        assert_eq!(ml.adaptation_count, 0);
    }

    #[test]
    fn test_adaptation() {
        let mut ml = create_test_meta_learner();
        let mut config = AgentConfig::new();
        let drives = create_test_drives();

        ml.adapt(0.5, &drives, &mut config);

        assert_eq!(ml.adaptation_count, 1);
    }

    #[test]
    fn test_high_surprise_adaptation() {
        let mut ml = create_test_meta_learner();
        let mut config = AgentConfig::new();
        let drives = create_test_drives();

        let initial_lr = config.base_learning_rate;
        ml.adapt(0.8, &drives, &mut config);

        assert!(config.base_learning_rate >= initial_lr);
    }

    #[test]
    fn test_performance_trend() {
        let mut ml = create_test_meta_learner();
        let mut config = AgentConfig::new();
        let drives = create_test_drives();

        for i in 0..30 {
            let surprise = if i < 15 { 0.8 } else { 0.3 };
            ml.adapt(surprise, &drives, &mut config);
        }

        assert!(ml.performance_trend > 0.0);
    }

    #[test]
    fn test_get_average_surprise() {
        let mut ml = create_test_meta_learner();
        let mut config = AgentConfig::new();
        let drives = create_test_drives();

        for _ in 0..20 {
            ml.adapt(0.5, &drives, &mut config);
        }

        let avg = ml.get_average_surprise(10);
        assert!((avg - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_reset() {
        let mut ml = create_test_meta_learner();
        let mut config = AgentConfig::new();
        let drives = create_test_drives();

        ml.adapt(0.5, &drives, &mut config);
        ml.reset();

        assert_eq!(ml.adaptation_count, 0);
        assert!(ml.surprise_history.is_empty());
    }

    #[test]
    fn test_lr_bounds() {
        let mut ml = create_test_meta_learner();
        ml.set_lr_bounds(0.005, 0.05);

        assert_eq!(ml.min_lr, 0.005);
        assert_eq!(ml.max_lr, 0.05);
    }
}
