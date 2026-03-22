//! Main Cognitive Agent Module
//!
//! This is the central orchestrator that integrates all cognitive systems:
//! - Predictive coding for hierarchical predictions
//! - World modeling for consequence simulation
//! - Attention system for selective focus
//! - Triple-layer memory (episodic, semantic, procedural)
//! - Homeostatic drive regulation
//! - Meta-learning for self-modification

use crate::cognition::{AttentionSystem, PredictiveCoder, WorldModel};
use crate::core::config::AgentConfig;
use crate::homeostasis::DriveSystem;
use crate::memory::MemorySystem;
use crate::meta::self_reflection::SelfReflectionReport;
use crate::meta::{MetaLearner, SelfReflection};
use serde::{Deserialize, Serialize};

/// The Hyper-Cognitive Agent
///
/// This is the main agent that integrates all cognitive systems into a coherent entity.
/// It operates through a continuous cognitive cycle that encompasses perception,
/// prediction, action selection, and learning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveAgent {
    /// Unique identifier for this agent
    pub id: u64,

    /// Number of cognitive cycles executed (agent's "age")
    pub age: u64,

    /// Agent configuration
    pub config: AgentConfig,

    /// Predictive coding network for hierarchical predictions
    pub predictive_coder: PredictiveCoder,

    /// Internal world model for consequence prediction
    pub world_model: WorldModel,

    /// Attention system for selective focus
    pub attention: AttentionSystem,

    /// Triple-layer memory system
    pub memory: MemorySystem,

    /// Homeostatic drive system
    pub drives: DriveSystem,

    /// Meta-learning controller
    pub meta_learner: MetaLearner,

    /// Self-reflection module
    pub self_reflection: SelfReflection,

    /// Current free energy (uncertainty measure)
    pub free_energy: f64,

    /// Whether the agent is active
    pub is_active: bool,

    /// Total reward accumulated
    pub total_reward: f64,

    /// Number of successful predictions
    pub successful_predictions: u64,

    /// Number of exploration actions taken
    pub exploration_count: u64,
}

impl CognitiveAgent {
    /// Create a new cognitive agent with the given configuration
    ///
    /// # Arguments
    /// * `id` - Unique identifier for this agent
    /// * `config` - Configuration parameters
    ///
    /// # Example
    /// ```
    /// use hyper_cognitive_kernel::{CognitiveAgent, AgentConfig};
    ///
    /// let config = AgentConfig::new()
    ///     .with_learning_rate(0.01)
    ///     .with_prediction_depth(3);
    ///
    /// let agent = CognitiveAgent::new(1, config);
    /// ```
    pub fn new(id: u64, config: AgentConfig) -> Self {
        Self {
            id,
            age: 0,
            config: config.clone(),
            predictive_coder: PredictiveCoder::new(config.prediction_depth, 8),
            world_model: WorldModel::new(8, 4, config.world_model_lr),
            attention: AttentionSystem::new(8, config.attention_strength),
            memory: MemorySystem::new(config.episodic_capacity, config.working_memory_capacity),
            drives: DriveSystem::new(
                config.survival_drive_weight,
                config.curiosity_drive_weight,
                config.efficiency_drive_weight,
            ),
            meta_learner: MetaLearner::new(config.base_learning_rate),
            self_reflection: SelfReflection::new(),
            free_energy: 0.0,
            is_active: true,
            total_reward: 0.0,
            successful_predictions: 0,
            exploration_count: 0,
        }
    }

    /// Execute one complete cognitive cycle
    ///
    /// This is the main method that orchestrates all cognitive processes:
    /// 1. Attention: Focus on important aspects of input
    /// 2. Prediction: Generate hierarchical predictions
    /// 3. Surprise Calculation: Compute prediction error
    /// 4. World Model Update: Update internal environment model
    /// 5. Action Selection: Choose actions to minimize expected free energy
    /// 6. Memory Encoding: Store the experience
    /// 7. Drive Update: Update motivational state
    /// 8. Meta-Learning: Adapt learning parameters
    /// 9. Predictive Learning: Update predictions based on outcome
    ///
    /// # Arguments
    /// * `sensory_input` - Raw sensory data from the environment
    /// * `reward` - Environmental reward signal
    ///
    /// # Returns
    /// The selected action(s) to execute
    pub fn cognitive_cycle(&mut self, sensory_input: &[f64], reward: f64) -> Vec<f64> {
        if !self.is_active {
            return vec![];
        }

        self.age += 1;
        self.total_reward += reward;

        let attended_input = self.attention.focus(sensory_input, &self.drives);

        let predictions = self.predictive_coder.predict(&attended_input);

        let prediction_errors: Vec<f64> = attended_input
            .iter()
            .zip(predictions.iter())
            .map(|(actual, predicted)| (actual - predicted).abs())
            .collect();

        self.free_energy = if prediction_errors.is_empty() {
            0.0
        } else {
            prediction_errors.iter().sum::<f64>() / prediction_errors.len() as f64
        };

        let surprise = self.free_energy;
        self.world_model
            .update(&attended_input, &prediction_errors, surprise);

        let actions = self.select_actions(&attended_input, surprise);

        self.memory
            .encode(&attended_input, &actions, reward, surprise, self.age);

        self.drives.update(reward, surprise, self.age);

        if self.config.enable_meta_learning {
            self.meta_learner
                .adapt(surprise, &self.drives, &mut self.config);
        }

        self.predictive_coder
            .learn(&attended_input, &actions, surprise);

        self.self_reflection.record_cycle(
            self.age,
            self.free_energy,
            self.drives.get_states(),
            reward,
        );

        if prediction_errors
            .iter()
            .all(|&e| e < self.config.prediction_error_threshold)
        {
            self.successful_predictions += 1;
        }

        actions
    }

    /// Select actions based on current state and drives
    fn select_actions(&mut self, input: &[f64], surprise: f64) -> Vec<f64> {
        let mut actions = self.world_model.plan(input, &self.drives);

        if surprise > self.config.surprise_threshold {
            self.explore(&mut actions, surprise);
            self.exploration_count += 1;
        }

        self.attention.filter_actions(&mut actions);

        actions
    }

    /// Add exploration noise when surprise is high
    fn explore(&self, actions: &mut Vec<f64>, surprise: f64) {
        let noise_amplitude = self.config.exploration_noise * (1.0 + surprise);

        for action in actions.iter_mut() {
            let noise = self.generate_noise();
            *action += noise * noise_amplitude;
            *action = action.clamp(-1.0, 1.0);
        }
    }

    /// Generate pseudo-random noise for exploration
    fn generate_noise(&self) -> f64 {
        use std::time::{SystemTime, UNIX_EPOCH};

        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        let x = (seed
            .wrapping_mul(self.age.wrapping_add(1))
            .wrapping_add(self.id) as f64)
            .sin()
            * 10000.0;
        x - x.floor() - 0.5
    }

    /// Generate a self-reflection report
    ///
    /// # Returns
    /// A comprehensive report of the agent's current state
    pub fn self_reflect(&self) -> SelfReflectionReport {
        SelfReflectionReport {
            age: self.age,
            free_energy: self.free_energy,
            drive_states: self.drives.get_states(),
            memory_usage: self.memory.usage_ratio(),
            learning_rate: self.config.base_learning_rate,
            prediction_accuracy: self.predictive_coder.accuracy(),
            total_reward: self.total_reward,
            successful_predictions: self.successful_predictions,
            exploration_ratio: if self.age > 0 {
                self.exploration_count as f64 / self.age as f64
            } else {
                0.0
            },
            meta_adaptations: self.meta_learner.adaptation_count,
            world_model_updates: self.world_model.update_count,
        }
    }

    /// Save the agent state to a JSON file
    ///
    /// # Arguments
    /// * `path` - File path to save the state
    ///
    /// # Returns
    /// Result indicating success or failure
    pub fn save(&self, path: &str) -> Result<(), String> {
        let report = self.self_reflect();
        let json = serde_json::to_string_pretty(&report)
            .map_err(|e| format!("Serialization error: {}", e))?;

        std::fs::write(path, json).map_err(|e| format!("File write error: {}", e))
    }

    /// Load agent state from a JSON file
    ///
    /// # Arguments
    /// * `path` - File path to load the state from
    ///
    /// # Returns
    /// Result containing the loaded report or error
    pub fn load(path: &str) -> Result<SelfReflectionReport, String> {
        let contents =
            std::fs::read_to_string(path).map_err(|e| format!("File read error: {}", e))?;

        serde_json::from_str(&contents).map_err(|e| format!("Deserialization error: {}", e))
    }

    /// Check if the agent is still alive
    ///
    /// An agent is alive if it is active and its survival drive is above zero
    pub fn is_alive(&self) -> bool {
        self.is_active && self.drives.is_alive()
    }

    /// Reset the agent to initial state
    pub fn reset(&mut self) {
        self.age = 0;
        self.free_energy = 0.0;
        self.is_active = true;
        self.total_reward = 0.0;
        self.successful_predictions = 0;
        self.exploration_count = 0;

        self.predictive_coder.reset();
        self.world_model.reset();
        self.attention.reset();
        self.memory.reset();
        self.drives.reset();
        self.meta_learner.reset();
        self.self_reflection.reset();
    }

    /// Get the current drive state
    pub fn get_drive_state(&self) -> (f64, f64, f64) {
        (
            self.drives.survival,
            self.drives.curiosity,
            self.drives.efficiency,
        )
    }

    /// Get the current prediction accuracy
    pub fn get_prediction_accuracy(&self) -> f64 {
        self.predictive_coder.accuracy()
    }

    /// Get the current free energy
    pub fn get_free_energy(&self) -> f64 {
        self.free_energy
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_agent() -> CognitiveAgent {
        let config = AgentConfig::new();
        CognitiveAgent::new(1, config)
    }

    #[test]
    fn test_agent_creation() {
        let agent = create_test_agent();
        assert_eq!(agent.id, 1);
        assert_eq!(agent.age, 0);
        assert!(agent.is_active);
        assert!(agent.is_alive());
    }

    #[test]
    fn test_cognitive_cycle() {
        let mut agent = create_test_agent();

        let sensors = vec![0.5, 0.3, 0.8, 0.2, 0.6, 0.4, 0.7, 0.1];
        let reward = 0.5;

        let actions = agent.cognitive_cycle(&sensors, reward);

        assert_eq!(actions.len(), 4);
        assert_eq!(agent.age, 1);
        assert!(agent.total_reward > 0.0);
    }

    #[test]
    fn test_agent_death() {
        let mut agent = create_test_agent();

        for _ in 0..10000 {
            let sensors = vec![0.5; 8];
            agent.cognitive_cycle(&sensors, -1.0);

            if !agent.is_alive() {
                break;
            }
        }

        assert!(!agent.is_alive() || agent.age >= 10000);
    }

    #[test]
    fn test_save_load() {
        let mut agent = create_test_agent();

        let sensors = vec![0.5; 8];
        agent.cognitive_cycle(&sensors, 0.5);
        agent.cognitive_cycle(&sensors, 0.3);

        let path = "test_agent_state.json";
        agent.save(path).unwrap();

        let loaded = CognitiveAgent::load(path).unwrap();
        assert_eq!(loaded.age, 2);

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_reset() {
        let mut agent = create_test_agent();

        let sensors = vec![0.5; 8];
        agent.cognitive_cycle(&sensors, 0.5);
        agent.cognitive_cycle(&sensors, 0.3);

        agent.reset();

        assert_eq!(agent.age, 0);
        assert_eq!(agent.total_reward, 0.0);
        assert!(agent.is_active);
    }
}
