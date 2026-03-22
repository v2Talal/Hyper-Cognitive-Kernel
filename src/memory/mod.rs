//! Memory Module
//!
//! Implements a triple-layer memory system inspired by cognitive science:
//! - **Episodic Memory**: Stores specific experiences with temporal context
//! - **Semantic Memory**: Stores abstract patterns and conceptual knowledge
//! - **Procedural Memory**: Stores skills and learned action sequences
//!
//! This architecture enables efficient learning while preventing catastrophic forgetting.

pub mod episodic;
pub mod procedural;
pub mod semantic;

pub use episodic::{Episode, EpisodicMemory};
pub use procedural::{ProceduralMemory, Skill};
pub use semantic::{Pattern, SemanticMemory};
use serde::{Deserialize, Serialize};

/// Integrated Memory System
///
/// Combines episodic, semantic, and procedural memory into a unified system
/// with automatic transfer between layers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySystem {
    /// Episodic memory for experiences
    pub episodic: EpisodicMemory,

    /// Semantic memory for patterns
    pub semantic: SemanticMemory,

    /// Procedural memory for skills
    pub procedural: ProceduralMemory,

    /// Whether to enable memory consolidation
    consolidation_enabled: bool,

    /// Consolidation threshold (episodes before semantic extraction)
    consolidation_threshold: usize,
}

impl MemorySystem {
    /// Create a new memory system
    ///
    /// # Arguments
    /// * `episodic_capacity` - Maximum number of episodes to store
    /// * `working_memory_capacity` - Working memory size for procedural memory
    pub fn new(episodic_capacity: usize, working_memory_capacity: usize) -> Self {
        Self {
            episodic: EpisodicMemory::new(episodic_capacity),
            semantic: SemanticMemory::new(),
            procedural: ProceduralMemory::new(working_memory_capacity),
            consolidation_enabled: true,
            consolidation_threshold: 10,
        }
    }

    /// Encode a new experience into memory
    ///
    /// Automatically handles:
    /// - Episodic encoding with temporal tags
    /// - Pattern extraction to semantic memory
    /// - Skill reinforcement in procedural memory
    ///
    /// # Arguments
    /// * `input` - Sensory input
    /// * `actions` - Actions taken
    /// * `reward` - Reward received
    /// * `surprise` - Prediction error magnitude
    /// * `timestamp` - Current time/cycle
    pub fn encode(
        &mut self,
        input: &[f64],
        actions: &[f64],
        reward: f64,
        surprise: f64,
        timestamp: u64,
    ) {
        self.episodic
            .store(input, actions, reward, surprise, timestamp);

        if self.consolidation_enabled
            && self.episodic.episodes.len() >= self.consolidation_threshold
        {
            if self.episodic.episodes.len() % self.consolidation_threshold == 0 {
                self.semantic.extract_patterns(&self.episodic);
            }
        }

        self.procedural.reinforce(actions, reward);

        if surprise > 0.7 {
            self.episodic.importance_boost(timestamp);
        }
    }

    /// Retrieve similar past experiences
    ///
    /// # Arguments
    /// * `current_input` - Current sensory state
    /// * `threshold` - Similarity threshold (0.0 to 1.0)
    ///
    /// # Returns
    /// Vector of similar episodes
    pub fn retrieve_similar(&self, current_input: &[f64], threshold: f64) -> Vec<&Episode> {
        self.episodic.retrieve_similar(current_input, threshold)
    }

    /// Get the most rewarding past action
    pub fn get_best_remembered_action(&self) -> Option<usize> {
        self.episodic
            .get_best_episode()
            .map(|e| self.find_best_action_in_episode(e))
    }

    /// Find best action in an episode based on reward
    fn find_best_action_in_episode(&self, episode: &Episode) -> usize {
        episode
            .actions
            .iter()
            .enumerate()
            .max_by(|(_, &a), (_, &b)| a.partial_cmp(&b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Calculate memory usage ratio
    pub fn usage_ratio(&self) -> f64 {
        self.episodic.usage_ratio()
    }

    /// Get total memory items
    pub fn total_items(&self) -> usize {
        self.episodic.episodes.len() + self.semantic.patterns.len() + self.procedural.skills.len()
    }

    /// Clear all memories
    pub fn clear(&mut self) {
        self.episodic.clear();
        self.semantic.clear();
        self.procedural.clear();
    }

    /// Reset memory system
    pub fn reset(&mut self) {
        self.episodic.reset();
        self.semantic.reset();
        self.procedural.reset();
    }

    /// Enable/disable consolidation
    pub fn set_consolidation(&mut self, enabled: bool) {
        self.consolidation_enabled = enabled;
    }

    /// Get memory statistics
    pub fn get_stats(&self) -> MemoryStats {
        MemoryStats {
            episodic_count: self.episodic.episodes.len(),
            episodic_capacity: self.episodic.capacity,
            semantic_patterns: self.semantic.patterns.len(),
            procedural_skills: self.procedural.skills.len(),
            usage_ratio: self.usage_ratio(),
            avg_episode_reward: self.episodic.average_reward(),
            best_recalled_action: self.procedural.get_best_action(),
        }
    }
}

/// Memory system statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub episodic_count: usize,
    pub episodic_capacity: usize,
    pub semantic_patterns: usize,
    pub procedural_skills: usize,
    pub usage_ratio: f64,
    pub avg_episode_reward: f64,
    pub best_recalled_action: Option<usize>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_system_creation() {
        let memory = MemorySystem::new(100, 50);
        // Starts with 0 episodes and 0 patterns
        assert_eq!(memory.episodic.episodes.len(), 0);
        assert_eq!(memory.semantic.patterns.len(), 0);
    }

    #[test]
    fn test_encode() {
        let mut memory = MemorySystem::new(100, 50);

        let input = vec![0.5, 0.3, 0.8, 0.2];
        let actions = vec![0.1, 0.2, 0.3, 0.4];

        memory.encode(&input, &actions, 0.5, 0.3, 1);

        assert_eq!(memory.episodic.episodes.len(), 1);
    }

    #[test]
    fn test_usage_ratio() {
        let mut memory = MemorySystem::new(10, 5);

        for i in 0..10 {
            let input = vec![0.5; 4];
            let actions = vec![0.1; 4];
            memory.encode(&input, &actions, 0.5, 0.3, i as u64);
        }

        assert!((memory.usage_ratio() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_reset() {
        let mut memory = MemorySystem::new(100, 50);

        memory.encode(&vec![0.5; 4], &vec![0.1; 4], 0.5, 0.3, 1);
        memory.reset();

        // After reset, episodic should be empty
        assert_eq!(memory.episodic.episodes.len(), 0);
    }
}
