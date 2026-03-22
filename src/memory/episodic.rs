//! Episodic Memory Module
//!
//! Stores specific experiences as episodes, each containing:
//! - Sensory input state
//! - Actions taken
//! - Reward received
//! - Surprise level
//! - Temporal timestamp
//!
//! Implements FIFO eviction when capacity is reached.

use serde::{Deserialize, Serialize};

/// A single episodic memory entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Episode {
    /// Sensory input that triggered this episode
    pub input: Vec<f64>,

    /// Actions taken during this episode
    pub actions: Vec<f64>,

    /// Reward received
    pub reward: f64,

    /// Prediction error (surprise) at time of encoding
    pub surprise: f64,

    /// Temporal tag (cycle number)
    pub timestamp: u64,

    /// Importance weight (can be modified by consolidation)
    pub importance: f64,
}

impl Episode {
    /// Create a new episode
    pub fn new(
        input: Vec<f64>,
        actions: Vec<f64>,
        reward: f64,
        surprise: f64,
        timestamp: u64,
    ) -> Self {
        let importance = 1.0 + surprise;
        Self {
            input,
            actions,
            reward,
            surprise,
            timestamp,
            importance,
        }
    }

    /// Calculate similarity with another episode
    ///
    /// Uses cosine similarity on input vectors
    pub fn similarity(&self, other: &[f64]) -> f64 {
        if self.input.is_empty() || other.is_empty() {
            return 0.0;
        }

        let len = self.input.len().min(other.len());

        let dot_product: f64 = self
            .input
            .iter()
            .take(len)
            .zip(other.iter().take(len))
            .map(|(a, b)| a * b)
            .sum();

        let norm_a: f64 = self
            .input
            .iter()
            .take(len)
            .map(|x| x * x)
            .sum::<f64>()
            .sqrt();

        let norm_b: f64 = other.iter().take(len).map(|x| x * x).sum::<f64>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }

        dot_product / (norm_a * norm_b)
    }

    /// Update importance score
    pub fn update_importance(&mut self, delta: f64) {
        self.importance = (self.importance + delta).max(0.1).min(10.0);
    }
}

/// Episodic Memory System
///
/// Stores and retrieves specific experiences with temporal ordering.
/// Implements importance-based forgetting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodicMemory {
    /// Stored episodes
    pub episodes: Vec<Episode>,

    /// Maximum capacity
    pub capacity: usize,

    /// Number of stores performed
    store_count: u64,

    /// Decay rate for old episodes
    decay_rate: f64,
}

impl EpisodicMemory {
    /// Create a new episodic memory
    ///
    /// # Arguments
    /// * `capacity` - Maximum number of episodes to store
    pub fn new(capacity: usize) -> Self {
        Self {
            episodes: Vec::with_capacity(capacity),
            capacity,
            store_count: 0,
            decay_rate: 0.001,
        }
    }

    /// Store a new episode
    ///
    /// If at capacity, removes the oldest or least important episode
    ///
    /// # Arguments
    /// * `input` - Sensory input
    /// * `actions` - Actions taken
    /// * `reward` - Reward received
    /// * `surprise` - Surprise level
    /// * `timestamp` - Current time
    pub fn store(
        &mut self,
        input: &[f64],
        actions: &[f64],
        reward: f64,
        surprise: f64,
        timestamp: u64,
    ) {
        let episode = Episode::new(
            input.to_vec(),
            actions.to_vec(),
            reward,
            surprise,
            timestamp,
        );

        if self.episodes.len() >= self.capacity {
            self.evict_least_important();
        }

        self.episodes.push(episode);
        self.store_count += 1;
    }

    /// Evict the least important episode
    fn evict_least_important(&mut self) {
        if self.episodes.is_empty() {
            return;
        }

        let decay = self.decay_rate;

        for episode in &mut self.episodes {
            episode.update_importance(-decay);
        }

        let min_idx = self
            .episodes
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.importance.partial_cmp(&b.importance).unwrap())
            .map(|(i, _)| i);

        if let Some(idx) = min_idx {
            self.episodes.remove(idx);
        }
    }

    /// Retrieve episodes similar to the current input
    ///
    /// # Arguments
    /// * `current_input` - Input to match against
    /// * `threshold` - Minimum similarity threshold
    ///
    /// # Returns
    /// Vector of episodes with similarity above threshold
    pub fn retrieve_similar(&self, current_input: &[f64], threshold: f64) -> Vec<&Episode> {
        self.episodes
            .iter()
            .filter(|e| e.similarity(current_input) >= threshold)
            .collect()
    }

    /// Get the episode with highest reward
    pub fn get_best_episode(&self) -> Option<&Episode> {
        self.episodes
            .iter()
            .max_by(|a, b| a.reward.partial_cmp(&b.reward).unwrap())
    }

    /// Get the most recent episodes
    ///
    /// # Arguments
    /// * `n` - Number of episodes to retrieve
    pub fn get_recent(&self, n: usize) -> Vec<&Episode> {
        self.episodes.iter().rev().take(n).collect()
    }

    /// Get episodes within a time range
    ///
    /// # Arguments
    /// * `start` - Start timestamp (inclusive)
    /// * `end` - End timestamp (inclusive)
    pub fn get_by_time_range(&self, start: u64, end: u64) -> Vec<&Episode> {
        self.episodes
            .iter()
            .filter(|e| e.timestamp >= start && e.timestamp <= end)
            .collect()
    }

    /// Boost importance of recent episodes (for high surprise events)
    ///
    /// # Arguments
    /// * `timestamp` - Timestamp to boost from
    pub fn importance_boost(&mut self, timestamp: u64) {
        let recent_window = 10;

        for episode in &mut self.episodes {
            if episode.timestamp >= timestamp.saturating_sub(recent_window) {
                episode.update_importance(0.1);
            }
        }
    }

    /// Calculate memory usage as ratio of capacity
    pub fn usage_ratio(&self) -> f64 {
        self.episodes.len() as f64 / self.capacity as f64
    }

    /// Calculate average reward across all episodes
    pub fn average_reward(&self) -> f64 {
        if self.episodes.is_empty() {
            return 0.0;
        }

        let sum: f64 = self.episodes.iter().map(|e| e.reward).sum();
        sum / self.episodes.len() as f64
    }

    /// Calculate average surprise across all episodes
    pub fn average_surprise(&self) -> f64 {
        if self.episodes.is_empty() {
            return 0.0;
        }

        let sum: f64 = self.episodes.iter().map(|e| e.surprise).sum();
        sum / self.episodes.len() as f64
    }

    /// Get episode count
    pub fn count(&self) -> usize {
        self.episodes.len()
    }

    /// Clear all episodes
    pub fn clear(&mut self) {
        self.episodes.clear();
    }

    /// Reset memory state
    pub fn reset(&mut self) {
        self.episodes.clear();
        self.store_count = 0;
    }

    /// Apply decay to all episodes
    pub fn apply_decay(&mut self) {
        for episode in &mut self.episodes {
            episode.update_importance(-self.decay_rate);
        }

        self.episodes.retain(|e| e.importance > 0.05);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_episodic_memory_creation() {
        let memory = EpisodicMemory::new(10);
        assert_eq!(memory.capacity, 10);
        assert_eq!(memory.count(), 0);
    }

    #[test]
    fn test_store() {
        let mut memory = EpisodicMemory::new(10);

        memory.store(&[0.5, 0.3], &[0.1, 0.2], 0.5, 0.3, 1);

        assert_eq!(memory.count(), 1);
    }

    #[test]
    fn test_capacity_eviction() {
        let mut memory = EpisodicMemory::new(3);

        for i in 0..5 {
            memory.store(&[0.5, 0.3], &[0.1, 0.2], 0.5, 0.3, i as u64);
        }

        assert_eq!(memory.count(), 3);
    }

    #[test]
    fn test_retrieve_similar() {
        let mut memory = EpisodicMemory::new(10);

        memory.store(&[0.5, 0.5], &[0.1, 0.1], 0.5, 0.3, 1);
        memory.store(&[0.8, 0.8], &[0.2, 0.2], 0.6, 0.4, 2);
        memory.store(&[0.2, 0.2], &[0.3, 0.3], 0.4, 0.2, 3);

        let similar = memory.retrieve_similar(&[0.5, 0.5], 0.8);

        assert!(!similar.is_empty());
    }

    #[test]
    fn test_get_best_episode() {
        let mut memory = EpisodicMemory::new(10);

        memory.store(&[0.5, 0.5], &[0.1, 0.1], 0.3, 0.3, 1);
        memory.store(&[0.8, 0.8], &[0.2, 0.2], 0.8, 0.4, 2);
        memory.store(&[0.2, 0.2], &[0.3, 0.3], 0.5, 0.2, 3);

        let best = memory.get_best_episode().unwrap();
        assert!((best.reward - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_usage_ratio() {
        let mut memory = EpisodicMemory::new(10);

        for i in 0..5 {
            memory.store(&[0.5, 0.5], &[0.1, 0.1], 0.5, 0.3, i as u64);
        }

        assert!((memory.usage_ratio() - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_reset() {
        let mut memory = EpisodicMemory::new(10);

        memory.store(&[0.5, 0.5], &[0.1, 0.1], 0.5, 0.3, 1);
        memory.reset();

        assert_eq!(memory.count(), 0);
    }
}
