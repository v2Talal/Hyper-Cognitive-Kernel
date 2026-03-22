//! Semantic Memory Module
//!
//! Stores abstract patterns and conceptual knowledge extracted from episodic memories.
//! Patterns are formed through consolidation of similar experiences.
//!
//! ## Pattern Representation
//!
//! Each pattern is a compressed representation of recurring experiences,
//! capturing the essential features while filtering out noise.

use super::episodic::EpisodicMemory;
use serde::{Deserialize, Serialize};

/// A semantic pattern extracted from episodic memories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pattern {
    /// Feature vector representing this pattern
    pub features: Vec<f64>,

    /// Confidence level (0.0 to 1.0) based on supporting episodes
    pub confidence: f64,

    /// Number of episodes that contributed to this pattern
    pub support_count: usize,

    /// When this pattern was last reinforced
    pub last_reinforced: u64,

    /// Pattern importance (for retrieval priority)
    pub importance: f64,
}

impl Pattern {
    /// Create a new pattern
    pub fn new(features: Vec<f64>, support_count: usize, timestamp: u64) -> Self {
        let confidence = (support_count as f64 / 10.0).min(1.0);

        Self {
            features,
            confidence,
            support_count,
            last_reinforced: timestamp,
            importance: 1.0,
        }
    }

    /// Update pattern with new evidence
    pub fn reinforce(&mut self, new_features: &[f64], timestamp: u64) {
        let alpha = 1.0 / (self.support_count as f64 + 1.0);

        for (i, &new_val) in new_features.iter().enumerate() {
            if i < self.features.len() {
                self.features[i] += alpha * (new_val - self.features[i]);
            }
        }

        self.support_count += 1;
        self.confidence = (self.confidence + 0.1).min(1.0);
        self.last_reinforced = timestamp;
    }

    /// Calculate similarity with another pattern or vector
    pub fn similarity(&self, other: &[f64]) -> f64 {
        if self.features.is_empty() || other.is_empty() {
            return 0.0;
        }

        let len = self.features.len().min(other.len());

        let dot_product: f64 = self
            .features
            .iter()
            .take(len)
            .zip(other.iter().take(len))
            .map(|(a, b)| a * b)
            .sum();

        let norm_self: f64 = self
            .features
            .iter()
            .take(len)
            .map(|x| x * x)
            .sum::<f64>()
            .sqrt();

        let norm_other: f64 = other.iter().take(len).map(|x| x * x).sum::<f64>().sqrt();

        if norm_self == 0.0 || norm_other == 0.0 {
            return 0.0;
        }

        dot_product / (norm_self * norm_other)
    }
}

/// Semantic Memory System
///
/// Extracts and maintains abstract patterns from episodic memories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticMemory {
    /// Stored patterns
    pub patterns: Vec<Pattern>,

    /// Maximum number of patterns to maintain
    max_patterns: usize,

    /// Minimum confidence to retain a pattern
    min_confidence: f64,

    /// Pattern extraction threshold
    extraction_threshold: usize,

    /// Current pattern count
    pattern_count: u64,
}

impl Default for SemanticMemory {
    fn default() -> Self {
        Self::new()
    }
}

impl SemanticMemory {
    /// Create a new semantic memory
    pub fn new() -> Self {
        Self {
            patterns: Vec::new(),
            max_patterns: 100,
            min_confidence: 0.1,
            extraction_threshold: 10,
            pattern_count: 0,
        }
    }

    /// Extract patterns from episodic memory
    ///
    /// Called periodically to consolidate episodic memories into semantic patterns.
    /// Uses k-means-like clustering to identify recurring patterns.
    ///
    /// # Arguments
    /// * `episodic` - Episodic memory to extract from
    pub fn extract_patterns(&mut self, episodic: &EpisodicMemory) {
        if episodic.episodes.len() < self.extraction_threshold {
            return;
        }

        let recent_episodes: Vec<_> = episodic.get_recent(self.extraction_threshold);

        if recent_episodes.is_empty() {
            return;
        }

        let pattern_size = recent_episodes[0].input.len();

        let avg_pattern = Self::compute_centroid(
            recent_episodes
                .iter()
                .map(|e| e.input.as_slice())
                .collect::<Vec<_>>()
                .as_slice(),
        );

        if let Some(similar) = self.find_similar_pattern(&avg_pattern) {
            self.patterns[similar]
                .reinforce(&avg_pattern, episodic.episodes.last().unwrap().timestamp);
        } else {
            if self.patterns.len() >= self.max_patterns {
                self.evict_low_confidence();
            }

            self.patterns.push(Pattern::new(
                avg_pattern,
                recent_episodes.len(),
                episodic.episodes.last().unwrap().timestamp,
            ));
        }

        self.pattern_count += 1;
    }

    /// Compute centroid of multiple vectors
    fn compute_centroid(vectors: &[&[f64]]) -> Vec<f64> {
        if vectors.is_empty() {
            return vec![];
        }

        let len = vectors[0].len();
        let mut centroid = vec![0.0; len];

        for v in vectors {
            for (i, &val) in v.iter().enumerate().take(len) {
                centroid[i] += val;
            }
        }

        for val in centroid.iter_mut() {
            *val /= vectors.len() as f64;
        }

        centroid
    }

    /// Find a pattern similar to the given features
    ///
    /// # Arguments
    /// * `features` - Feature vector to match
    ///
    /// # Returns
    /// Index of similar pattern, or None if no match
    fn find_similar_pattern(&self, features: &[f64]) -> Option<usize> {
        let threshold = 0.9;

        for (i, pattern) in self.patterns.iter().enumerate() {
            if pattern.similarity(features) >= threshold {
                return Some(i);
            }
        }

        None
    }

    /// Remove patterns with low confidence
    fn evict_low_confidence(&mut self) {
        self.patterns
            .retain(|p| p.confidence >= self.min_confidence);

        if self.patterns.is_empty() {
            return;
        }

        let min_idx = self
            .patterns
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.confidence.partial_cmp(&b.confidence).unwrap())
            .map(|(i, _)| i);

        if let Some(idx) = min_idx {
            self.patterns.remove(idx);
        }
    }

    /// Retrieve the most relevant pattern for given input
    ///
    /// # Arguments
    /// * `input` - Input to match against patterns
    /// * `min_confidence` - Minimum confidence threshold
    ///
    /// # Returns
    /// Most similar pattern, or None
    pub fn retrieve(&self, input: &[f64], min_confidence: f64) -> Option<&Pattern> {
        self.patterns
            .iter()
            .filter(|p| p.confidence >= min_confidence)
            .max_by(|a, b| {
                let sim_a = a.similarity(input);
                let sim_b = b.similarity(input);
                sim_a.partial_cmp(&sim_b).unwrap()
            })
    }

    /// Get all patterns above confidence threshold
    pub fn get_high_confidence(&self, threshold: f64) -> Vec<&Pattern> {
        self.patterns
            .iter()
            .filter(|p| p.confidence >= threshold)
            .collect()
    }

    /// Get pattern count
    pub fn count(&self) -> usize {
        self.patterns.len()
    }

    /// Get average confidence
    pub fn average_confidence(&self) -> f64 {
        if self.patterns.is_empty() {
            return 0.0;
        }

        let sum: f64 = self.patterns.iter().map(|p| p.confidence).sum();
        sum / self.patterns.len() as f64
    }

    /// Clear all patterns
    pub fn clear(&mut self) {
        self.patterns.clear();
    }

    /// Reset semantic memory
    pub fn reset(&mut self) {
        self.patterns.clear();
        self.pattern_count = 0;
    }

    /// Get statistics
    pub fn get_stats(&self) -> SemanticStats {
        SemanticStats {
            pattern_count: self.patterns.len(),
            max_patterns: self.max_patterns,
            avg_confidence: self.average_confidence(),
            extraction_count: self.pattern_count,
        }
    }
}

/// Semantic memory statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticStats {
    pub pattern_count: usize,
    pub max_patterns: usize,
    pub avg_confidence: f64,
    pub extraction_count: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_semantic_memory_creation() {
        let memory = SemanticMemory::new();
        assert_eq!(memory.count(), 0);
    }

    #[test]
    fn test_pattern_similarity() {
        let pattern = Pattern::new(vec![0.5, 0.5, 0.5], 5, 1);
        let other = vec![0.6, 0.6, 0.6];

        let similarity = pattern.similarity(&other);

        assert!(similarity > 0.9);
    }

    #[test]
    fn test_pattern_reinforce() {
        let mut pattern = Pattern::new(vec![0.5, 0.5], 5, 1);
        let new_features = vec![0.7, 0.7];

        let old_confidence = pattern.confidence;
        pattern.reinforce(&new_features, 2);

        assert!(pattern.confidence > old_confidence);
        assert_eq!(pattern.support_count, 6);
    }

    #[test]
    fn test_extract_patterns() {
        let mut semantic = SemanticMemory::new();
        let mut episodic = EpisodicMemory::new(20);

        for i in 0..15 {
            episodic.store(&[0.5, 0.5], &[0.1, 0.1], 0.5, 0.3, i as u64);
        }

        semantic.extract_patterns(&episodic);

        assert!(semantic.count() > 0);
    }

    #[test]
    fn test_retrieve() {
        let mut memory = SemanticMemory::new();
        memory.patterns.push(Pattern::new(vec![0.5, 0.5], 5, 1));

        let retrieved = memory.retrieve(&[0.6, 0.6], 0.1);

        assert!(retrieved.is_some());
    }

    #[test]
    fn test_clear() {
        let mut memory = SemanticMemory::new();
        memory.patterns.push(Pattern::new(vec![0.5, 0.5], 5, 1));

        memory.clear();

        assert_eq!(memory.count(), 0);
    }
}
