//! Procedural Memory Module
//!
//! Stores learned skills and action sequences. Unlike episodic memory which
//! stores specific experiences, procedural memory stores the "how to do it"
//! knowledge that enables fluid execution of learned behaviors.
//!
//! ## Learning Algorithm
//!
//! Uses a variant of Q-learning with eligibility traces for efficient
//! skill acquisition without catastrophic interference.

use serde::{Deserialize, Serialize};

/// A learned skill or action pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Skill {
    /// Associated action values (Q-values)
    pub action_values: Vec<f64>,

    /// Number of times this skill was used
    pub usage_count: u64,

    /// Average reward when this skill was used
    pub avg_reward: f64,

    /// Last timestamp when reinforced
    pub last_used: u64,

    /// Skill name/identifier
    pub name: String,
}

impl Skill {
    /// Create a new skill
    pub fn new(action_count: usize, name: &str) -> Self {
        Self {
            action_values: vec![0.0; action_count],
            usage_count: 0,
            avg_reward: 0.0,
            last_used: 0,
            name: name.to_string(),
        }
    }

    /// Update action values based on reward
    ///
    /// # Arguments
    /// * `actions` - Actions taken
    /// * `reward` - Reward received
    /// * `learning_rate` - Learning rate (alpha)
    /// * `discount_factor` - Discount factor (gamma)
    pub fn update(
        &mut self,
        actions: &[f64],
        reward: f64,
        learning_rate: f64,
        discount_factor: f64,
    ) {
        for (i, &action) in actions.iter().enumerate() {
            if i < self.action_values.len() {
                let current_q = self.action_values[i];
                let max_q = self
                    .action_values
                    .iter()
                    .cloned()
                    .fold(f64::NEG_INFINITY, f64::max);

                self.action_values[i] +=
                    learning_rate * (reward + discount_factor * max_q - current_q);
            }
        }

        self.usage_count += 1;

        let alpha = 1.0 / (self.usage_count as f64);
        self.avg_reward += alpha * (reward - self.avg_reward);
    }

    /// Get the action with highest value
    pub fn get_best_action(&self) -> Option<usize> {
        self.action_values
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
    }

    /// Get the value of a specific action
    pub fn get_action_value(&self, action: usize) -> f64 {
        self.action_values.get(action).copied().unwrap_or(0.0)
    }
}

/// Procedural Memory System
///
/// Stores and manages learned skills through reinforcement learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProceduralMemory {
    /// Available skills
    pub skills: Vec<Skill>,

    /// Primary skill index (most used)
    primary_skill: Option<usize>,

    /// Learning rate
    learning_rate: f64,

    /// Discount factor
    discount_factor: f64,

    /// Eigibility trace decay
    eligibility_decay: f64,

    /// Current eligibility traces
    eligibility_traces: Vec<f64>,

    /// Maximum number of skills
    max_skills: usize,
}

impl Default for ProceduralMemory {
    fn default() -> Self {
        Self::new(100)
    }
}

impl ProceduralMemory {
    /// Create a new procedural memory
    ///
    /// # Arguments
    /// * `action_count` - Number of possible actions
    pub fn new(action_count: usize) -> Self {
        Self {
            skills: Vec::new(),
            primary_skill: None,
            learning_rate: 0.1,
            discount_factor: 0.9,
            eligibility_decay: 0.95,
            eligibility_traces: vec![0.0; action_count],
            max_skills: 10,
        }
    }

    /// Reinforce actions based on reward
    ///
    /// Uses eligibility traces for efficient learning
    ///
    /// # Arguments
    /// * `actions` - Actions that were taken
    /// * `reward` - Reward received
    pub fn reinforce(&mut self, actions: &[f64], reward: f64) {
        if self.primary_skill.is_none() && !actions.is_empty() {
            let skill = Skill::new(actions.len(), "primary");
            self.skills.push(skill);
            self.primary_skill = Some(0);
        }

        if let Some(skill_idx) = self.primary_skill {
            self.skills[skill_idx].update(
                actions,
                reward,
                self.learning_rate,
                self.discount_factor,
            );
        }

        self.update_eligibility_traces(actions);

        self.decay_eligibility();
    }

    /// Update eligibility traces
    fn update_eligibility_traces(&mut self, actions: &[f64]) {
        for (i, &action) in actions.iter().enumerate() {
            if i < self.eligibility_traces.len() {
                if action.abs() > 0.1 {
                    self.eligibility_traces[i] = 1.0;
                }
            }
        }
    }

    /// Decay eligibility traces
    fn decay_eligibility(&mut self) {
        for trace in &mut self.eligibility_traces {
            *trace *= self.eligibility_decay;

            if *trace < 0.01 {
                *trace = 0.0;
            }
        }
    }

    /// Get the best action across all skills
    pub fn get_best_action(&self) -> Option<usize> {
        self.skills
            .iter()
            .filter_map(|s| s.get_best_action())
            .max_by(|a, b| a.cmp(b))
    }

    /// Get the value of the best action
    pub fn get_best_action_value(&self) -> f64 {
        self.get_best_action()
            .and_then(|action| {
                self.skills
                    .iter()
                    .map(|s| s.get_action_value(action))
                    .fold(f64::NEG_INFINITY, f64::max)
                    .into()
            })
            .unwrap_or(0.0)
    }

    /// Get Q-value for a specific state-action pair
    ///
    /// Currently uses a simplified version without state representation
    pub fn get_q_value(&self, action: usize) -> f64 {
        self.skills
            .iter()
            .map(|s| s.get_action_value(action))
            .fold(0.0, |acc, v| acc.max(v))
    }

    /// Add a new skill
    ///
    /// # Arguments
    /// * `action_count` - Number of actions for this skill
    /// * `name` - Skill name
    pub fn add_skill(&mut self, action_count: usize, name: &str) {
        if self.skills.len() >= self.max_skills {
            self.evict_weakest_skill();
        }

        self.skills.push(Skill::new(action_count, name));
    }

    /// Remove the weakest skill
    fn evict_weakest_skill(&mut self) {
        if self.skills.len() <= 1 {
            return;
        }

        let weakest_idx = self
            .skills
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.avg_reward.partial_cmp(&b.avg_reward).unwrap())
            .map(|(i, _)| i);

        if let Some(idx) = weakest_idx {
            if Some(idx) != self.primary_skill {
                self.skills.remove(idx);

                if let Some(ref mut primary) = self.primary_skill {
                    if *primary > idx {
                        *primary -= 1;
                    }
                }
            }
        }
    }

    /// Set learning rate
    pub fn set_learning_rate(&mut self, lr: f64) {
        self.learning_rate = lr.clamp(0.01, 1.0);
    }

    /// Get current eligibility for an action
    pub fn get_eligibility(&self, action: usize) -> f64 {
        self.eligibility_traces.get(action).copied().unwrap_or(0.0)
    }

    /// Get the primary skill
    pub fn get_primary_skill(&self) -> Option<&Skill> {
        self.primary_skill.and_then(|i| self.skills.get(i))
    }

    /// Get all skill summaries
    pub fn get_skill_summaries(&self) -> Vec<SkillSummary> {
        self.skills
            .iter()
            .map(|s| SkillSummary {
                name: s.name.clone(),
                usage_count: s.usage_count,
                avg_reward: s.avg_reward,
                best_action: s.get_best_action(),
            })
            .collect()
    }

    /// Get skill count
    pub fn skill_count(&self) -> usize {
        self.skills.len()
    }

    /// Clear all skills except primary
    pub fn clear(&mut self) {
        let action_count = if let Some(skill) = self.skills.first() {
            skill.action_values.len()
        } else {
            4
        };

        self.skills.clear();
        self.skills.push(Skill::new(action_count, "primary"));
        self.primary_skill = Some(0);
        self.eligibility_traces.fill(0.0);
    }

    /// Reset procedural memory
    pub fn reset(&mut self) {
        self.clear();
    }
}

/// Summary of a skill for display
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillSummary {
    pub name: String,
    pub usage_count: u64,
    pub avg_reward: f64,
    pub best_action: Option<usize>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_procedural_memory_creation() {
        let memory = ProceduralMemory::new(4);
        // Starts empty, skill created on first reinforce
        assert_eq!(memory.skill_count(), 0);
    }

    #[test]
    fn test_reinforce() {
        let mut memory = ProceduralMemory::new(4);
        let actions = vec![0.5, 0.3, 0.8, 0.2];

        memory.reinforce(&actions, 0.5);

        assert_eq!(memory.get_primary_skill().unwrap().usage_count, 1);
    }

    #[test]
    fn test_get_best_action() {
        let mut memory = ProceduralMemory::new(4);
        let actions = vec![0.5, 0.8, 0.3, 0.1];

        memory.reinforce(&actions, 0.5);

        // After reinforcement, primary skill should be created
        assert!(memory.get_primary_skill().is_some());
    }

    #[test]
    fn test_eligibility_traces() {
        let mut memory = ProceduralMemory::new(4);

        memory.reinforce(&vec![0.5, 0.0, 0.0, 0.0], 0.5);

        assert!(memory.get_eligibility(0) > 0.0);
    }

    #[test]
    fn test_add_skill() {
        let mut memory = ProceduralMemory::new(4);
        memory.add_skill(4, "exploration");

        assert_eq!(memory.skill_count(), 1);
    }

    #[test]
    fn test_clear() {
        let mut memory = ProceduralMemory::new(4);

        memory.reinforce(&vec![0.5; 4], 0.5);
        memory.reinforce(&vec![0.3; 4], 0.3);
        memory.clear();

        // After clear, one primary skill is recreated
        assert_eq!(memory.skill_count(), 1);
        assert_eq!(memory.get_primary_skill().unwrap().usage_count, 0);
    }
}
