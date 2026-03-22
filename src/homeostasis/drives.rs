//! Drive System Module
//!
//! Implements multiple motivational drives that regulate agent behavior

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriveSystem {
    pub survival: f64,
    pub curiosity: f64,
    pub efficiency: f64,
    pub survival_weight: f64,
    pub curiosity_weight: f64,
    pub efficiency_weight: f64,
    update_count: u64,
}

impl DriveSystem {
    pub fn new(survival_w: f64, curiosity_w: f64, efficiency_w: f64) -> Self {
        Self {
            survival: 1.0,
            curiosity: 0.5,
            efficiency: 1.0,
            survival_weight: survival_w,
            curiosity_weight: curiosity_w,
            efficiency_weight: efficiency_w,
            update_count: 0,
        }
    }

    pub fn update(&mut self, reward: f64, surprise: f64, age: u64) {
        self.survival = (self.survival - 0.001 + reward * 0.1).clamp(0.0, 1.0);
        self.curiosity = (self.curiosity + surprise * 0.1 - 0.01).clamp(0.0, 1.0);
        self.efficiency = (self.efficiency + 0.0001).clamp(0.0, 1.0);
        self.update_count += 1;
    }

    pub fn get_states(&self) -> Vec<(String, f64)> {
        vec![
            ("survival".to_string(), self.survival),
            ("curiosity".to_string(), self.curiosity),
            ("efficiency".to_string(), self.efficiency),
        ]
    }

    pub fn get_primary_drive(&self) -> f64 {
        self.survival * self.survival_weight
    }

    pub fn get_modulation(&self, action_idx: usize) -> f64 {
        match action_idx % 3 {
            0 => self.survival * self.survival_weight,
            1 => self.curiosity * self.curiosity_weight,
            2 => self.efficiency * self.efficiency_weight,
            _ => 1.0,
        }
    }

    pub fn get_attention_boost(&self, input_idx: usize) -> f64 {
        if input_idx % 2 == 0 {
            1.0 + self.survival * 0.5
        } else {
            1.0 + self.curiosity * 0.5
        }
    }

    pub fn is_alive(&self) -> bool {
        self.survival > 0.0
    }

    pub fn reset(&mut self) {
        self.survival = 1.0;
        self.curiosity = 0.5;
        self.efficiency = 1.0;
        self.update_count = 0;
    }

    pub fn get_total_drive(&self) -> f64 {
        self.survival * self.survival_weight
            + self.curiosity * self.curiosity_weight
            + self.efficiency * self.efficiency_weight
    }

    pub fn set_weights(&mut self, survival: f64, curiosity: f64, efficiency: f64) {
        self.survival_weight = survival.clamp(0.0, 2.0);
        self.curiosity_weight = curiosity.clamp(0.0, 2.0);
        self.efficiency_weight = efficiency.clamp(0.0, 2.0);
    }

    pub fn apply_penalty(&mut self, amount: f64) {
        self.survival = (self.survival - amount).max(0.0);
    }

    pub fn apply_reward(&mut self, amount: f64) {
        self.survival = (self.survival + amount).min(1.0);
        self.curiosity = (self.curiosity + amount * 0.5).min(1.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_drive_creation() {
        let drives = DriveSystem::new(1.0, 0.5, 0.3);
        assert_eq!(drives.survival, 1.0);
        assert_eq!(drives.curiosity, 0.5);
    }

    #[test]
    fn test_update() {
        let mut drives = DriveSystem::new(1.0, 0.5, 0.3);
        drives.update(0.5, 0.3, 1);
        assert!(drives.survival > 0.0);
    }

    #[test]
    fn test_is_alive() {
        let drives = DriveSystem::new(1.0, 0.5, 0.3);
        assert!(drives.is_alive());
    }
}
