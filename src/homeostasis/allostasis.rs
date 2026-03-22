//! Allostasis Module
//!
//! Implements allostatic regulation - the process of maintaining stability
//! through active adjustment of internal states

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Allostasis {
    setpoints: Vec<f64>,
    current_values: Vec<f64>,
    deviations: Vec<f64>,
    allostatic_load: f64,
    adaptation_rate: f64,
}

impl Allostasis {
    pub fn new(num_systems: usize) -> Self {
        Self {
            setpoints: vec![0.5; num_systems],
            current_values: vec![0.5; num_systems],
            deviations: vec![0.0; num_systems],
            allostatic_load: 0.0,
            adaptation_rate: 0.01,
        }
    }

    pub fn update(&mut self, system_idx: usize, value: f64) {
        if system_idx < self.current_values.len() {
            let old_value = self.current_values[system_idx];
            self.current_values[system_idx] = value;
            self.deviations[system_idx] = value - self.setpoints[system_idx];

            let deviation_cost = self.deviations[system_idx].abs();
            self.allostatic_load += deviation_cost * self.adaptation_rate;
        }
    }

    pub fn adjust_setpoint(&mut self, system_idx: usize, new_setpoint: f64) {
        if system_idx < self.setpoints.len() {
            self.setpoints[system_idx] = new_setpoint.clamp(0.0, 1.0);
        }
    }

    pub fn get_deviation(&self, system_idx: usize) -> f64 {
        self.deviations.get(system_idx).copied().unwrap_or(0.0)
    }

    pub fn get_allostatic_load(&self) -> f64 {
        self.allostatic_load.min(1.0)
    }

    pub fn reset(&mut self) {
        for i in 0..self.current_values.len() {
            self.current_values[i] = self.setpoints[i];
            self.deviations[i] = 0.0;
        }
        self.allostatic_load = 0.0;
    }

    pub fn decay_load(&mut self, factor: f64) {
        self.allostatic_load *= (1.0 - factor);
    }
}

impl Default for Allostasis {
    fn default() -> Self {
        Self::new(3)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allostasis_creation() {
        let allo = Allostasis::new(3);
        assert_eq!(allo.setpoints.len(), 3);
    }

    #[test]
    fn test_update() {
        let mut allo = Allostasis::new(3);
        allo.update(0, 0.7);
        assert!((allo.current_values[0] - 0.7).abs() < 0.001);
    }

    #[test]
    fn test_allostatic_load() {
        let mut allo = Allostasis::new(3);
        allo.update(0, 0.8);
        assert!(allo.get_allostatic_load() >= 0.0);
    }
}
