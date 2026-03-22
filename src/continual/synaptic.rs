//! Synaptic Intelligence (SI) for Continual Learning

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynapticIntelligence {
    pub omega: Vec<f64>,
    pub params_prev: Vec<f64>,
    pub params_current: Vec<f64>,
    pub c: f64,
    pub zeta: f64,
    pub accumulated_omega: Vec<f64>,
    pub task_count: usize,
}

impl SynapticIntelligence {
    pub fn new(param_count: usize) -> Self {
        Self {
            omega: vec![0.0; param_count],
            params_prev: vec![0.0; param_count],
            params_current: vec![0.0; param_count],
            c: 0.5,
            zeta: 1e-9,
            accumulated_omega: vec![0.0; param_count],
            task_count: 0,
        }
    }

    pub fn update(&mut self, params: &[f64], gradients: &[f64]) {
        self.params_prev = self.params_current.clone();
        self.params_current = params.to_vec();

        for i in 0..self.omega.len().min(gradients.len()) {
            let param_change = (self.params_current[i] - self.params_prev[i]).abs();

            if param_change > self.zeta {
                self.omega[i] += gradients[i].abs() / (param_change + self.zeta);
            }

            self.accumulated_omega[i] += self.omega[i];
        }
    }

    pub fn compute_penalty(&self, params: &[f64]) -> f64 {
        let mut penalty = 0.0;

        for (i, &p) in params.iter().enumerate() {
            if i < self.accumulated_omega.len() && self.accumulated_omega[i] > 0.0 {
                let omega_normalized = self.omega[i] / (self.accumulated_omega[i] + self.zeta);
                penalty += omega_normalized * (p - self.params_prev[i]).powi(2);
            }
        }

        self.c * penalty
    }

    pub fn consolidate(&mut self) {
        self.params_prev = self.params_current.clone();

        for omega in &mut self.omega {
            *omega = 0.0;
        }

        self.task_count += 1;
    }

    pub fn get_importance_weights(&self) -> Vec<f64> {
        let mut weights = Vec::with_capacity(self.omega.len());

        let total: f64 = self.omega.iter().sum();

        if total > 0.0 {
            for &omega in &self.omega {
                weights.push(omega / total);
            }
        } else {
            weights.resize(self.omega.len(), 0.0);
        }

        weights
    }

    pub fn prune_low_importance(&self, threshold: f64) -> Vec<bool> {
        self.omega.iter().map(|&w| w > threshold).collect()
    }

    pub fn reset(&mut self) {
        self.omega.fill(0.0);
        self.params_prev.fill(0.0);
        self.params_current.fill(0.0);
        self.accumulated_omega.fill(0.0);
        self.task_count = 0;
    }

    pub fn get_stats(&self) -> SIStats {
        let avg_omega = if !self.omega.is_empty() {
            self.omega.iter().sum::<f64>() / self.omega.len() as f64
        } else {
            0.0
        };

        let max_omega = self.omega.iter().cloned().fold(0.0_f64, f64::max);
        let min_omega = self.omega.iter().cloned().fold(f64::MAX, f64::min);

        SIStats {
            task_count: self.task_count,
            avg_importance: avg_omega,
            max_importance: max_omega,
            min_importance: min_omega,
            total_params: self.omega.len(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SIStats {
    pub task_count: usize,
    pub avg_importance: f64,
    pub max_importance: f64,
    pub min_importance: f64,
    pub total_params: usize,
}
