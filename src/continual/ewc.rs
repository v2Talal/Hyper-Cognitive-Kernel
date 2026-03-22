//! Elastic Weight Consolidation (EWC) for Continual Learning

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EWC {
    pub lambda: f64,
    pub fisher_dict: HashMap<String, Vec<f64>>,
    pub optimal_params: HashMap<String, Vec<f64>>,
    pub param_count: usize,
}

impl EWC {
    pub fn new(lambda: f64) -> Self {
        Self {
            lambda,
            fisher_dict: HashMap::new(),
            optimal_params: HashMap::new(),
            param_count: 0,
        }
    }

    pub fn compute_penalty(&self, current_params: &[f64]) -> f64 {
        let mut penalty = 0.0;

        for (task_key, fisher) in &self.fisher_dict {
            if let Some(opt_params) = self.optimal_params.get(task_key) {
                for (f, &opt_p) in fisher.iter().zip(opt_params.iter()) {
                    let param_idx = current_params.len().saturating_sub(fisher.len());
                    if param_idx < current_params.len() {
                        let diff = current_params[param_idx] - opt_p;
                        penalty += f * diff * diff;
                    }
                }
            }
        }

        self.lambda * penalty
    }

    pub fn update(&mut self, task_key: String, fisher: Vec<f64>, optimal_params: Vec<f64>) {
        self.fisher_dict.insert(task_key.clone(), fisher);
        self.optimal_params.insert(task_key, optimal_params);
    }

    pub fn consolidate(&mut self, fisher: Vec<f64>, params: Vec<f64>) {
        let task_count = self.fisher_dict.len();
        let task_key = format!("task_{}", task_count);

        self.fisher_dict.insert(task_key.clone(), fisher);
        self.optimal_params.insert(task_key, params);
    }

    pub fn compute_online_fisher(&mut self, gradients: &[f64]) -> Vec<f64> {
        let mut fisher = vec![0.0; gradients.len()];

        for (i, grad) in gradients.iter().enumerate() {
            fisher[i] = 0.95 * fisher[i] + 0.05 * grad * grad;
        }

        fisher
    }

    pub fn get_importance_ranking(&self) -> Vec<(usize, f64)> {
        let mut importance: Vec<(usize, f64)> = self
            .fisher_dict
            .values()
            .flat_map(|f| f.iter().enumerate())
            .map(|(i, &v)| (i, v))
            .collect();

        importance.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        importance
    }

    pub fn prune_low_importance(&self, threshold: f64) -> Vec<bool> {
        let mut mask = Vec::new();

        for fisher in self.fisher_dict.values() {
            for &f in fisher {
                mask.push(f > threshold);
            }
        }

        mask
    }
}
