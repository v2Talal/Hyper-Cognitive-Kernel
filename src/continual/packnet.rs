//! PackNet: Pack and Prune for Continual Learning

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackNet {
    pub masks: Vec<Vec<bool>>,
    pub prune_ratio: f64,
    pub mask_history: Vec<Vec<Vec<bool>>>,
    pub task_masks: Vec<Vec<Vec<bool>>>,
    pub available_capacity: Vec<f64>,
}

impl PackNet {
    pub fn new(num_layers: usize, layer_sizes: &[usize]) -> Self {
        let masks: Vec<Vec<bool>> = layer_sizes.iter().map(|&size| vec![true; size]).collect();

        let available_capacity: Vec<f64> = layer_sizes.iter().map(|&size| size as f64).collect();

        Self {
            masks,
            prune_ratio: 0.5,
            mask_history: Vec::new(),
            task_masks: Vec::new(),
            available_capacity,
        }
    }

    pub fn prune(&mut self, layer_idx: usize, importance_scores: &[f64]) {
        if layer_idx >= self.masks.len() {
            return;
        }

        let current_capacity = self.masks[layer_idx].iter().filter(|&&m| m).count();
        let target_keep = (current_capacity as f64 * self.prune_ratio) as usize;

        let mut indexed_scores: Vec<(usize, f64)> = importance_scores
            .iter()
            .enumerate()
            .map(|(i, &s)| (i, s))
            .collect();

        indexed_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        for (i, &(idx, _)) in indexed_scores.iter().enumerate() {
            if i >= target_keep && self.masks[layer_idx][idx] {
                self.masks[layer_idx][idx] = false;
            }
        }

        self.available_capacity[layer_idx] =
            self.masks[layer_idx].iter().filter(|&&m| m).count() as f64;
    }

    pub fn apply_mask(&self, weights: &[f64], layer_idx: usize) -> Vec<f64> {
        if layer_idx >= self.masks.len() {
            return weights.to_vec();
        }

        weights
            .iter()
            .zip(self.masks[layer_idx].iter())
            .map(|(&w, &m)| if m { w } else { 0.0 })
            .collect()
    }

    pub fn save_task_mask(&mut self) {
        self.task_masks.push(self.masks.clone());
        self.mask_history.push(self.masks.clone());
    }

    pub fn get_task_mask(&self, task_id: usize) -> Option<&Vec<Vec<bool>>> {
        self.task_masks.get(task_id)
    }

    pub fn combine_task_masks(&self) -> Vec<Vec<bool>> {
        if self.task_masks.is_empty() {
            return self.masks.clone();
        }

        let mut combined = vec![vec![false; self.masks[0].len()]; self.masks.len()];

        for task_mask in &self.task_masks {
            for (layer_idx, layer) in task_mask.iter().enumerate() {
                for (neuron_idx, &mask) in layer.iter().enumerate() {
                    if layer_idx < combined.len() && neuron_idx < combined[layer_idx].len() {
                        combined[layer_idx][neuron_idx] = combined[layer_idx][neuron_idx] || mask;
                    }
                }
            }
        }

        combined
    }

    pub fn finetune_task(&mut self, task_id: usize, importance_scores: &[f64]) {
        let task_mask = self.get_task_mask(task_id);

        if let Some(mask) = task_mask {
            let mut finetune_mask: Vec<Vec<bool>> = mask.clone();

            for layer_idx in 0..finetune_mask.len() {
                let available: Vec<usize> = finetune_mask[layer_idx]
                    .iter()
                    .enumerate()
                    .filter(|(_, &m)| !m)
                    .map(|(i, _)| i)
                    .collect();

                let prune_count = (available.len() as f64 * self.prune_ratio) as usize;

                use rand::seq::SliceRandom;
                let mut indices = available;
                indices.shuffle(&mut rand::thread_rng());

                for idx in indices.into_iter().take(prune_count) {
                    finetune_mask[layer_idx][idx] = true;
                }
            }

            self.masks = finetune_mask;
        }
    }

    pub fn reset(&mut self) {
        for mask in &mut self.masks {
            for m in mask.iter_mut() {
                *m = true;
            }
        }
        self.task_masks.clear();
        self.mask_history.clear();

        let original_sizes: Vec<usize> = self
            .available_capacity
            .iter()
            .map(|&c| c as usize)
            .collect();
        self.available_capacity = original_sizes.iter().map(|&c| c as f64).collect();
    }

    pub fn get_utilization(&self) -> Vec<f64> {
        let mut utilization = Vec::new();

        for (mask, &original) in self.masks.iter().zip(self.available_capacity.iter()) {
            let current = mask.iter().filter(|&&m| m).count() as f64;
            utilization.push(current / original);
        }

        utilization
    }
}
