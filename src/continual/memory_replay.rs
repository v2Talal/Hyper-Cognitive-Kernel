//! Memory Replay for Continual Learning

use super::super::continual::ReplaySample;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryReplay {
    pub buffer_size: usize,
    pub replay_ratio: f64,
    pub samples: Vec<ReplaySample>,
    pub her_ratio: f64,
}

impl MemoryReplay {
    pub fn new(buffer_size: usize) -> Self {
        Self {
            buffer_size,
            replay_ratio: 0.3,
            samples: Vec::with_capacity(buffer_size),
            her_ratio: 0.0,
        }
    }

    pub fn with_her(buffer_size: usize, her_ratio: f64) -> Self {
        Self {
            buffer_size,
            replay_ratio: 0.3,
            samples: Vec::with_capacity(buffer_size),
            her_ratio,
        }
    }

    pub fn add(&mut self, input: Vec<f64>, target: Vec<f64>, task_id: usize) {
        if self.samples.len() >= self.buffer_size {
            self.remove_oldest();
        }

        let sample = ReplaySample {
            input,
            target,
            task_id,
            age: 0,
            importance: 1.0,
        };

        self.samples.push(sample);
    }

    fn remove_oldest(&mut self) {
        if self.samples.is_empty() {
            return;
        }

        if let Some(min_idx) = self
            .samples
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.age.cmp(&b.age))
            .map(|(i, _)| i)
        {
            self.samples.remove(min_idx);
        }
    }

    pub fn sample(&self, batch_size: usize) -> Vec<&ReplaySample> {
        if self.samples.is_empty() {
            return Vec::new();
        }

        let batch_size = batch_size.min(self.samples.len());
        let mut indices: Vec<usize> = (0..self.samples.len()).collect();

        use rand::seq::SliceRandom;
        indices.shuffle(&mut rand::thread_rng());

        indices
            .into_iter()
            .take(batch_size)
            .map(|i| &self.samples[i])
            .collect()
    }

    pub fn sample_balanced(&self, batch_size: usize) -> Vec<&ReplaySample> {
        if self.samples.is_empty() {
            return Vec::new();
        }

        let task_ids: Vec<usize> = self
            .samples
            .iter()
            .map(|s| s.task_id)
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        if task_ids.is_empty() {
            return self.sample(batch_size);
        }

        let per_task = batch_size / task_ids.len();
        let mut result = Vec::new();

        for task_id in task_ids {
            let task_samples: Vec<_> = self
                .samples
                .iter()
                .filter(|s| s.task_id == task_id)
                .collect();

            let mut indices: Vec<usize> = (0..task_samples.len()).collect();
            use rand::seq::SliceRandom;
            indices.shuffle(&mut rand::thread_rng());

            for idx in indices.into_iter().take(per_task) {
                result.push(task_samples[idx]);
            }
        }

        result
    }

    pub fn age_samples(&mut self) {
        for sample in &mut self.samples {
            sample.age += 1;
        }
    }

    pub fn update_priorities(&mut self, indices: &[usize], losses: &[f64]) {
        for (&idx, &loss) in indices.iter().zip(losses.iter()) {
            if idx < self.samples.len() {
                self.samples[idx].importance = 1.0 / (loss + 0.01);
            }
        }
    }

    pub fn her_transform(
        &self,
        sample: &ReplaySample,
        achieved: &[f64],
        desired: &[f64],
    ) -> (Vec<f64>, Vec<f64>) {
        let new_target: Vec<f64> = sample
            .target
            .iter()
            .zip(desired.iter())
            .map(|(t, d)| if *t > 0.0 { *d } else { *t })
            .collect();

        (desired.to_vec(), new_target)
    }
}
