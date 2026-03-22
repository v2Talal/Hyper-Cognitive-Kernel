//! Adaptive Learning Rate for Real-time Learning

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveLearningRate {
    pub base_lr: f64,
    pub min_lr: f64,
    pub max_lr: f64,
    pub decay_factor: f64,
    pub growth_factor: f64,
    pub window_size: usize,
    pub loss_history: Vec<f64>,
    pub lr_history: Vec<f64>,
    pub patience: usize,
    pub no_improve_count: usize,
    pub best_loss: f64,
    pub last_lr: f64,
}

impl AdaptiveLearningRate {
    pub fn new() -> Self {
        Self {
            base_lr: 0.001,
            min_lr: 0.00001,
            max_lr: 0.1,
            decay_factor: 0.95,
            growth_factor: 1.05,
            window_size: 20,
            loss_history: Vec::with_capacity(100),
            lr_history: Vec::with_capacity(100),
            patience: 5,
            no_improve_count: 0,
            best_loss: f64::MAX,
            last_lr: 0.001,
        }
    }

    pub fn with_lr(mut self, lr: f64) -> Self {
        self.base_lr = lr;
        self.last_lr = lr;
        self
    }

    pub fn compute(&mut self, current_loss: f64) -> f64 {
        self.loss_history.push(current_loss);

        if self.loss_history.len() > self.window_size * 2 {
            self.loss_history.remove(0);
        }

        if self.loss_history.len() < self.window_size {
            self.last_lr = self.base_lr;
            self.lr_history.push(self.last_lr);
            return self.last_lr;
        }

        let min_history = self.window_size * 2;
        if self.loss_history.len() < min_history {
            self.last_lr = self.base_lr;
            self.lr_history.push(self.last_lr);
            return self.last_lr;
        }

        let recent_losses = &self.loss_history[self.loss_history.len() - self.window_size..];
        let older_losses = &self.loss_history[self.loss_history.len() - self.window_size * 2
            ..self.loss_history.len() - self.window_size];

        let recent_avg = recent_losses.iter().sum::<f64>() / recent_losses.len() as f64;
        let older_avg = older_losses.iter().sum::<f64>() / older_losses.len() as f64;

        let improvement = older_avg - recent_avg;
        let relative_improvement = improvement / older_avg.max(0.0001);

        if recent_avg < self.best_loss {
            self.best_loss = recent_avg;
            self.no_improve_count = 0;
        } else {
            self.no_improve_count += 1;
        }

        let new_lr = if self.no_improve_count >= self.patience {
            self.last_lr * self.decay_factor
        } else if relative_improvement > 0.1 {
            (self.last_lr * self.growth_factor).min(self.max_lr)
        } else if relative_improvement > 0.05 {
            self.last_lr
        } else if relative_improvement < -0.1 {
            (self.last_lr * self.decay_factor).max(self.min_lr)
        } else {
            self.last_lr
        };

        self.last_lr = new_lr.clamp(self.min_lr, self.max_lr);
        self.lr_history.push(self.last_lr);

        self.last_lr
    }

    pub fn reset(&mut self) {
        self.loss_history.clear();
        self.lr_history.clear();
        self.no_improve_count = 0;
        self.best_loss = f64::MAX;
        self.last_lr = self.base_lr;
    }

    pub fn get_current_lr(&self) -> f64 {
        self.last_lr
    }

    pub fn get_lr_trend(&self) -> f64 {
        if self.lr_history.len() < 10 {
            return 0.0;
        }

        let recent_avg = self.lr_history.iter().rev().take(5).sum::<f64>() / 5.0;

        let older_avg = self.lr_history.iter().rev().skip(5).take(5).sum::<f64>() / 5.0;

        (recent_avg - older_avg) / older_avg.max(0.0001)
    }

    pub fn suggest_lr(&self) -> f64 {
        if self.lr_history.is_empty() {
            return self.base_lr;
        }

        let best_idx = self
            .loss_history
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        if best_idx < self.lr_history.len() {
            self.lr_history[best_idx]
        } else {
            self.base_lr
        }
    }
}

impl Default for AdaptiveLearningRate {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningRateSchedule {
    schedule_type: ScheduleType,
    current_step: usize,
    initial_lr: f64,
    final_lr: f64,
    warmup_steps: usize,
    decay_steps: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum ScheduleType {
    Constant,
    Step,
    Exponential,
    Cosine,
    WarmupCosine,
}

impl LearningRateSchedule {
    pub fn constant(lr: f64) -> Self {
        Self {
            schedule_type: ScheduleType::Constant,
            current_step: 0,
            initial_lr: lr,
            final_lr: lr,
            warmup_steps: 0,
            decay_steps: usize::MAX,
        }
    }

    pub fn cosine_decay(lr: f64, decay_steps: usize) -> Self {
        Self {
            schedule_type: ScheduleType::Cosine,
            current_step: 0,
            initial_lr: lr,
            final_lr: lr * 0.01,
            warmup_steps: 0,
            decay_steps,
        }
    }

    pub fn warmup_cosine(lr: f64, warmup_steps: usize, total_steps: usize) -> Self {
        Self {
            schedule_type: ScheduleType::WarmupCosine,
            current_step: 0,
            initial_lr: lr * 0.1,
            final_lr: lr * 0.01,
            warmup_steps,
            decay_steps: total_steps,
        }
    }

    pub fn get_lr(&mut self) -> f64 {
        self.current_step += 1;

        match self.schedule_type {
            ScheduleType::Constant => self.initial_lr,

            ScheduleType::Step => {
                let decay_factor = ((self.current_step / 1000) as f64).floor();
                self.initial_lr * 0.1_f64.powf(decay_factor)
            }

            ScheduleType::Exponential => {
                let progress = (self.current_step as f64) / (self.decay_steps as f64).max(1.0);
                let factor = progress.min(1.0);
                self.initial_lr * (self.final_lr / self.initial_lr).powf(factor)
            }

            ScheduleType::Cosine => {
                let progress = (self.current_step as f64) / (self.decay_steps as f64).max(1.0);
                let factor = (progress * std::f64::consts::PI).cos() * 0.5 + 0.5;
                self.final_lr + (self.initial_lr - self.final_lr) * factor
            }

            ScheduleType::WarmupCosine => {
                let warmup_progress =
                    (self.current_step as f64) / (self.warmup_steps as f64).max(1.0);
                let decay_progress = ((self.current_step.saturating_sub(self.warmup_steps)) as f64)
                    / ((self.decay_steps - self.warmup_steps) as f64).max(1.0);

                if self.current_step <= self.warmup_steps {
                    self.initial_lr
                        + (self.initial_lr * 10.0 - self.initial_lr) * warmup_progress.min(1.0)
                } else {
                    let cosine_factor =
                        (decay_progress.min(1.0) * std::f64::consts::PI).cos() * 0.5 + 0.5;
                    self.final_lr + (self.initial_lr * 10.0 - self.final_lr) * cosine_factor
                }
            }
        }
    }
}
