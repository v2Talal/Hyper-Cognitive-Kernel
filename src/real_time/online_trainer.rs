//! Online Trainer for Real-time Learning

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use super::super::continual::ContinualLearner;
use super::super::neural::{NeuralNetwork, NeuralStats};
use super::{LearningDecision, RealTimeLearningConfig, RealTimeStats, Sample};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnlineTrainerConfig {
    pub gradient_accumulation_steps: usize,
    pub eval_every_n_steps: usize,
    pub save_checkpoint_every: usize,
    pub early_stopping_patience: usize,
    pub early_stopping_threshold: f64,
    pub use_continual_learning: bool,
    pub use_elastic_weights: bool,
    pub ewc_lambda: f64,
}

impl Default for OnlineTrainerConfig {
    fn default() -> Self {
        Self {
            gradient_accumulation_steps: 1,
            eval_every_n_steps: 100,
            save_checkpoint_every: 1000,
            early_stopping_patience: 10,
            early_stopping_threshold: 0.001,
            use_continual_learning: true,
            use_elastic_weights: true,
            ewc_lambda: 5000.0,
        }
    }
}

pub struct OnlineTrainer {
    model: Arc<RwLock<NeuralNetwork>>,
    continual_learner: Option<ContinualLearner>,
    config: OnlineTrainerConfig,
    rt_config: RealTimeLearningConfig,
    stats: TrainingStats,
    step: usize,
    best_loss: f64,
    no_improve_steps: usize,
    accumulated_gradients: Vec<f64>,
    accumulated_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingStats {
    pub current_step: usize,
    pub total_samples: u64,
    pub total_loss: f64,
    pub avg_loss: f64,
    pub best_loss: f64,
    pub eval_count: usize,
    pub checkpoint_count: usize,
    pub early_stopped: bool,
    pub training_time_ms: u64,
    pub steps_per_second: f64,
}

impl Default for TrainingStats {
    fn default() -> Self {
        Self {
            current_step: 0,
            total_samples: 0,
            total_loss: 0.0,
            avg_loss: 0.0,
            best_loss: f64::MAX,
            eval_count: 0,
            checkpoint_count: 0,
            early_stopped: false,
            training_time_ms: 0,
            steps_per_second: 0.0,
        }
    }
}

impl OnlineTrainer {
    pub fn new(model: NeuralNetwork) -> Self {
        Self::with_config(model, OnlineTrainerConfig::default())
    }

    pub fn with_config(model: NeuralNetwork, config: OnlineTrainerConfig) -> Self {
        let continual_learner = if config.use_continual_learning {
            Some(ContinualLearner::with_ewc(config.ewc_lambda))
        } else {
            None
        };

        Self {
            model: Arc::new(RwLock::new(model)),
            continual_learner,
            config: config.clone(),
            rt_config: RealTimeLearningConfig::default(),
            stats: TrainingStats::default(),
            step: 0,
            best_loss: f64::MAX,
            no_improve_steps: 0,
            accumulated_gradients: Vec::new(),
            accumulated_count: 0,
        }
    }

    pub fn train_sample(&mut self, input: &[f64], target: &[f64]) -> f64 {
        self.step += 1;
        self.stats.current_step = self.step;

        let start_time = std::time::Instant::now();

        let loss = {
            let mut model = self.model.write();
            let loss = model.online_train(input, target);

            if let Some(ref mut cl) = self.continual_learner {
                if self.config.use_elastic_weights {
                    let ewc_loss = cl.compute_ewc_penalty(&model);
                    loss + ewc_loss
                } else {
                    loss
                }
            } else {
                loss
            }
        };

        self.stats.total_samples += 1;
        self.stats.total_loss += loss;
        self.stats.avg_loss = self.stats.total_loss / self.stats.total_samples as f64;
        self.stats.training_time_ms += start_time.elapsed().as_millis() as u64;

        if loss < self.best_loss {
            self.best_loss = loss;
            self.no_improve_steps = 0;

            if let Some(ref mut cl) = self.continual_learner {
                let model = self.model.read();
                let params = cl.get_model_params(&model);
                let fisher = cl.compute_fisher_diagonal(&model);
                cl.save_task(params, fisher);
            }
        } else {
            self.no_improve_steps += 1;
        }

        if self.step % self.config.eval_every_n_steps == 0 {
            self.stats.eval_count += 1;
        }

        if self.no_improve_steps
            >= self.config.early_stopping_patience * self.config.eval_every_n_steps
        {
            self.stats.early_stopped = true;
        }

        if self.stats.training_time_ms > 0 {
            self.stats.steps_per_second =
                (self.step as f64) / (self.stats.training_time_ms as f64 / 1000.0);
        }

        loss
    }

    pub fn train_batch(&mut self, batch: &[(&[f64], &[f64])]) -> f64 {
        let total_loss: f64 = batch
            .iter()
            .map(|(input, target)| self.train_sample(input, target))
            .sum();

        total_loss / batch.len() as f64
    }

    pub fn train_stream<I, T>(&mut self, stream: I) -> f64
    where
        I: Iterator<Item = (Vec<f64>, Vec<f64>)>,
        T: Into<Vec<f64>>,
    {
        let mut total_loss = 0.0;
        let mut count = 0;

        for (input, target) in stream {
            total_loss += self.train_sample(&input, &target);
            count += 1;
        }

        if count > 0 {
            total_loss / count as f64
        } else {
            0.0
        }
    }

    pub fn predict(&self, input: &[f64]) -> Vec<f64> {
        self.model.write().predict(input)
    }

    pub fn predict_class(&self, input: &[f64]) -> usize {
        self.model.write().predict_class(input)
    }

    pub fn get_model(&self) -> NeuralNetwork {
        self.model.read().clone()
    }

    pub fn get_stats(&self) -> TrainingStats {
        self.stats.clone()
    }

    pub fn get_neural_stats(&self) -> NeuralStats {
        self.model.read().get_stats()
    }

    pub fn should_stop(&self) -> bool {
        self.stats.early_stopped
    }

    pub fn reset(&mut self) {
        self.step = 0;
        self.stats = TrainingStats::default();
        self.best_loss = f64::MAX;
        self.no_improve_steps = 0;
        self.accumulated_gradients.clear();
        self.accumulated_count = 0;

        if let Some(ref mut cl) = self.continual_learner {
            cl.reset();
        }
    }

    pub fn load_checkpoint(&mut self, path: &str) -> Result<(), String> {
        let contents = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read checkpoint: {}", e))?;

        let checkpoint: Checkpoint = serde_json::from_str(&contents)
            .map_err(|e| format!("Failed to parse checkpoint: {}", e))?;

        self.step = checkpoint.step;
        self.stats = checkpoint.stats;

        {
            let mut model = self.model.write();
            *model = checkpoint.model;
        }

        if let Some(ref mut cl) = self.continual_learner {
            cl.load_fisher_and_params(&checkpoint.fisher_info, &checkpoint.opt_params);
        }

        Ok(())
    }

    pub fn save_checkpoint(&mut self, path: &str) -> Result<(), String> {
        let model = self.model.read().clone();

        let (fisher_info, opt_params) = if let Some(ref cl) = self.continual_learner {
            cl.get_checkpoint_data()
        } else {
            (Vec::new(), Vec::new())
        };

        let checkpoint = Checkpoint {
            step: self.step,
            stats: self.stats.clone(),
            model,
            fisher_info,
            opt_params,
        };

        let json = serde_json::to_string_pretty(&checkpoint)
            .map_err(|e| format!("Failed to serialize checkpoint: {}", e))?;

        std::fs::write(path, json).map_err(|e| format!("Failed to write checkpoint: {}", e))?;

        self.stats.checkpoint_count += 1;
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Checkpoint {
    step: usize,
    stats: TrainingStats,
    model: NeuralNetwork,
    fisher_info: Vec<f64>,
    opt_params: Vec<f64>,
}

pub struct StreamingTrainer {
    trainer: OnlineTrainer,
    buffer: Vec<(Vec<f64>, Vec<f64>)>,
    buffer_size: usize,
}

impl StreamingTrainer {
    pub fn new(model: NeuralNetwork, buffer_size: usize) -> Self {
        Self {
            trainer: OnlineTrainer::new(model),
            buffer: Vec::with_capacity(buffer_size),
            buffer_size,
        }
    }

    pub fn add_sample(&mut self, input: Vec<f64>, target: Vec<f64>) {
        if self.buffer.len() >= self.buffer_size {
            self.buffer.remove(0);
        }
        self.buffer.push((input, target));
    }

    pub fn train_on_buffer(&mut self) -> f64 {
        let batch: Vec<(&[f64], &[f64])> = self
            .buffer
            .iter()
            .map(|(i, t)| (i.as_slice(), t.as_slice()))
            .collect();

        self.trainer.train_batch(&batch)
    }

    pub fn train_streaming(&mut self) -> f64 {
        let batch: Vec<(&[f64], &[f64])> = self
            .buffer
            .iter()
            .map(|(i, t)| (i.as_slice(), t.as_slice()))
            .collect();

        self.trainer.train_batch(&batch)
    }

    pub fn get_trainer(&self) -> &OnlineTrainer {
        &self.trainer
    }

    pub fn get_trainer_mut(&mut self) -> &mut OnlineTrainer {
        &mut self.trainer
    }
}
