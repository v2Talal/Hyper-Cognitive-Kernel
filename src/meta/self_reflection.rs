//! Self-Reflection Module
//!
//! Provides introspection capabilities for analyzing agent performance

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfReflection {
    state_history: Vec<CognitiveState>,
    history_capacity: usize,
    performance_metrics: PerformanceMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CognitiveState {
    age: u64,
    free_energy: f64,
    drive_states: Vec<(String, f64)>,
    reward: f64,
    timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct PerformanceMetrics {
    total_cycles: u64,
    peak_performance: f64,
    avg_free_energy: f64,
    best_drive_state: Option<(String, f64)>,
}

impl Default for SelfReflection {
    fn default() -> Self {
        Self::new()
    }
}

impl SelfReflection {
    pub fn new() -> Self {
        Self {
            state_history: Vec::with_capacity(100),
            history_capacity: 100,
            performance_metrics: PerformanceMetrics::default(),
        }
    }

    pub fn record_cycle(
        &mut self,
        age: u64,
        free_energy: f64,
        drive_states: Vec<(String, f64)>,
        reward: f64,
    ) {
        let state = CognitiveState {
            age,
            free_energy,
            drive_states: drive_states.clone(),
            reward,
            timestamp: age,
        };

        self.state_history.push(state);

        if self.state_history.len() > self.history_capacity {
            self.state_history.remove(0);
        }

        self.update_metrics(age, free_energy, &drive_states);
    }

    fn update_metrics(&mut self, age: u64, free_energy: f64, drive_states: &[(String, f64)]) {
        self.performance_metrics.total_cycles = age;

        let performance = 1.0 - free_energy.min(1.0);
        if performance > self.performance_metrics.peak_performance {
            self.performance_metrics.peak_performance = performance;
        }

        let total_fe: f64 = self.state_history.iter().map(|s| s.free_energy).sum();
        let count = self.state_history.len() as f64;
        self.performance_metrics.avg_free_energy = if count > 0.0 { total_fe / count } else { 0.0 };

        for (name, value) in drive_states {
            if name == "survival" {
                if let Some((_, existing)) = self.performance_metrics.best_drive_state {
                    if *value > existing {
                        self.performance_metrics.best_drive_state = Some((name.clone(), *value));
                    }
                } else {
                    self.performance_metrics.best_drive_state = Some((name.clone(), *value));
                }
            }
        }
    }

    pub fn generate_report(&self) -> SelfReflectionReport {
        SelfReflectionReport {
            age: self.state_history.last().map(|s| s.age).unwrap_or(0),
            free_energy: self
                .state_history
                .last()
                .map(|s| s.free_energy)
                .unwrap_or(0.0),
            drive_states: self
                .state_history
                .last()
                .map(|s| s.drive_states.clone())
                .unwrap_or_default(),
            memory_usage: 0.0,
            learning_rate: 0.01,
            prediction_accuracy: self.calculate_accuracy(),
            total_reward: self.calculate_total_reward(),
            successful_predictions: 0,
            exploration_ratio: 0.0,
            meta_adaptations: 0,
            world_model_updates: 0,
        }
    }

    fn calculate_accuracy(&self) -> f64 {
        if self.state_history.is_empty() {
            return 0.0;
        }
        let good = self
            .state_history
            .iter()
            .filter(|s| s.free_energy < 0.3)
            .count();
        good as f64 / self.state_history.len() as f64
    }

    fn calculate_total_reward(&self) -> f64 {
        self.state_history.iter().map(|s| s.reward).sum()
    }

    pub fn analyze_trends(&self, window: usize) -> TrendAnalysis {
        let window = window.min(self.state_history.len());
        if window == 0 {
            return TrendAnalysis::default();
        }

        let recent: Vec<_> = self.state_history.iter().rev().take(window).collect();
        let avg_fe: f64 = recent.iter().map(|s| s.free_energy).sum::<f64>() / window as f64;
        let avg_reward: f64 = recent.iter().map(|s| s.reward).sum::<f64>() / window as f64;

        TrendAnalysis {
            window_size: window,
            avg_free_energy: avg_fe,
            avg_reward,
            free_energy_std: 0.0,
            free_energy_trend: 0.0,
            reward_trend: 0.0,
            stability_score: 1.0 - avg_fe.min(1.0),
        }
    }

    pub fn cognitive_efficiency(&self) -> f64 {
        if self.state_history.is_empty() {
            return 0.0;
        }
        let avg_fe = self
            .state_history
            .iter()
            .map(|s| s.free_energy)
            .sum::<f64>()
            / self.state_history.len() as f64;
        1.0 / (1.0 + avg_fe)
    }

    pub fn reset(&mut self) {
        self.state_history.clear();
        self.performance_metrics = PerformanceMetrics::default();
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfReflectionReport {
    pub age: u64,
    pub free_energy: f64,
    pub drive_states: Vec<(String, f64)>,
    pub memory_usage: f64,
    pub learning_rate: f64,
    pub prediction_accuracy: f64,
    pub total_reward: f64,
    pub successful_predictions: u64,
    pub exploration_ratio: f64,
    pub meta_adaptations: u64,
    pub world_model_updates: u64,
}

impl Default for SelfReflectionReport {
    fn default() -> Self {
        Self {
            age: 0,
            free_energy: 0.0,
            drive_states: vec![],
            memory_usage: 0.0,
            learning_rate: 0.01,
            prediction_accuracy: 0.0,
            total_reward: 0.0,
            successful_predictions: 0,
            exploration_ratio: 0.0,
            meta_adaptations: 0,
            world_model_updates: 0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TrendAnalysis {
    pub window_size: usize,
    pub avg_free_energy: f64,
    pub avg_reward: f64,
    pub free_energy_std: f64,
    pub free_energy_trend: f64,
    pub reward_trend: f64,
    pub stability_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehaviorPattern {
    pub pattern_type: String,
    pub start_age: u64,
    pub end_age: u64,
    pub frequency: usize,
    pub average_reward: f64,
}
