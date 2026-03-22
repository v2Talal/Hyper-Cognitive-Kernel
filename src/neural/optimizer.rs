//! Online Optimizers for Real-time Learning

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OnlineOptimizer {
    SGD(SGDConfig),
    Adam(AdamConfig),
    RMSprop(RMSpropConfig),
}

impl OnlineOptimizer {
    pub fn update(&mut self, weights: &mut [f64], gradients: &[f64], learning_rate: f64) {
        match self {
            OnlineOptimizer::SGD(cfg) => cfg.update(weights, gradients, learning_rate),
            OnlineOptimizer::Adam(cfg) => cfg.update(weights, gradients, learning_rate),
            OnlineOptimizer::RMSprop(cfg) => cfg.update(weights, gradients, learning_rate),
        }
    }

    pub fn reset(&mut self) {
        match self {
            OnlineOptimizer::SGD(cfg) => cfg.reset(),
            OnlineOptimizer::Adam(cfg) => cfg.reset(),
            OnlineOptimizer::RMSprop(cfg) => cfg.reset(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SGDConfig {
    pub momentum: f64,
    pub velocity: Vec<f64>,
    pub nesterov: bool,
}

impl SGDConfig {
    pub fn new() -> Self {
        Self {
            momentum: 0.9,
            velocity: Vec::new(),
            nesterov: true,
        }
    }

    pub fn update(&mut self, weights: &mut [f64], gradients: &[f64], learning_rate: f64) {
        if self.velocity.len() != weights.len() {
            self.velocity.resize(weights.len(), 0.0);
        }

        for i in 0..weights.len().min(gradients.len()) {
            let grad = gradients[i].clamp(-10.0, 10.0);

            if self.nesterov {
                self.velocity[i] = self.momentum * self.velocity[i] + grad;
                weights[i] -= learning_rate * (grad + self.momentum * self.velocity[i]);
            } else {
                self.velocity[i] = self.momentum * self.velocity[i] + grad;
                weights[i] -= learning_rate * self.velocity[i];
            }

            weights[i] = weights[i].clamp(-100.0, 100.0);
        }
    }

    pub fn reset(&mut self) {
        self.velocity.fill(0.0);
    }
}

impl Default for SGDConfig {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdamConfig {
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
    pub m: Vec<f64>,
    pub v: Vec<f64>,
    pub t: usize,
    pub corrected_bias1: f64,
    pub corrected_bias2: f64,
}

impl AdamConfig {
    pub fn new() -> Self {
        Self {
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            m: Vec::new(),
            v: Vec::new(),
            t: 0,
            corrected_bias1: 0.0,
            corrected_bias2: 0.0,
        }
    }

    pub fn update(&mut self, weights: &mut [f64], gradients: &[f64], learning_rate: f64) {
        if self.m.len() != weights.len() {
            self.m.resize(weights.len(), 0.0);
            self.v.resize(weights.len(), 0.0);
        }

        self.t += 1;

        let beta1_power = self.beta1.powi(self.t as i32);
        let beta2_power = self.beta2.powi(self.t as i32);

        self.corrected_bias1 = 1.0 - beta1_power;
        self.corrected_bias2 = 1.0 - beta2_power;

        for i in 0..weights.len().min(gradients.len()) {
            let grad = gradients[i].clamp(-10.0, 10.0);

            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * grad;
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * grad * grad;

            let m_hat = self.m[i] / self.corrected_bias1;
            let v_hat = self.v[i] / self.corrected_bias2;

            let update = learning_rate * m_hat / (v_hat.sqrt() + self.epsilon);
            weights[i] -= update;
            weights[i] = weights[i].clamp(-100.0, 100.0);
        }
    }

    pub fn reset(&mut self) {
        self.m.fill(0.0);
        self.v.fill(0.0);
        self.t = 0;
    }
}

impl Default for AdamConfig {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RMSpropConfig {
    pub decay: f64,
    pub epsilon: f64,
    pub square_avg: Vec<f64>,
    pub momentum: f64,
    pub velocity: Vec<f64>,
}

impl RMSpropConfig {
    pub fn new() -> Self {
        Self {
            decay: 0.9,
            epsilon: 1e-8,
            square_avg: Vec::new(),
            momentum: 0.0,
            velocity: Vec::new(),
        }
    }

    pub fn update(&mut self, weights: &mut [f64], gradients: &[f64], learning_rate: f64) {
        if self.square_avg.len() != weights.len() {
            self.square_avg.resize(weights.len(), 0.0);
            self.velocity.resize(weights.len(), 0.0);
        }

        for i in 0..weights.len().min(gradients.len()) {
            let grad = gradients[i].clamp(-10.0, 10.0);

            self.square_avg[i] = self.decay * self.square_avg[i] + (1.0 - self.decay) * grad * grad;

            if self.momentum > 0.0 {
                self.velocity[i] = self.momentum * self.velocity[i]
                    + learning_rate * grad / (self.square_avg[i].sqrt() + self.epsilon);
                weights[i] -= self.velocity[i];
            } else {
                weights[i] -= learning_rate * grad / (self.square_avg[i].sqrt() + self.epsilon);
            }

            weights[i] = weights[i].clamp(-100.0, 100.0);
        }
    }

    pub fn reset(&mut self) {
        self.square_avg.fill(0.0);
        self.velocity.fill(0.0);
    }
}

impl Default for RMSpropConfig {
    fn default() -> Self {
        Self::new()
    }
}
