//! Activation Functions for Neural Networks

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ActivationFunction {
    Identity,
    ReLU,
    LeakyReLU,
    Sigmoid,
    Tanh,
    Softmax,
    Swish,
    GELU,
}

impl ActivationFunction {
    pub fn apply(&self, x: f64) -> f64 {
        match self {
            ActivationFunction::Identity => x,
            ActivationFunction::ReLU => x.max(0.0),
            ActivationFunction::LeakyReLU => {
                if x > 0.0 {
                    x
                } else {
                    x * 0.01
                }
            }
            ActivationFunction::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            ActivationFunction::Tanh => x.tanh(),
            ActivationFunction::Softmax => 1.0 / (1.0 + (-x).exp()),
            ActivationFunction::Swish => x / (1.0 + (-x).exp()),
            ActivationFunction::GELU => 0.5 * x * (1.0 + (x * 1.702).tanh()),
        }
    }

    pub fn derivative(&self, x: f64) -> f64 {
        match self {
            ActivationFunction::Identity => 1.0,
            ActivationFunction::ReLU => {
                if x > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            ActivationFunction::LeakyReLU => {
                if x > 0.0 {
                    1.0
                } else {
                    0.01
                }
            }
            ActivationFunction::Sigmoid => {
                let s = 1.0 / (1.0 + (-x).exp());
                s * (1.0 - s)
            }
            ActivationFunction::Tanh => 1.0 - x.tanh().powi(2),
            ActivationFunction::Softmax => x * (1.0 - x),
            ActivationFunction::Swish => {
                let s = 1.0 / (1.0 + (-x).exp());
                s + x * s * (1.0 - s)
            }
            ActivationFunction::GELU => {
                let t = (1.0 + (x * 1.702).tanh());
                0.5 * t + 0.5 * x * (1.702 * (1.0 - (x * 1.702).tanh().powi(2))) / (t * t)
            }
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            ActivationFunction::Identity => "Identity",
            ActivationFunction::ReLU => "ReLU",
            ActivationFunction::LeakyReLU => "LeakyReLU",
            ActivationFunction::Sigmoid => "Sigmoid",
            ActivationFunction::Tanh => "Tanh",
            ActivationFunction::Softmax => "Softmax",
            ActivationFunction::Swish => "Swish",
            ActivationFunction::GELU => "GELU",
        }
    }
}

impl Default for ActivationFunction {
    fn default() -> Self {
        ActivationFunction::ReLU
    }
}

pub struct ActivationFunctions;

impl ActivationFunctions {
    pub fn softmax(inputs: &[f64]) -> Vec<f64> {
        let max_val = inputs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = inputs.iter().map(|x| (x - max_val).exp()).collect();
        let sum: f64 = exps.iter().sum();
        exps.iter().map(|x| x / sum).collect()
    }

    pub fn log_softmax(inputs: &[f64]) -> Vec<f64> {
        let max_val = inputs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let shifted: Vec<f64> = inputs.iter().map(|x| x - max_val).collect();
        let sum: f64 = shifted.iter().map(|x| x.exp()).sum::<f64>().ln();
        shifted.iter().map(|x| x - sum).collect()
    }

    pub fn mish(x: f64) -> f64 {
        x * (1.0 + (-x).exp() + (-x).exp() * (-x).exp()).recip().tanh()
    }

    pub fn hard_sigmoid(x: f64) -> f64 {
        (x + 1.0).clamp(0.0, 2.0) / 2.0
    }

    pub fn hard_tanh(x: f64) -> f64 {
        x.clamp(-1.0, 1.0)
    }

    pub fn elu(x: f64) -> f64 {
        if x >= 0.0 {
            x
        } else {
            x.exp() - 1.0
        }
    }

    pub fn selu(x: f64) -> f64 {
        let alpha = 1.6732632423543772848170429916717;
        let scale = 1.0507009873554804934193349852946;
        if x >= 0.0 {
            scale * x
        } else {
            scale * alpha * (x.exp() - 1.0)
        }
    }
}
