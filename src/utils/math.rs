//! Mathematical Utilities Module

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MathUtils;

impl MathUtils {
    pub fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    pub fn derivative_sigmoid(x: f64) -> f64 {
        let s = Self::sigmoid(x);
        s * (1.0 - s)
    }

    pub fn relu(x: f64) -> f64 {
        x.max(0.0)
    }

    pub fn derivative_relu(x: f64) -> f64 {
        if x > 0.0 {
            1.0
        } else {
            0.0
        }
    }

    pub fn tanh(x: f64) -> f64 {
        x.tanh()
    }

    pub fn softmax(inputs: &[f64]) -> Vec<f64> {
        let max_val = inputs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = inputs.iter().map(|x| (x - max_val).exp()).collect();
        let sum: f64 = exps.iter().sum();
        exps.iter().map(|x| x / sum).collect()
    }

    pub fn clip(value: f64, min: f64, max: f64) -> f64 {
        value.clamp(min, max)
    }

    pub fn noise(seed: u64) -> f64 {
        let x = (seed as f64).sin() * 10000.0;
        x - x.floor()
    }

    pub fn xorshift_f64(seed: u64) -> f64 {
        let mut x = seed;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        let normalized = (x as i64).abs() as f64 / (i64::MAX as f64);
        normalized * 2.0 - 1.0
    }

    pub fn mean(values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        values.iter().sum::<f64>() / values.len() as f64
    }

    pub fn std(values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        let mean = Self::mean(values);
        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
        variance.sqrt()
    }

    pub fn normalize(values: &mut [f64]) {
        let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = max - min;
        if range > 0.0 {
            for v in values.iter_mut() {
                *v = (*v - min) / range;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid() {
        assert!((MathUtils::sigmoid(0.0) - 0.5).abs() < 0.001);
        assert!(MathUtils::sigmoid(10.0) > 0.99);
    }

    #[test]
    fn test_softmax() {
        let inputs = vec![1.0, 2.0, 3.0];
        let result = MathUtils::softmax(&inputs);
        let sum: f64 = result.iter().sum();
        assert!((sum - 1.0).abs() < 0.001);
    }
}
