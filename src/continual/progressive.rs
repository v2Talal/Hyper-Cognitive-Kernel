//! Progressive Neural Networks for Continual Learning

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressiveNetwork {
    pub columns: Vec<Column>,
    pub lateral_connections: HashMap<(usize, usize), LateralConnection>,
    pub column_count: usize,
    pub plasticity: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Column {
    pub id: usize,
    pub input_size: usize,
    pub hidden_sizes: Vec<usize>,
    pub output_size: usize,
    pub weights: Vec<Vec<Vec<f64>>>,
    pub biases: Vec<Vec<f64>>,
    pub frozen: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LateralConnection {
    pub from_column: usize,
    pub to_column: usize,
    pub weights: Vec<Vec<f64>>,
    pub strength: f64,
}

impl ProgressiveNetwork {
    pub fn new(input_size: usize, hidden_sizes: Vec<usize>, output_size: usize) -> Self {
        Self {
            columns: vec![Column::new(
                0,
                input_size,
                hidden_sizes.clone(),
                output_size,
            )],
            lateral_connections: HashMap::new(),
            column_count: 1,
            plasticity: 0.01,
        }
    }

    pub fn add_column(&mut self, output_size: usize) {
        let input_size =
            self.columns[0].input_size + self.columns.iter().map(|c| c.output_size).sum::<usize>();

        let new_column = Column::new(
            self.column_count,
            input_size,
            self.columns[0].hidden_sizes.clone(),
            output_size,
        );

        self.add_lateral_connections(self.column_count);
        self.columns.push(new_column);
        self.column_count += 1;
    }

    fn add_lateral_connections(&mut self, new_col_idx: usize) {
        for (i, col) in self.columns.iter().enumerate() {
            if i == new_col_idx {
                continue;
            }

            let connection = LateralConnection::new(i, new_col_idx, col.output_size);
            self.lateral_connections
                .insert((i, new_col_idx), connection);
        }
    }

    pub fn forward(&self, input: &[f64], column_id: usize) -> Vec<f64> {
        if column_id >= self.columns.len() {
            return vec![0.0; self.columns.last().map(|c| c.output_size).unwrap_or(0)];
        }

        let col = &self.columns[column_id];

        let mut activations = input.to_vec();
        for (layer_idx, (weights, biases)) in col.weights.iter().zip(col.biases.iter()).enumerate()
        {
            let new_activations = self.dense_layer(&activations, weights, biases);

            if layer_idx < col.weights.len() - 1 {
                activations = self.relu(&new_activations);
            } else {
                activations = new_activations;
            }
        }

        activations
    }

    pub fn forward_with_lateral(&self, input: &[f64], column_id: usize) -> Vec<f64> {
        if column_id >= self.columns.len() {
            return vec![0.0; self.columns.last().map(|c| c.output_size).unwrap_or(0)];
        }

        let base_output = self.forward(input, column_id);

        let mut lateral_input = base_output.clone();

        for (key, conn) in &self.lateral_connections {
            if key.1 == column_id {
                let source_output = self.forward(input, key.0);
                let lateral_contribution: Vec<f64> = source_output
                    .iter()
                    .zip(conn.weights.iter())
                    .map(|(o, w)| w.iter().map(|&ww| ww * o).sum::<f64>() * conn.strength)
                    .collect();

                for (i, contrib) in lateral_contribution.iter().enumerate() {
                    if i < lateral_input.len() {
                        lateral_input[i] += contrib;
                    }
                }
            }
        }

        self.softmax(&lateral_input)
    }

    fn dense_layer(&self, input: &[f64], weights: &[Vec<f64>], biases: &[f64]) -> Vec<f64> {
        weights
            .iter()
            .zip(biases.iter())
            .map(|(w, b)| {
                let sum = input.iter().zip(w.iter()).map(|(i, w)| i * w).sum::<f64>();
                sum + b
            })
            .collect()
    }

    fn relu(&self, x: &[f64]) -> Vec<f64> {
        x.iter().map(|v| v.max(0.0)).collect()
    }

    fn softmax(&self, x: &[f64]) -> Vec<f64> {
        let max_val = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = x.iter().map(|x| (x - max_val).exp()).collect();
        let sum: f64 = exps.iter().sum();
        exps.iter().map(|x| x / sum).collect()
    }

    pub fn freeze_column(&mut self, column_id: usize) {
        if column_id < self.columns.len() {
            self.columns[column_id].frozen = true;
        }
    }

    pub fn unfreeze_column(&mut self, column_id: usize) {
        if column_id < self.columns.len() {
            self.columns[column_id].frozen = false;
        }
    }

    pub fn update_lateral_strength(&mut self, from: usize, to: usize, strength: f64) {
        if let Some(conn) = self.lateral_connections.get_mut(&(from, to)) {
            conn.strength = strength.clamp(0.0, 1.0);
        }
    }
}

impl Column {
    pub fn new(id: usize, input_size: usize, hidden_sizes: Vec<usize>, output_size: usize) -> Self {
        let mut rng_seed = (id as u64).wrapping_mul(1000);

        let mut weights = Vec::new();
        let mut biases = Vec::new();
        let mut prev_size = input_size;

        for &size in &hidden_sizes {
            let w: Vec<Vec<f64>> = (0..size)
                .map(|_| {
                    rng_seed = rng_seed.wrapping_mul(1103515245).wrapping_add(12345);
                    (0..prev_size)
                        .map(|_| {
                            rng_seed = rng_seed.wrapping_mul(1103515245).wrapping_add(12345);
                            Self::xorshift_f64(rng_seed) * 0.1
                        })
                        .collect()
                })
                .collect();

            let b = vec![0.0; size];
            weights.push(w);
            biases.push(b);
            prev_size = size;
        }

        let final_weights: Vec<Vec<f64>> = (0..output_size)
            .map(|_| {
                rng_seed = rng_seed.wrapping_mul(1103515245).wrapping_add(12345);
                (0..prev_size)
                    .map(|_| {
                        rng_seed = rng_seed.wrapping_mul(1103515245).wrapping_add(12345);
                        Self::xorshift_f64(rng_seed) * 0.1
                    })
                    .collect()
            })
            .collect();

        weights.push(final_weights);
        biases.push(vec![0.0; output_size]);

        Self {
            id,
            input_size,
            hidden_sizes,
            output_size,
            weights,
            biases,
            frozen: false,
        }
    }

    fn xorshift_f64(seed: u64) -> f64 {
        let mut x = seed;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        (x as i64).abs() as f64 / (i64::MAX as f64) * 2.0 - 1.0
    }
}

impl LateralConnection {
    pub fn new(from: usize, to: usize, size: usize) -> Self {
        let mut rng_seed = (from as u64).wrapping_mul((to + 1) as u64);

        let weights: Vec<Vec<f64>> = (0..size)
            .map(|_| {
                rng_seed = rng_seed.wrapping_mul(1103515245).wrapping_add(12345);
                (0..size)
                    .map(|_| {
                        rng_seed = rng_seed.wrapping_mul(1103515245).wrapping_add(12345);
                        LateralConnection::xorshift_f64(rng_seed) * 0.01
                    })
                    .collect()
            })
            .collect();

        Self {
            from_column: from,
            to_column: to,
            weights,
            strength: 0.1,
        }
    }

    fn xorshift_f64(seed: u64) -> f64 {
        let mut x = seed;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        (x as i64).abs() as f64 / (i64::MAX as f64) * 2.0 - 1.0
    }
}
