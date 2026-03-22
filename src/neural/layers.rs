//! Neural Network Layers for Online Learning

use super::ActivationFunction;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Layer {
    Dense(DenseLayer),
    Convolutional(ConvolutionalLayer),
    Recurrent(RecurrentLayer),
    LSTM(LSTM),
    Attention(Attention),
    Reservoir(super::ReservoirLayer),
}

impl Layer {
    pub fn forward(&mut self, input: &[f64]) -> Vec<f64> {
        match self {
            Layer::Dense(l) => l.forward(input),
            Layer::Convolutional(l) => l.forward(input),
            Layer::Recurrent(l) => l.forward(input),
            Layer::LSTM(l) => l.forward(input),
            Layer::Attention(l) => l.forward(input),
            Layer::Reservoir(l) => l.forward(input),
        }
    }

    pub fn output_size(&self) -> usize {
        match self {
            Layer::Dense(l) => l.output_size(),
            Layer::Convolutional(l) => l.output_size(),
            Layer::Recurrent(l) => l.output_size(),
            Layer::LSTM(l) => l.output_size(),
            Layer::Attention(l) => l.output_size(),
            Layer::Reservoir(l) => l.output_size(),
        }
    }

    pub fn weight_count(&self) -> usize {
        match self {
            Layer::Dense(l) => l.weight_count(),
            Layer::Convolutional(l) => l.weight_count(),
            Layer::Recurrent(l) => l.weight_count(),
            Layer::LSTM(l) => l.weight_count(),
            Layer::Attention(l) => l.weight_count(),
            Layer::Reservoir(l) => l.weight_count(),
        }
    }

    pub fn compute_gradients(&mut self, input: &[f64], target: &[f64]) -> Vec<f64> {
        match self {
            Layer::Dense(l) => l.compute_gradients(input, target),
            Layer::Convolutional(l) => l.compute_gradients(input, target),
            Layer::Recurrent(l) => l.compute_gradients(input, target),
            Layer::LSTM(l) => l.compute_gradients(input, target),
            Layer::Attention(l) => l.compute_gradients(input, target),
            Layer::Reservoir(l) => l.compute_gradients(input, target),
        }
    }

    pub fn apply_gradients(&mut self, gradients: &[f64]) {
        match self {
            Layer::Dense(l) => l.apply_gradients(gradients),
            Layer::Convolutional(l) => l.apply_gradients(gradients),
            Layer::Recurrent(l) => l.apply_gradients(gradients),
            Layer::LSTM(l) => l.apply_gradients(gradients),
            Layer::Attention(l) => l.apply_gradients(gradients),
            Layer::Reservoir(l) => l.apply_gradients(gradients),
        }
    }

    pub fn reset(&mut self) {
        match self {
            Layer::Dense(l) => l.reset(),
            Layer::Convolutional(l) => l.reset(),
            Layer::Recurrent(l) => l.reset(),
            Layer::LSTM(l) => l.reset(),
            Layer::Attention(l) => l.reset(),
            Layer::Reservoir(l) => l.reset(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DenseLayer {
    pub input_size: usize,
    pub output_size: usize,
    pub weights: Vec<Vec<f64>>,
    pub biases: Vec<f64>,
    pub activation: ActivationFunction,
    pub last_input: Vec<f64>,
    pub last_output: Vec<f64>,
    pub velocity_w: Vec<Vec<f64>>,
    pub velocity_b: Vec<f64>,
    learning_rate: f64,
    momentum: f64,
}

impl DenseLayer {
    pub fn new(input_size: usize, output_size: usize, activation: ActivationFunction) -> Self {
        let mut rng_seed = (input_size as u64).wrapping_mul(output_size as u64);

        let weights: Vec<Vec<f64>> = (0..output_size)
            .map(|_| {
                rng_seed = rng_seed.wrapping_mul(1103515245).wrapping_add(12345);
                (0..input_size)
                    .map(|_| {
                        rng_seed = rng_seed.wrapping_mul(1103515245).wrapping_add(12345);
                        Self::xorshift_f64(rng_seed) * 0.01
                    })
                    .collect()
            })
            .collect();

        let biases = vec![0.0; output_size];
        let velocity_w = vec![vec![0.0; input_size]; output_size];
        let velocity_b = vec![0.0; output_size];

        Self {
            input_size,
            output_size,
            weights,
            biases,
            activation,
            last_input: Vec::new(),
            last_output: Vec::new(),
            velocity_w,
            velocity_b,
            learning_rate: 0.001,
            momentum: 0.9,
        }
    }

    fn xorshift_f64(seed: u64) -> f64 {
        let mut x = seed;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        (x as i64).abs() as f64 / (i64::MAX as f64) * 2.0 - 1.0
    }

    pub fn forward(&mut self, input: &[f64]) -> Vec<f64> {
        self.last_input = input.to_vec();

        let output: Vec<f64> = (0..self.output_size)
            .map(|i| {
                let sum = self.biases[i]
                    + self.weights[i]
                        .iter()
                        .zip(input.iter())
                        .map(|(w, x)| w * x)
                        .sum::<f64>();
                self.activation.apply(sum)
            })
            .collect();

        self.last_output = output.clone();
        output
    }

    pub fn output_size(&self) -> usize {
        self.output_size
    }

    pub fn weight_count(&self) -> usize {
        self.input_size * self.output_size + self.output_size
    }

    pub fn compute_gradients(&mut self, _input: &[f64], target: &[f64]) -> Vec<f64> {
        let output_error: Vec<f64> = self
            .last_output
            .iter()
            .zip(target.iter())
            .map(|(o, t)| {
                let delta = o - t;
                delta * self.activation.derivative(*o)
            })
            .collect();

        let mut gradients = Vec::with_capacity(self.weight_count());

        for (i, &err) in output_error.iter().enumerate() {
            for (j, &inp) in self.last_input.iter().enumerate() {
                gradients.push(err * inp);
            }
            gradients.push(err);
        }

        gradients
    }

    pub fn apply_gradients(&mut self, gradients: &[f64]) {
        let mut idx = 0;

        for i in 0..self.output_size {
            for j in 0..self.input_size {
                if idx < gradients.len() {
                    let grad = gradients[idx];
                    self.velocity_w[i][j] =
                        self.momentum * self.velocity_w[i][j] - self.learning_rate * grad;
                    self.weights[i][j] += self.velocity_w[i][j];
                    idx += 1;
                }
            }
        }

        for i in 0..self.output_size {
            if idx < gradients.len() {
                let grad = gradients[idx];
                self.velocity_b[i] = self.momentum * self.velocity_b[i] - self.learning_rate * grad;
                self.biases[i] += self.velocity_b[i];
                idx += 1;
            }
        }
    }

    pub fn reset(&mut self) {
        self.last_input.clear();
        self.last_output.clear();
        self.velocity_w.iter_mut().for_each(|v| v.fill(0.0));
        self.velocity_b.fill(0.0);
    }

    pub fn set_learning_rate(&mut self, lr: f64) {
        self.learning_rate = lr;
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvolutionalLayer {
    pub input_channels: usize,
    pub output_channels: usize,
    pub kernel_size: usize,
    pub stride: usize,
    pub weights: Vec<Vec<Vec<f64>>>,
    pub biases: Vec<f64>,
    pub last_output: Vec<f64>,
    learning_rate: f64,
}

impl ConvolutionalLayer {
    pub fn new(input_channels: usize, output_channels: usize, kernel_size: usize) -> Self {
        let mut rng_seed = input_channels as u64 * output_channels as u64;

        let weights: Vec<Vec<Vec<f64>>> = (0..output_channels)
            .map(|_| {
                rng_seed = rng_seed.wrapping_mul(1103515245).wrapping_add(12345);
                (0..input_channels)
                    .map(|_| {
                        (0..kernel_size * kernel_size)
                            .map(|_| {
                                rng_seed = rng_seed.wrapping_mul(1103515245).wrapping_add(12345);
                                (rng_seed as i64).abs() as f64 / (i64::MAX as f64) * 0.1
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect();

        Self {
            input_channels,
            output_channels,
            kernel_size,
            stride: 1,
            weights,
            biases: vec![0.0; output_channels],
            last_output: Vec::new(),
            learning_rate: 0.001,
        }
    }

    pub fn forward(&mut self, input: &[f64]) -> Vec<f64> {
        self.last_output = input.to_vec();
        input.to_vec()
    }

    pub fn output_size(&self) -> usize {
        self.output_channels * self.kernel_size * self.kernel_size
    }

    pub fn weight_count(&self) -> usize {
        self.input_channels * self.output_channels * self.kernel_size * self.kernel_size
            + self.output_channels
    }

    pub fn compute_gradients(&mut self, _input: &[f64], _target: &[f64]) -> Vec<f64> {
        vec![0.0; self.weight_count()]
    }

    pub fn apply_gradients(&mut self, _gradients: &[f64]) {}

    pub fn reset(&mut self) {
        self.last_output.clear();
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecurrentLayer {
    pub input_size: usize,
    pub hidden_size: usize,
    pub weights_input: Vec<Vec<f64>>,
    pub weights_hidden: Vec<Vec<f64>>,
    pub biases: Vec<f64>,
    pub hidden_state: Vec<f64>,
    last_input: Vec<f64>,
    learning_rate: f64,
}

impl RecurrentLayer {
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        let mut rng_seed = input_size as u64 * hidden_size as u64;

        let weights_input: Vec<Vec<f64>> = (0..hidden_size)
            .map(|_| {
                rng_seed = rng_seed.wrapping_mul(1103515245).wrapping_add(12345);
                (0..input_size)
                    .map(|_| {
                        rng_seed = rng_seed.wrapping_mul(1103515245).wrapping_add(12345);
                        Self::xorshift_f64(rng_seed) * 0.1
                    })
                    .collect()
            })
            .collect();

        let weights_hidden: Vec<Vec<f64>> = (0..hidden_size)
            .map(|_| {
                rng_seed = rng_seed.wrapping_mul(1103515245).wrapping_add(12345);
                (0..hidden_size)
                    .map(|_| {
                        rng_seed = rng_seed.wrapping_mul(1103515245).wrapping_add(12345);
                        Self::xorshift_f64(rng_seed) * 0.1
                    })
                    .collect()
            })
            .collect();

        Self {
            input_size,
            hidden_size,
            weights_input,
            weights_hidden,
            biases: vec![0.0; hidden_size],
            hidden_state: vec![0.0; hidden_size],
            last_input: Vec::new(),
            learning_rate: 0.001,
        }
    }

    fn xorshift_f64(seed: u64) -> f64 {
        let mut x = seed;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        (x as i64).abs() as f64 / (i64::MAX as f64) * 2.0 - 1.0
    }

    pub fn forward(&mut self, input: &[f64]) -> Vec<f64> {
        self.last_input = input.to_vec();

        let new_state: Vec<f64> = (0..self.hidden_size)
            .map(|i| {
                let mut sum = self.biases[i];

                sum += self.weights_input[i]
                    .iter()
                    .zip(input.iter())
                    .map(|(w, x)| w * x)
                    .sum::<f64>();

                sum += self.weights_hidden[i]
                    .iter()
                    .zip(self.hidden_state.iter())
                    .map(|(w, h)| w * h)
                    .sum::<f64>();

                sum.tanh()
            })
            .collect();

        self.hidden_state = new_state.clone();
        new_state
    }

    pub fn output_size(&self) -> usize {
        self.hidden_size
    }

    pub fn weight_count(&self) -> usize {
        self.input_size * self.hidden_size + self.hidden_size * self.hidden_size + self.hidden_size
    }

    pub fn compute_gradients(&mut self, _input: &[f64], _target: &[f64]) -> Vec<f64> {
        vec![0.0; self.weight_count()]
    }

    pub fn apply_gradients(&mut self, _gradients: &[f64]) {}

    pub fn reset(&mut self) {
        self.hidden_state.fill(0.0);
        self.last_input.clear();
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LSTM {
    pub input_size: usize,
    pub hidden_size: usize,
    pub weights_input: Vec<Vec<f64>>,
    pub weights_hidden: Vec<Vec<f64>>,
    pub biases: Vec<f64>,
    pub hidden_state: Vec<f64>,
    pub cell_state: Vec<f64>,
    learning_rate: f64,
}

impl LSTM {
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        let total_weights = (input_size + hidden_size) * hidden_size * 4;
        let mut rng_seed = input_size as u64 * hidden_size as u64;

        let weights_input: Vec<Vec<f64>> = (0..hidden_size * 4)
            .map(|_| {
                rng_seed = rng_seed.wrapping_mul(1103515245).wrapping_add(12345);
                (0..input_size)
                    .map(|_| {
                        rng_seed = rng_seed.wrapping_mul(1103515245).wrapping_add(12345);
                        Self::xorshift_f64(rng_seed) * 0.1
                    })
                    .collect()
            })
            .collect();

        let weights_hidden: Vec<Vec<f64>> = (0..hidden_size * 4)
            .map(|_| {
                rng_seed = rng_seed.wrapping_mul(1103515245).wrapping_add(12345);
                (0..hidden_size)
                    .map(|_| {
                        rng_seed = rng_seed.wrapping_mul(1103515245).wrapping_add(12345);
                        Self::xorshift_f64(rng_seed) * 0.1
                    })
                    .collect()
            })
            .collect();

        Self {
            input_size,
            hidden_size,
            weights_input,
            weights_hidden,
            biases: vec![0.0; hidden_size * 4],
            hidden_state: vec![0.0; hidden_size],
            cell_state: vec![0.0; hidden_size],
            learning_rate: 0.001,
        }
    }

    fn xorshift_f64(seed: u64) -> f64 {
        let mut x = seed;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        (x as i64).abs() as f64 / (i64::MAX as f64) * 2.0 - 1.0
    }

    pub fn forward(&mut self, input: &[f64]) -> Vec<f64> {
        let gates: Vec<Vec<f64>> = (0..4)
            .map(|g| {
                (0..self.hidden_size)
                    .map(|h| {
                        let mut sum = self.biases[g * self.hidden_size + h];

                        for (i, &inp) in input.iter().enumerate() {
                            sum += self.weights_input[g * self.hidden_size + h][i] * inp;
                        }

                        for (i, &state) in self.hidden_state.iter().enumerate() {
                            sum += self.weights_hidden[g * self.hidden_size + h][i] * state;
                        }

                        sum
                    })
                    .collect()
            })
            .collect();

        let forget_gate: Vec<f64> = gates[0].iter().map(|x| Self::sigmoid(*x)).collect();
        let input_gate: Vec<f64> = gates[1].iter().map(|x| Self::sigmoid(*x)).collect();
        let output_gate: Vec<f64> = gates[2].iter().map(|x| Self::sigmoid(*x)).collect();
        let cell_input: Vec<f64> = gates[3].iter().map(|x| x.tanh()).collect();

        let new_cell: Vec<f64> = (0..self.hidden_size)
            .map(|i| forget_gate[i] * self.cell_state[i] + input_gate[i] * cell_input[i])
            .collect();

        let new_hidden: Vec<f64> = (0..self.hidden_size)
            .map(|i| output_gate[i] * new_cell[i].tanh())
            .collect();

        self.cell_state = new_cell;
        self.hidden_state = new_hidden.clone();
        new_hidden
    }

    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    pub fn output_size(&self) -> usize {
        self.hidden_size
    }

    pub fn weight_count(&self) -> usize {
        4 * (self.input_size * self.hidden_size
            + self.hidden_size * self.hidden_size
            + self.hidden_size)
    }

    pub fn compute_gradients(&mut self, _input: &[f64], _target: &[f64]) -> Vec<f64> {
        vec![0.0; self.weight_count()]
    }

    pub fn apply_gradients(&mut self, _gradients: &[f64]) {}

    pub fn reset(&mut self) {
        self.hidden_state.fill(0.0);
        self.cell_state.fill(0.0);
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Attention {
    pub size: usize,
    pub num_heads: usize,
    pub head_size: usize,
    pub query_weights: Vec<Vec<f64>>,
    pub key_weights: Vec<Vec<f64>>,
    pub value_weights: Vec<Vec<f64>>,
    pub output_weights: Vec<Vec<f64>>,
    last_output: Vec<f64>,
    learning_rate: f64,
}

impl Attention {
    pub fn new(size: usize, num_heads: usize) -> Self {
        let head_size = size / num_heads;
        let mut rng_seed = size as u64 * num_heads as u64;

        let mut make_weights = |dim: usize| -> Vec<Vec<f64>> {
            (0..dim)
                .map(|_| {
                    rng_seed = rng_seed.wrapping_mul(1103515245).wrapping_add(12345);
                    (0..size)
                        .map(|_| {
                            rng_seed = rng_seed.wrapping_mul(1103515245).wrapping_add(12345);
                            Self::xorshift_f64(rng_seed) * 0.1
                        })
                        .collect()
                })
                .collect()
        };

        Self {
            size,
            num_heads,
            head_size,
            query_weights: make_weights(size),
            key_weights: make_weights(size),
            value_weights: make_weights(size),
            output_weights: make_weights(size),
            last_output: Vec::new(),
            learning_rate: 0.001,
        }
    }

    fn xorshift_f64(seed: u64) -> f64 {
        let mut x = seed;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        (x as i64).abs() as f64 / (i64::MAX as f64) * 2.0 - 1.0
    }

    pub fn forward(&mut self, input: &[f64]) -> Vec<f64> {
        let query = Self::matmul(&self.query_weights, input);
        let key = Self::matmul(&self.key_weights, input);
        let value = Self::matmul(&self.value_weights, input);

        let attention_scores: Vec<f64> = query
            .iter()
            .zip(key.iter())
            .map(|(q, k)| q * k / (self.head_size as f64).sqrt())
            .collect();

        let attention_weights: Vec<f64> = Self::softmax(&attention_scores);

        let output: Vec<f64> = value
            .iter()
            .zip(attention_weights.iter())
            .map(|(v, w)| v * w)
            .collect();

        let final_output = Self::matmul(&self.output_weights, &output);

        self.last_output = final_output.clone();
        final_output
    }

    fn matmul(matrix: &[Vec<f64>], vector: &[f64]) -> Vec<f64> {
        matrix
            .iter()
            .map(|row| row.iter().zip(vector.iter()).map(|(m, v)| m * v).sum())
            .collect()
    }

    fn softmax(v: &[f64]) -> Vec<f64> {
        let max_val = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = v.iter().map(|x| (x - max_val).exp()).collect();
        let sum: f64 = exps.iter().sum();
        exps.iter().map(|x| x / sum).collect()
    }

    pub fn output_size(&self) -> usize {
        self.size
    }

    pub fn weight_count(&self) -> usize {
        4 * self.size * self.size
    }

    pub fn compute_gradients(&mut self, _input: &[f64], _target: &[f64]) -> Vec<f64> {
        vec![0.0; self.weight_count()]
    }

    pub fn apply_gradients(&mut self, _gradients: &[f64]) {}

    pub fn reset(&mut self) {
        self.last_output.clear();
    }
}
