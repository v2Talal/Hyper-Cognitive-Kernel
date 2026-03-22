//! Natural Language Processing Module
//!
//! Provides language understanding and generation capabilities for cognitive agents.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NLPConfig {
    pub vocab_size: usize,
    pub embedding_dim: usize,
    pub max_sequence_length: usize,
    pub use_attention: bool,
    pub language: Language,
}

impl Default for NLPConfig {
    fn default() -> Self {
        Self {
            vocab_size: 10000,
            embedding_dim: 128,
            max_sequence_length: 256,
            use_attention: true,
            language: Language::English,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Language {
    English,
    Arabic,
    Chinese,
    Multilingual,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tokenizer {
    vocab: HashMap<String, usize>,
    reverse_vocab: HashMap<usize, String>,
    unk_token: String,
    pad_token: String,
    bos_token: String,
    eos_token: String,
    special_tokens: Vec<String>,
}

impl Tokenizer {
    pub fn new(vocab_size: usize) -> Self {
        let mut vocab = HashMap::new();
        let mut reverse_vocab = HashMap::new();

        let special_tokens = vec![
            "<PAD>".to_string(),
            "<UNK>".to_string(),
            "<BOS>".to_string(),
            "<EOS>".to_string(),
            "<SEP>".to_string(),
        ];

        for (i, token) in special_tokens.iter().enumerate() {
            vocab.insert(token.clone(), i);
            reverse_vocab.insert(i, token.clone());
        }

        Self {
            vocab,
            reverse_vocab,
            unk_token: "<UNK>".to_string(),
            pad_token: "<PAD>".to_string(),
            bos_token: "<BOS>".to_string(),
            eos_token: "<EOS>".to_string(),
            special_tokens,
        }
    }

    pub fn tokenize(&self, text: &str) -> Vec<String> {
        text.to_lowercase()
            .split(|c: char| !c.is_alphanumeric() && c != ' ')
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .collect()
    }

    pub fn encode(&mut self, text: &str, max_length: usize) -> Vec<usize> {
        let tokens = self.tokenize(text);
        let mut ids: Vec<usize> = tokens
            .iter()
            .map(|t| {
                let unk_id = self.vocab.get(&self.unk_token).copied().unwrap_or(1);
                match self.vocab.get(t) {
                    Some(&id) => id,
                    None => {
                        let new_id = self.vocab.len();
                        if new_id < 10000 {
                            self.vocab.insert(t.clone(), new_id);
                            self.reverse_vocab.insert(new_id, t.clone());
                            new_id
                        } else {
                            unk_id
                        }
                    }
                }
            })
            .collect();

        if ids.len() > max_length {
            ids = ids[..max_length].to_vec();
        }

        while ids.len() < max_length {
            let pad_id = self.vocab.get(&self.pad_token).copied().unwrap_or(0);
            ids.push(pad_id);
        }

        ids
    }

    pub fn decode(&self, ids: &[usize]) -> String {
        ids.iter()
            .filter(|&&id| id > 4)
            .filter_map(|&id| self.reverse_vocab.get(&id))
            .map(|s| s.as_str())
            .collect::<Vec<_>>()
            .join(" ")
    }

    pub fn add_token(&mut self, token: &str) -> usize {
        if !self.vocab.contains_key(token) {
            let id = self.vocab.len();
            self.vocab.insert(token.to_string(), id);
            self.reverse_vocab.insert(id, token.to_string());
            id
        } else {
            *self.vocab.get(token).unwrap()
        }
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Embedding {
    weights: Vec<Vec<f64>>,
    embedding_dim: usize,
    frozen: bool,
}

impl Embedding {
    pub fn new(vocab_size: usize, embedding_dim: usize) -> Self {
        let mut rng_seed = vocab_size as u64 * embedding_dim as u64;

        let weights: Vec<Vec<f64>> = (0..vocab_size)
            .map(|_| {
                rng_seed = rng_seed.wrapping_mul(1103515245).wrapping_add(12345);
                (0..embedding_dim)
                    .map(|_| {
                        rng_seed = rng_seed.wrapping_mul(1103515245).wrapping_add(12345);
                        Self::xorshift_f64(rng_seed) * 0.1
                    })
                    .collect()
            })
            .collect();

        Self {
            weights,
            embedding_dim,
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

    pub fn forward(&self, token_ids: &[usize]) -> Vec<Vec<f64>> {
        token_ids
            .iter()
            .map(|&id| {
                self.weights
                    .get(id % self.weights.len())
                    .cloned()
                    .unwrap_or_else(|| vec![0.0; self.embedding_dim])
            })
            .collect()
    }

    pub fn freeze(&mut self) {
        self.frozen = true;
    }

    pub fn unfreeze(&mut self) {
        self.frozen = false;
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionPooling {
    pub hidden_size: usize,
    query_weights: Vec<f64>,
    key_weights: Vec<f64>,
    value_weights: Vec<f64>,
}

impl AttentionPooling {
    pub fn new(hidden_size: usize) -> Self {
        let mut rng_seed = hidden_size as u64;

        let mut make_weights = |_| {
            rng_seed = rng_seed.wrapping_mul(1103515245).wrapping_add(12345);
            (0..hidden_size)
                .map(|_| {
                    rng_seed = rng_seed.wrapping_mul(1103515245).wrapping_add(12345);
                    Self::xorshift_f64(rng_seed) * 0.1
                })
                .collect()
        };

        Self {
            hidden_size,
            query_weights: make_weights(()),
            key_weights: make_weights(()),
            value_weights: make_weights(()),
        }
    }

    fn xorshift_f64(seed: u64) -> f64 {
        let mut x = seed;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        (x as i64).abs() as f64 / (i64::MAX as f64) * 2.0 - 1.0
    }

    pub fn forward(&self, hidden_states: &[Vec<f64>]) -> Vec<f64> {
        if hidden_states.is_empty() {
            return vec![0.0; self.hidden_size];
        }

        let scores: Vec<f64> = hidden_states
            .iter()
            .map(|h| {
                h.iter()
                    .zip(self.query_weights.iter())
                    .map(|(x, w)| x * w)
                    .sum()
            })
            .collect();

        let max_score = scores.iter().cloned().fold(f64::MIN, f64::max);
        let exp_scores: Vec<f64> = scores.iter().map(|s| (s - max_score).exp()).collect();
        let sum_exp: f64 = exp_scores.iter().sum();

        let weights: Vec<f64> = exp_scores.iter().map(|e| e / sum_exp).collect();

        let mut output = vec![0.0; self.hidden_size];

        for (h, &w) in hidden_states.iter().zip(weights.iter()) {
            for i in 0..self.hidden_size {
                output[i] += h[i] * w;
            }
        }

        output
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextEncoder {
    pub tokenizer: Tokenizer,
    pub embedding: Embedding,
    pub attention: AttentionPooling,
    config: NLPConfig,
}

impl TextEncoder {
    pub fn new(config: NLPConfig) -> Self {
        let tokenizer = Tokenizer::new(config.vocab_size);
        let embedding = Embedding::new(config.vocab_size, config.embedding_dim);
        let attention = AttentionPooling::new(config.embedding_dim);

        Self {
            tokenizer,
            embedding,
            attention,
            config,
        }
    }

    pub fn encode(&mut self, text: &str) -> Vec<f64> {
        let token_ids = self.tokenizer.encode(text, self.config.max_sequence_length);
        let hidden_states = self.embedding.forward(&token_ids);
        self.attention.forward(&hidden_states)
    }

    pub fn batch_encode(&mut self, texts: &[String]) -> Vec<Vec<f64>> {
        texts.iter().map(|t| self.encode(t)).collect()
    }
}

pub struct SentimentAnalyzer {
    encoder: TextEncoder,
    classifier_weights: Vec<Vec<f64>>,
}

impl SentimentAnalyzer {
    pub fn new() -> Self {
        let config = NLPConfig::default();
        let embedding_dim = config.embedding_dim;
        let encoder = TextEncoder::new(config);
        let mut rng_seed = 42u64;
        let classifier_weights: Vec<Vec<f64>> = (0..3)
            .map(|_| {
                rng_seed = rng_seed.wrapping_mul(1103515245).wrapping_add(12345);
                (0..embedding_dim)
                    .map(|_| {
                        rng_seed = rng_seed.wrapping_mul(1103515245).wrapping_add(12345);
                        Self::xorshift_f64(rng_seed) * 0.1
                    })
                    .collect()
            })
            .collect();

        Self {
            encoder,
            classifier_weights,
        }
    }

    fn xorshift_f64(seed: u64) -> f64 {
        let mut x = seed;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        (x as i64).abs() as f64 / (i64::MAX as f64) * 2.0 - 1.0
    }

    pub fn analyze(&mut self, text: &str) -> SentimentResult {
        let embedding = self.encoder.encode(text);

        let scores: Vec<f64> = self
            .classifier_weights
            .iter()
            .map(|w| embedding.iter().zip(w.iter()).map(|(e, c)| e * c).sum())
            .collect();

        let exp_scores: Vec<f64> = scores.iter().map(|s| s.exp()).collect();
        let sum_exp: f64 = exp_scores.iter().sum();
        let probs: Vec<f64> = exp_scores.iter().map(|e| e / sum_exp).collect();

        let sentiment = match probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        {
            Some((i, _)) => match i {
                0 => Sentiment::Negative,
                1 => Sentiment::Neutral,
                _ => Sentiment::Positive,
            },
            None => Sentiment::Neutral,
        };

        SentimentResult {
            sentiment,
            confidence: *probs
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap_or(&0.0),
            scores: probs,
        }
    }
}

impl Default for SentimentAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Sentiment {
    Positive,
    Neutral,
    Negative,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentimentResult {
    pub sentiment: Sentiment,
    pub confidence: f64,
    pub scores: Vec<f64>,
}

pub struct IntentClassifier {
    encoder: TextEncoder,
    intents: Vec<Intent>,
    weights: Vec<Vec<f64>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Intent {
    pub name: String,
    pub description: String,
    pub examples: Vec<String>,
}

impl IntentClassifier {
    pub fn new() -> Self {
        let encoder = TextEncoder::new(NLPConfig::default());

        Self {
            encoder,
            intents: Vec::new(),
            weights: Vec::new(),
        }
    }

    pub fn add_intent(&mut self, intent: Intent) {
        self.intents.push(intent);
    }

    pub fn classify(&mut self, text: &str) -> Option<(&Intent, f64)> {
        if self.intents.is_empty() {
            return None;
        }

        let embedding = self.encoder.encode(text);

        let mut best_intent = None;
        let mut best_score = f64::MIN;

        for (i, intent) in self.intents.iter().enumerate() {
            let score = if self.weights.len() > i {
                embedding
                    .iter()
                    .zip(self.weights[i].iter())
                    .map(|(e, w)| e * w)
                    .sum()
            } else {
                0.0
            };

            if score > best_score {
                best_score = score;
                best_intent = Some(intent);
            }
        }

        best_intent.map(|i| (i, best_score))
    }
}

impl Default for IntentClassifier {
    fn default() -> Self {
        Self::new()
    }
}

pub struct TextGenerator {
    encoder: TextEncoder,
    output_weights: Vec<Vec<f64>>,
    temperature: f64,
    max_length: usize,
}

impl TextGenerator {
    pub fn new() -> Self {
        let encoder = TextEncoder::new(NLPConfig::default());
        let output_weights = vec![vec![0.0; 128]; 10000];

        Self {
            encoder,
            output_weights,
            temperature: 1.0,
            max_length: 50,
        }
    }

    pub fn generate(&mut self, prompt: &str) -> String {
        let tokens = self.encoder.tokenizer.tokenize(prompt);
        let mut output = tokens.clone();

        for _ in 0..self.max_length {
            let token_ids = self.encoder.tokenizer.encode(&output.join(" "), 20);
            let embedding = self.encoder.embedding.forward(&token_ids);

            let last_hidden = embedding.last().cloned().unwrap_or_else(|| vec![0.0; 128]);

            let mut logits: Vec<f64> = self
                .output_weights
                .iter()
                .map(|w| last_hidden.iter().zip(w.iter()).map(|(e, w)| e * w).sum())
                .collect();

            for logit in &mut logits {
                *logit /= self.temperature;
            }

            let max_logit = logits.iter().cloned().fold(f64::MIN, f64::max);
            let exp_logits: Vec<f64> = logits.iter().map(|l| (l - max_logit).exp()).collect();
            let sum_exp: f64 = exp_logits.iter().sum();
            let probs: Vec<f64> = exp_logits.iter().map(|e| e / sum_exp).collect();

            let next_token_id = self.sample(&probs);

            if let Some(token) = self.encoder.tokenizer.reverse_vocab.get(&next_token_id) {
                if token == "<EOS>" || token == "<PAD>" {
                    break;
                }
                output.push(token.clone());
            }
        }

        output.join(" ")
    }

    fn sample(&self, probs: &[f64]) -> usize {
        let r = rand::random::<f64>();
        let mut cumsum = 0.0;

        for (i, &p) in probs.iter().enumerate() {
            cumsum += p;
            if r <= cumsum {
                return i;
            }
        }

        probs.len() - 1
    }

    pub fn set_temperature(&mut self, temp: f64) {
        self.temperature = temp.clamp(0.1, 2.0);
    }
}

impl Default for TextGenerator {
    fn default() -> Self {
        Self::new()
    }
}
