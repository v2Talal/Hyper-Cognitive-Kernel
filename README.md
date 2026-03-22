# Hyper-Cognitive Kernel

A production-ready cognitive agent framework in Rust implementing real-time online learning with Active Inference, Predictive Coding, and Deep Neural Networks.

## Overview

Hyper-Cognitive Kernel is designed for autonomous agents that learn continuously from streaming data without batch processing. It combines cognitive science principles with modern machine learning to create adaptive, self-improving agents.

## Core Features

### Real-time Online Learning
- **Instant Learning**: Process single samples immediately
- **No Batching**: Zero latency between observation and learning
- **Adaptive Rates**: Automatic per-layer learning rate adjustment
- **Continuous Improvement**: Models evolve with every interaction

### Cognitive Architecture
| Component | Description |
|-----------|-------------|
| **Predictive Coding** | Hierarchical predictions with error-driven updates |
| **World Model** | Internal environment simulation |
| **Triple Memory** | Episodic, semantic, and procedural memory systems |
| **Meta-Learning** | Self-modifying hyperparameters |
| **Homeostatic Drives** | Survival, curiosity, and efficiency motivation |

### Advanced Modules

| Module | Capabilities |
|--------|-------------|
| **Neural Networks** | Dense, LSTM, Attention, Reservoir layers |
| **Continual Learning** | EWC, Memory Replay, Progressive Networks, PackNet |
| **Distributed Learning** | Federated learning, Gossip protocols, Pub/Sub |
| **Vision** | CNN features, edge detection, spatial analysis |
| **NLP** | Tokenization, embeddings, sentiment analysis |
| **RL Integration** | DQN, DDQN, SAC with Active Inference |
| **Environment** | MQTT, WebSocket, HTTP API clients |

## Quick Start

```rust
use hyper_cognitive_kernel::{CognitiveAgent, AgentConfig};

let config = AgentConfig::new()
    .with_learning_rate(0.01)
    .with_prediction_depth(3);

let mut agent = CognitiveAgent::new(1, config);

let sensors = vec![0.5, 0.3, 0.8, 0.2, 0.6, 0.4, 0.7, 0.1];
let reward = 1.0;

let actions = agent.cognitive_cycle(&sensors, reward);
```

## Examples

### Neural Networks
```rust
use hyper_cognitive_kernel::neural::{NeuralNetwork, ActivationFunction};

let mut net = NeuralNetwork::new(4, 2);
net.add_dense(64, ActivationFunction::ReLU);
net.add_lstm(32);
net.add_attention(4);

let loss = net.online_train(&input, &target);
```

### Continual Learning
```rust
use hyper_cognitive_kernel::continual::ContinualLearner;

let mut learner = ContinualLearner::with_ewc(5000.0);
learner.add_replay_sample(input, target, task_id);
```

### Natural Language
```rust
use hyper_cognitive_kernel::nlp::{Tokenizer, SentimentAnalyzer};

let mut tokenizer = Tokenizer::new(10000);
let ids = tokenizer.encode("Hello world", 20);

let mut analyzer = SentimentAnalyzer::new();
let result = analyzer.analyze("Great product!");
```

### Vision Processing
```rust
use hyper_cognitive_kernel::vision::{Image, FeatureExtractor};

let image = Image::from_raw(data, 224, 224, 3);
let features = extractor.extract_features(&image);
```

### Environment Interface
```rust
use hyper_cognitive_kernel::environment::{MQTTClient, MQTTConfig};

let config = MQTTConfig::default();
let client = MQTTClient::new(config);
```

## Installation

Add to `Cargo.toml`:

```toml
[dependencies]
hyper_cognitive_kernel = "1.0"
```

## Building

```bash
cargo build --release
cargo test
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    HYPER-COGNITIVE KERNEL                    │
├─────────────────────────────────────────────────────────────┤
│  ADVANCED MODULES                                            │
│  ├── Neural Networks (LSTM, Attention, Reservoir)             │
│  ├── Continual Learning (EWC, Memory Replay, PackNet)        │
│  ├── Distributed Learning (Federated, Gossip)                 │
│  ├── Vision Processing (CNN, Optical Flow)                    │
│  ├── NLP (Embeddings, Sentiment, Intent)                    │
│  └── RL + Active Inference                                   │
├─────────────────────────────────────────────────────────────┤
│  REAL-TIME LEARNING ENGINE                                   │
│  ├── Online Gradient Descent                                  │
│  ├── Adaptive Learning Rate                                   │
│  └── Elastic Weight Consolidation                             │
├─────────────────────────────────────────────────────────────┤
│  CORE COGNITIVE SYSTEMS                                      │
│  ├── Predictive Coding Network                                │
│  ├── World Model                                            │
│  ├── Triple-Layer Memory                                    │
│  ├── Meta-Learning                                          │
│  └── Homeostatic Drives                                     │
└─────────────────────────────────────────────────────────────┘
```

## Technical Details

- **Language**: Rust 2021 Edition
- **Dependencies**: serde, parking_lot, rayon, tokio, reqwest
- **Serialization**: Full JSON support via serde
- **Async**: Tokio for async operations
- **Parallelism**: Rayon for data parallelism

## Performance

- Low-latency inference optimized
- Memory-efficient streaming operations
- No batch processing overhead
- Parallel processing via Rayon

## Testing

```bash
cargo test --lib
```

## Contributing

1. Follow Rust idioms and style
2. Ensure tests pass
3. Minimize compiler warnings

## License

MIT License - see LICENSE file

## References

Based on principles from:
- Active Inference (Karl Friston)
- Hierarchical Predictive Processing
- Continual Learning (EWC, PackNet)
- Deep Reinforcement Learning
