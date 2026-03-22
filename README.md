# Hyper-Cognitive Kernel

A production-ready cognitive agent framework in Rust implementing real-time online learning with Active Inference, Predictive Coding, and Deep Neural Networks.

## Description

Real-time cognitive agent framework: instant learning from single samples, no batch processing, adaptive learning rates, catastrophic forgetting prevention, and multi-agent distributed learning.

---

## Project Structure

```
hyper_cognitive_kernel/
│
├── Cargo.toml              # Project configuration & dependencies
├── Cargo.lock              # Locked dependency versions
├── LICENSE                # MIT License
├── README.md              # This file
├── .gitignore             # Git ignore rules
│
└── src/                   # Source code (47 Rust files)
    │
    ├── lib.rs             # Library entry point - exports all modules
    ├── main.rs            # Demo program - showcases all features
    │
    ├── ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    │                     CORE MODULES
    │         (Essential cognitive agent components)
    │
    ├── core/
    │   ├── mod.rs         # Core module exports
    │   ├── agent.rs       # CognitiveAgent - main agent orchestrator
    │   └── config.rs      # AgentConfig - configuration parameters
    │
    ├── cognition/
    │   ├── mod.rs         # Cognition module exports
    │   ├── predictive_coding.rs  # Hierarchical prediction with error signals
    │   ├── world_model.rs        # Internal environment simulation
    │   └── attention.rs          # Selective attention mechanism
    │
    ├── memory/
    │   ├── mod.rs         # Memory system exports
    │   ├── episodic.rs     # Episode storage (experiences)
    │   ├── semantic.rs     # General knowledge patterns
    │   └── procedural.rs   # Skills and action policies
    │
    ├── meta/
    │   ├── mod.rs         # Meta-learning exports
    │   ├── learning_controller.rs  # Adaptive learning rate controller
    │   └── self_reflection.rs     # Self-analysis and reporting
    │
    └── homeostasis/
        ├── mod.rs         # Drive system exports
        ├── drives.rs       # Survival, curiosity, efficiency drives
        └── allostasis.rs  # Allostatic load management
    │
    ├── ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    │                 ADVANCED MODULES
    │         (Deep learning & specialized capabilities)
    │
    ├── neural/
    │   ├── mod.rs         # Neural network exports
    │   ├── layers.rs      # Dense, LSTM, Attention, Reservoir layers
    │   ├── activations.rs # ReLU, Sigmoid, Tanh, Softmax, ELU
    │   └── optimizer.rs   # SGD, Adam, RMSprop optimizers
    │
    ├── continual/
    │   ├── mod.rs         # Continual learning exports
    │   ├── ewc.rs         # Elastic Weight Consolidation
    │   ├── memory_replay.rs  # Experience replay buffer
    │   ├── progressive.rs   # Progressive Neural Networks
    │   ├── packnet.rs       # Pack & Prune strategy
    │   └── synaptic.rs      # Synaptic Intelligence
    │
    ├── real_time/
    │   ├── mod.rs         # Real-time learning exports
    │   ├── adaptive_lr.rs   # Per-layer adaptive learning rate
    │   └── online_trainer.rs  # Single-sample training
    │
    ├── distributed/
    │   ├── mod.rs         # Distributed learning exports
    │   ├── learner.rs     # Federated learning coordinator
    │   ├── pubsub.rs      # Publish/Subscribe messaging
    │   └── gossip.rs      # Gossip protocol for peer sync
    │
    ├── vision/
    │   └── mod.rs         # Image processing & CNN features
    │
    ├── nlp/
    │   └── mod.rs         # Tokenizer, Embedding, Sentiment analysis
    │
    ├── rl_integration/
    │   └── mod.rs         # DQN, DDQN, SAC + Active Inference
    │
    ├── environment/
    │   ├── mod.rs         # Environment interface exports
    │   ├── mqtt_client.rs    # MQTT protocol client
    │   ├── websocket_client.rs  # WebSocket client
    │   ├── http_api.rs       # REST API client
    │   └── message_bus.rs    # Inter-agent messaging
    │
    └── utils/
        ├── mod.rs         # Utilities exports
        ├── math.rs        # Math helpers (sigmoid, softmax, etc)
        └── logger.rs      # Logging functionality
```

---

## Module Descriptions

### Core Modules

| Module | Files | Description |
|--------|-------|-------------|
| **core** | agent.rs, config.rs | Main agent orchestrator and configuration |
| **cognition** | predictive_coding.rs, world_model.rs, attention.rs | Predictive coding, world simulation, attention |
| **memory** | episodic.rs, semantic.rs, procedural.rs | Triple-layer memory system |
| **meta** | learning_controller.rs, self_reflection.rs | Self-modifying learning, introspection |
| **homeostasis** | drives.rs, allostasis.rs | Motivation and drive regulation |

### Advanced Modules

| Module | Files | Description |
|--------|-------|-------------|
| **neural** | layers.rs, activations.rs, optimizer.rs | Deep neural networks (LSTM, Attention, Reservoir) |
| **continual** | ewc.rs, memory_replay.rs, progressive.rs, packnet.rs, synaptic.rs | Catastrophic forgetting prevention |
| **real_time** | adaptive_lr.rs, online_trainer.rs | Single-sample learning, adaptive rates |
| **distributed** | learner.rs, pubsub.rs, gossip.rs | Multi-agent federated learning |
| **vision** | mod.rs | Image feature extraction |
| **nlp** | mod.rs | Tokenization, embeddings, sentiment |
| **rl_integration** | mod.rs | Deep RL with Active Inference |
| **environment** | mqtt_client.rs, websocket_client.rs, http_api.rs, message_bus.rs | External system integration |
| **utils** | math.rs, logger.rs | Helper functions |

---

## File Count

```
Total Rust Files: 47
├── Core Modules: 13 files
├── Advanced Modules: 22 files
├── Utilities: 4 files
├── Entry Points: 2 files (lib.rs, main.rs)
└── Tests: Integrated in modules
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    HYPER-COGNITIVE KERNEL                   │
├─────────────────────────────────────────────────────────────┤
│  ADVANCED MODULES                                           │
│  ├── Neural Networks (LSTM, Attention, Reservoir)           │
│  ├── Continual Learning (EWC, Memory Replay, PackNet)       │
│  ├── Distributed Learning (Federated, Gossip)               │
│  ├── Vision Processing (CNN, Optical Flow)                  │
│  ├── NLP (Embeddings, Sentiment, Intent)                    │
│  ├── RL + Active Inference                                  │
│  └── Environment Interface (MQTT, WebSocket, API)           │
├─────────────────────────────────────────────────────────────┤
│  REAL-TIME LEARNING ENGINE                                  │
│  ├── Online Gradient Descent                                │
│  ├── Adaptive Learning Rate                                 │
│  └── Elastic Weight Consolidation                           │
├─────────────────────────────────────────────────────────────┤
│  CORE COGNITIVE SYSTEMS                                     │
│  ├── Predictive Coding Network                              │
│  ├── World Model                                            │
│  ├── Triple-Layer Memory                                    │
│  ├── Meta-Learning                                          │
│  └── Homeostatic Drives                                     │
└─────────────────────────────────────────────────────────────┘
```

---

## Installation

```bash
git clone https://github.com/v2Talal/Hyper-Cognitive-Kernel.git
cd hyper-cognitive-kernel
cargo build --release
cargo run --release
cargo test --lib
```

---

## Usage Example

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

---

## Technical Stack

- **Language**: Rust Edition
- **Dependencies**: serde, parking_lot, rayon, tokio, reqwest
- **Serialization**: JSON via serde
- **Async**: Tokio runtime
- **Parallelism**: Rayon data parallelism

---

## License

MIT License - see LICENSE file
