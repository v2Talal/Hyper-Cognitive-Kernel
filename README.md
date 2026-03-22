# Hyper-Cognitive Kernel

A production-ready cognitive agent framework in Rust implementing real-time online learning with Active Inference, Predictive Coding, and Deep Neural Networks.

## Description

Real-time cognitive agent framework: instant learning from single samples, no batch processing, adaptive learning rates, catastrophic forgetting prevention, and multi-agent distributed learning.

---

## Execution Flow

The `cognitive_cycle()` processes data through this pipeline:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        COGNITIVE CYCLE FLOW                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. PERCEPTION                                                      │
│     └─► Raw sensors → Feature extraction → Normalized input         │
│                                                                     │
│  2. MEMORY RETRIEVAL                                                │
│     └─► Query episodic memory → Retrieve similar episodes           │
│     └─► Query semantic memory → Get general patterns                │
│                                                                     │
│  3. PREDICTIVE CODING                                               │
│     └─► Generate predictions at each hierarchy level                │
│     └─► Compute prediction error (surprise)                         │
│                                                                     │
│  4. WORLD MODEL                                                     │
│     └─► Simulate environment consequences                           │
│     └─► Predict next state and rewards                              │
│                                                                     │
│  5. ATTENTION                                                       │
│     └─► Focus on salient features                                   │
│     └─► Modulate processing based on drives                         │
│                                                                     │
│  6. ACTION SELECTION                                                │
│     └─► Policy from procedural memory                               │
│     └─► RL integration for exploration/exploitation                 │
│     └─► Drive modulation                                            │
│                                                                     │
│  7. LEARNING                                                        │
│     └─► Online gradient update                                      │
│     └─► Adaptive learning rate adjustment                           │
│     └─► EWC for continual learning                                  │
│     └─► Memory consolidation                                        │
│                                                                     │
│  8. META-LEARNING                                                   │
│     └─► Adapt learning parameters                                   │
│     └─► Self-reflection and reporting                               │
│                                                                     │
│  9. MEMORY ENCODING                                                 │
│     └─► Store new episode in episodic memory                        │
│     └─► Update semantic patterns                                    │
│     └─► Reinforce procedural skills                                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Architecture Overview

```
┌────────────────────────────────────────────────────────────────────┐
│                         Hyper-Cognitive-Kernel                     │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│   ┌─────────────────────────────────────────────────────────────┐  │
│   │                     PLUGINS (Hot-swappable)                 │  │
│   ├─────────────────────────────────────────────────────────────┤  │
│   │  [Neural]         [Vision]        [NLP]          [RL]       │  │
│   │  LSTM/Attn       CNN/Edge        Tokenize       DQN/DDQN    │  │
│   │  Reservoir         Flow          Sentiment        SAC       │  │
│   └─────────────────────────────────────────────────────────────┘  │
│                                                                    │
│   ┌─────────────────────────────────────────────────────────────┐  │
│   │                    REAL-TIME LEARNING ENGINE                │  │
│   │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │  │
│   │  │ Online GD   │  │ Adaptive LR │  │ EWC / Memory Replay │  │  │
│   │  └─────────────┘  └─────────────┘  └─────────────────────┘  │  │
│   └─────────────────────────────────────────────────────────────┘  │
│                                                                    │
│   ┌─────────────────────────────────────────────────────────────┐  │
│   │                      CORE COGNITIVE SYSTEMS                 │  │
│   │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐     │  │
│   │  │Predictive│  │  World   │  │  Triple  │  │  Meta    │     │  │
│   │  │ Coding   │  │  Model   │  │  Memory  │  │ Learning │     │  │
│   │  └──────────┘  └──────────┘  └──────────┘  └──────────┘     │  │
│   │  ┌──────────┐  ┌──────────┐                                 │  │
│   │  │Attention │  │ Drives/  │                                 │  │
│   │  │          │  │Homeostas │                                 │  │
│   │  └──────────┘  └──────────┘                                 │  │
│   └─────────────────────────────────────────────────────────────┘  │
│                                                                    │
│   ┌─────────────────────────────────────────────────────────────┐  │
│   │                   ENVIRONMENT INTERFACE                     │  │
│   │      [MQTT]          [WebSocket]         [HTTP API]         │  │
│   └─────────────────────────────────────────────────────────────┘  │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
Hyper-Cognitive-Kernel/
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
    │                 PLUGINS (Advanced Modules)
    │              (Hot-swappable specialized capabilities)
    │
    ├── neural/            # Deep Learning Plugin
    │   ├── mod.rs         # Neural network exports
    │   ├── layers.rs      # Dense, LSTM, Attention, Reservoir layers
    │   ├── activations.rs # ReLU, Sigmoid, Tanh, Softmax, ELU
    │   └── optimizer.rs   # SGD, Adam, RMSprop optimizers
    │
    ├── continual/         # Continual Learning Plugin
    │   ├── mod.rs         # Continual learning exports
    │   ├── ewc.rs         # Elastic Weight Consolidation
    │   ├── memory_replay.rs  # Experience replay buffer
    │   ├── progressive.rs   # Progressive Neural Networks
    │   ├── packnet.rs       # Pack & Prune strategy
    │   └── synaptic.rs      # Synaptic Intelligence
    │
    ├── real_time/         # Real-time Learning Plugin
    │   ├── mod.rs         # Real-time learning exports
    │   ├── adaptive_lr.rs   # Per-layer adaptive learning rate
    │   └── online_trainer.rs  # Single-sample training
    │
    ├── distributed/       # Distributed Learning Plugin
    │   ├── mod.rs         # Distributed learning exports
    │   ├── learner.rs     # Federated learning coordinator
    │   ├── pubsub.rs      # Publish/Subscribe messaging
    │   └── gossip.rs      # Gossip protocol for peer sync
    │
    ├── vision/            # Vision Plugin
    │   └── mod.rs         # Image processing & CNN features
    │
    ├── nlp/              # NLP Plugin
    │   └── mod.rs         # Tokenizer, Embedding, Sentiment analysis
    │
    ├── rl_integration/    # Reinforcement Learning Plugin
    │   └── mod.rs         # DQN, DDQN, SAC + Active Inference
    │
    ├── environment/       # Environment Interface Plugin
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

### Core Modules (Always Active)

| Module | Files | Description |
|--------|-------|-------------|
| **core** | agent.rs, config.rs | Main agent orchestrator and configuration |
| **cognition** | predictive_coding.rs, world_model.rs, attention.rs | Predictive coding, world simulation, attention |
| **memory** | episodic.rs, semantic.rs, procedural.rs | Triple-layer memory system |
| **meta** | learning_controller.rs, self_reflection.rs | Self-modifying learning, introspection |
| **homeostasis** | drives.rs, allostasis.rs | Motivation and drive regulation |

### Plugins (Hot-swappable)

| Plugin | Files | Description |
|--------|-------|-------------|
| **neural** | layers.rs, activations.rs, optimizer.rs | Deep neural networks (LSTM, Attention, Reservoir) |
| **continual** | ewc.rs, memory_replay.rs, progressive.rs, packnet.rs, synaptic.rs | Catastrophic forgetting prevention |
| **real_time** | adaptive_lr.rs, online_trainer.rs | Single-sample learning, adaptive rates |
| **distributed** | learner.rs, pubsub.rs, gossip.rs | Multi-agent federated learning |
| **vision** | mod.rs | Image feature extraction |
| **nlp** | mod.rs | Tokenization, embeddings, sentiment |
| **rl_integration** | mod.rs | Deep RL with Active Inference |
| **environment** | mqtt_client.rs, websocket_client.rs, http_api.rs, message_bus.rs | External system integration |

---

## Real-World Scenarios

### Scenario 1: IoT Sensor Network

```rust
// Agent learns from MQTT sensor stream in real-time
use hyper_cognitive_kernel::environment::{MQTTClient, MQTTConfig};
use hyper_cognitive_kernel::{CognitiveAgent, AgentConfig};

let config = MQTTConfig::default();
let mut mqtt = MQTTClient::new(config);
mqtt.connect("tcp://sensors.local:1883").unwrap();
mqtt.subscribe("sensors/temperature/#");

let mut agent = CognitiveAgent::new(1, AgentConfig::new());

for message in mqtt.receive_stream() {
    let sensors = parse_temperature_data(&message);
    let actions = agent.cognitive_cycle(&sensors, 0.0);
    // Agent learns continuously - no batching!
}
```

### Scenario 2: Multi-Agent Coordination

```rust
// Multiple agents share knowledge via distributed learning
use hyper_cognitive_kernel::distributed::{DistributedLearner, GossipProtocol};

let mut learner = DistributedLearner::new("agent_1".into(), Default::default());
let gossip = GossipProtocol::new(Default::default());

// Share model updates with peers
loop {
    let local_update = learner.create_update(&gradient);
    gossip.broadcast(local_update);
    
    for update in gossip.receive_messages() {
        learner.receive_update(update);
    }
}
```

### Scenario 3: Vision-Based Navigation

```rust
// Agent learns from camera feed
use hyper_cognitive_kernel::vision::{Image, FeatureExtractor};

let mut extractor = FeatureExtractor::new();
let image = Image::from_raw(camera_data, 640, 480, 3);
let features = extractor.extract_features(&image);

// Combine with other sensors
let mut combined_input = features;
combined_input.extend(lidar_data);
combined_input.extend(gps_data);

let action = agent.cognitive_cycle(&combined_input, reward);
```

---

## Performance Benchmarks

| Metric | Value | Conditions |
|--------|-------|------------|
| **Inference Latency** | < 1ms | Single forward pass, 8 inputs |
| **Learning Latency** | < 2ms | Single sample update |
| **Memory per Agent** | ~50MB | Full cognitive system loaded |
| **Throughput** | 10,000 samples/sec | Continuous streaming |
| **Cold Start** | < 100ms | Agent initialization |

Tested on: AMD Ryzen 7 5800X, 32GB RAM

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

## Quick Start

```rust
use hyper_cognitive_kernel::{CognitiveAgent, AgentConfig};

let config = AgentConfig::new()
    .with_learning_rate(0.01)
    .with_prediction_depth(3);

let mut agent = CognitiveAgent::new(1, config);

// Continuous learning loop
loop {
    let sensors = get_next_sensor_reading();
    let reward = calculate_reward(&sensors);
    
    // Learn from single sample immediately - no batching!
    let actions = agent.cognitive_cycle(&sensors, reward);
    
    execute_actions(actions);
}
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
