//! # Hyper-Cognitive Kernel
//!
//! A cognitive agent framework with real-time learning and distributed intelligence.
//!
//! ## Features
//!
//! - **Predictive Coding** - Hierarchical predictions with error-based learning
//! - **World Modeling** - Internal simulation of environmental consequences
//! - **Triple-Layer Memory** - Episodic, semantic, and procedural memory
//! - **Neural Networks** - LSTM, Attention, Reservoir for deep learning
//! - **Continual Learning** - EWC, Memory Replay to prevent forgetting
//! - **Distributed Learning** - Federated learning and gossip protocols
//! - **RL + Active Inference** - Intelligent action selection

#![warn(missing_docs)]
#![warn(rust_2018_idioms)]
#![forbid(unsafe_code)]
#![allow(unused_imports)]

pub mod cognition;
pub mod core;
pub mod homeostasis;
pub mod memory;
pub mod meta;
pub mod utils;

pub mod continual;
pub mod distributed;
pub mod environment;
pub mod neural;
pub mod nlp;
pub mod real_time;
pub mod rl_integration;
pub mod vision;

pub use core::agent::CognitiveAgent;
pub use core::config::AgentConfig;
pub use homeostasis::drives::DriveSystem;
pub use memory::MemorySystem;
pub use meta::self_reflection::SelfReflection;
pub use meta::MetaLearner;

pub use continual::ContinualLearner;
pub use distributed::{DistributedLearner, GossipProtocol, PubSub};
pub use environment::{EnvironmentInterface, MQTTClient, WebSocketClient, HTTPAPI};
pub use neural::{ActivationFunction, Layer, NeuralNetwork};
pub use nlp::{SentimentAnalyzer, TextEncoder, Tokenizer};
pub use real_time::{OnlineTrainer, RealTimeLearningController};
pub use rl_integration::{RLAgent, RLConfig};
pub use vision::{FeatureExtractor, Image};

pub mod prelude {
    pub use crate::cognition::{AttentionSystem, PredictiveCoder, WorldModel};
    pub use crate::continual::ContinualLearner;
    pub use crate::core::{AgentConfig, CognitiveAgent};
    pub use crate::homeostasis::{Allostasis, DriveSystem};
    pub use crate::memory::{EpisodicMemory, MemorySystem, ProceduralMemory, SemanticMemory};
    pub use crate::meta::{MetaLearner, SelfReflection};
    pub use crate::neural::{ActivationFunction, NeuralNetwork};
    pub use crate::real_time::{OnlineTrainer, RealTimeLearningController};
    pub use crate::utils::{Logger, MathUtils};
}
