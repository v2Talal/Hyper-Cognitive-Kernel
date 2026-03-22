//! Distributed Learning Module
//!
//! Provides multi-agent collaborative learning capabilities.

pub mod gossip;
pub mod learner;
pub mod pubsub;

pub use gossip::{GossipConfig, GossipMessage, GossipProtocol};
pub use learner::{
    AggregationMethod, DistributedConfig, DistributedLearner, ModelUpdate, PeerInfo,
};
pub use pubsub::{PubSub, PubSubConfig, PubSubMessage};
