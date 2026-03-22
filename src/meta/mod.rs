//! Meta-Learning Module
//!
//! Implements self-modifying learning systems that can adapt their own
//! learning parameters and strategies based on experience.
//!
//! ## Components
//!
//! - `MetaLearner`: Adjusts learning rates and strategies
//! - `SelfReflection`: Analyzes and reports on agent performance

pub mod learning_controller;
pub mod self_reflection;

pub use learning_controller::MetaLearner;
pub use self_reflection::{SelfReflection, SelfReflectionReport};
