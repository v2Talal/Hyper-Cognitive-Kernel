//! Cognition Module
//!
//! This module contains the core cognitive systems:
//! - `PredictiveCoder`: Hierarchical predictive coding network
//! - `WorldModel`: Internal simulation of the environment
//! - `AttentionSystem`: Selective focus mechanism

pub mod attention;
pub mod predictive_coding;
pub mod world_model;

pub use attention::AttentionSystem;
pub use predictive_coding::PredictiveCoder;
pub use world_model::WorldModel;
