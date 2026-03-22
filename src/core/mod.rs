//! Core module containing the main agent and configuration
//!
//! This module provides the foundational components for the cognitive agent:
//! - `AgentConfig`: Configuration parameters for agent behavior
//! - `CognitiveAgent`: The main agent that orchestrates all cognitive processes

pub mod agent;
pub mod config;

pub use agent::CognitiveAgent;
pub use config::AgentConfig;
