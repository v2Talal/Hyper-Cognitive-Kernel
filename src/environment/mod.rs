//! Environment Interface Module
//!
//! Provides real-time communication with external environments through:
//! - MQTT for IoT and real-time sensor data
//! - WebSocket for browser and real-time web applications
//! - REST API for HTTP-based communication
//! - gRPC for high-performance inter-service communication

pub mod http_api;
pub mod message_bus;
pub mod mqtt_client;
pub mod websocket_client;

pub use http_api::HTTPAPI;
pub use message_bus::MessageBus;
pub use mqtt_client::MQTTClient;
pub use websocket_client::WebSocketClient;

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentObservation {
    pub timestamp: u64,
    pub sensors: Vec<f64>,
    pub metadata: ObservationMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservationMetadata {
    pub source: String,
    pub quality: f64,
    pub latency_ms: u64,
    pub sequence_num: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentAction {
    pub action_id: u64,
    pub values: Vec<f64>,
    pub metadata: ActionMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionMetadata {
    pub timestamp: u64,
    pub priority: Priority,
    pub retry_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum Priority {
    Critical,
    High,
    Normal,
    Low,
}

impl Default for Priority {
    fn default() -> Self {
        Priority::Normal
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentConfig {
    pub connection_type: ConnectionType,
    pub host: String,
    pub port: u16,
    pub use_tls: bool,
    pub auth_token: Option<String>,
    pub reconnect_interval_ms: u64,
    pub max_reconnect_attempts: u32,
    pub timeout_ms: u64,
}

impl Default for EnvironmentConfig {
    fn default() -> Self {
        Self {
            connection_type: ConnectionType::WebSocket,
            host: "localhost".to_string(),
            port: 8080,
            use_tls: false,
            auth_token: None,
            reconnect_interval_ms: 1000,
            max_reconnect_attempts: 10,
            timeout_ms: 5000,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConnectionType {
    MQTT,
    WebSocket,
    HTTP,
    TCP,
    UDP,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionState {
    pub connected: bool,
    pub last_connected: Option<u64>,
    pub reconnect_count: u32,
    pub message_count: u64,
    pub error_count: u32,
}

impl Default for ConnectionState {
    fn default() -> Self {
        Self {
            connected: false,
            last_connected: None,
            reconnect_count: 0,
            message_count: 0,
            error_count: 0,
        }
    }
}

pub struct EnvironmentInterface {
    config: EnvironmentConfig,
    state: Arc<RwLock<ConnectionState>>,
    message_handlers: Vec<Box<dyn MessageHandler>>,
}

pub trait MessageHandler: Send + Sync {
    fn handle(&mut self, observation: EnvironmentObservation) -> Option<Vec<f64>>;
    fn handle_action_result(&mut self, result: ActionResult);
}

pub struct ActionResult {
    pub success: bool,
    pub reward: f64,
    pub done: bool,
    pub info: std::collections::HashMap<String, String>,
}

impl EnvironmentInterface {
    pub fn new(config: EnvironmentConfig) -> Self {
        Self {
            config,
            state: Arc::new(RwLock::new(ConnectionState::default())),
            message_handlers: Vec::new(),
        }
    }

    pub fn connect(&mut self) -> Result<(), String> {
        let mut state = self.state.write();
        state.connected = true;
        state.last_connected = Some(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
        );
        Ok(())
    }

    pub fn disconnect(&mut self) {
        let mut state = self.state.write();
        state.connected = false;
    }

    pub fn is_connected(&self) -> bool {
        self.state.read().connected
    }

    pub fn send_action(&self, action: &AgentAction) -> Result<(), String> {
        if !self.is_connected() {
            return Err("Not connected".to_string());
        }

        let mut state = self.state.write();
        state.message_count += 1;

        Ok(())
    }

    pub fn register_handler(&mut self, handler: Box<dyn MessageHandler>) {
        self.message_handlers.push(handler);
    }

    pub fn get_state(&self) -> ConnectionState {
        self.state.read().clone()
    }
}

impl Clone for EnvironmentInterface {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            state: Arc::clone(&self.state),
            message_handlers: Vec::new(),
        }
    }
}
