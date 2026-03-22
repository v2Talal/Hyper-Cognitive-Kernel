//! WebSocket Client for Real-time Web Communication

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebSocketConfig {
    pub url: String,
    pub protocols: Vec<String>,
    pub ping_interval_ms: u64,
    pub pong_timeout_ms: u64,
    pub max_message_size: usize,
    pub binary_mode: bool,
}

impl Default for WebSocketConfig {
    fn default() -> Self {
        Self {
            url: "ws://localhost:8080/ws".to_string(),
            protocols: vec!["hck-v1".to_string()],
            ping_interval_ms: 30000,
            pong_timeout_ms: 10000,
            max_message_size: 1024 * 1024,
            binary_mode: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebSocketMessage {
    pub msg_type: MessageType,
    pub data: Vec<u8>,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MessageType {
    Text,
    Binary,
    Ping,
    Pong,
    Close,
}

pub struct WebSocketClient {
    config: WebSocketConfig,
    connected: bool,
    send_queue: VecDeque<WebSocketMessage>,
    receive_queue: VecDeque<WebSocketMessage>,
    last_ping: u64,
}

impl WebSocketClient {
    pub fn new(config: WebSocketConfig) -> Self {
        Self {
            config,
            connected: false,
            send_queue: VecDeque::new(),
            receive_queue: VecDeque::new(),
            last_ping: 0,
        }
    }

    pub fn connect(&mut self) -> Result<(), String> {
        self.connected = true;
        self.last_ping = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        Ok(())
    }

    pub fn disconnect(&mut self) {
        self.connected = false;
        self.send_queue.clear();
    }

    pub fn is_connected(&self) -> bool {
        self.connected
    }

    pub fn send(&mut self, msg_type: MessageType, data: Vec<u8>) -> Result<(), String> {
        if !self.connected {
            return Err("Not connected".to_string());
        }

        let message = WebSocketMessage {
            msg_type,
            data,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
        };

        self.send_queue.push_back(message);
        Ok(())
    }

    pub fn receive(&mut self) -> Option<WebSocketMessage> {
        self.receive_queue.pop_front()
    }

    pub fn receive_all(&mut self) -> Vec<WebSocketMessage> {
        let messages: Vec<_> = self.receive_queue.drain(..).collect();
        messages
    }

    pub fn send_text(&mut self, text: &str) -> Result<(), String> {
        self.send(MessageType::Text, text.as_bytes().to_vec())
    }

    pub fn send_binary(&mut self, data: &[u8]) -> Result<(), String> {
        self.send(MessageType::Binary, data.to_vec())
    }

    pub fn ping(&mut self) -> Result<(), String> {
        self.send(MessageType::Ping, vec![])
    }

    pub fn queued_messages(&self) -> usize {
        self.send_queue.len() + self.receive_queue.len()
    }

    pub fn get_config(&self) -> &WebSocketConfig {
        &self.config
    }

    pub fn set_binary_mode(&mut self, binary: bool) {
        self.config.binary_mode = binary;
    }
}
