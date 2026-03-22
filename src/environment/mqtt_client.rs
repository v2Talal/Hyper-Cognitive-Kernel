//! MQTT Client for IoT and Real-time Sensor Data

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MQTTConfig {
    pub broker_host: String,
    pub broker_port: u16,
    pub client_id: String,
    pub username: Option<String>,
    pub password: Option<String>,
    pub topics: Vec<TopicSubscription>,
    pub qos: QoS,
    pub keep_alive_secs: u16,
}

impl Default for MQTTConfig {
    fn default() -> Self {
        Self {
            broker_host: "localhost".to_string(),
            broker_port: 1883,
            client_id: "hck_agent".to_string(),
            username: None,
            password: None,
            topics: Vec::new(),
            qos: QoS::AtLeastOnce,
            keep_alive_secs: 60,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopicSubscription {
    pub topic: String,
    pub qos: QoS,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum QoS {
    AtMostOnce,
    AtLeastOnce,
    ExactlyOnce,
}

impl Default for QoS {
    fn default() -> Self {
        QoS::AtLeastOnce
    }
}

pub struct MQTTClient {
    config: MQTTConfig,
    connected: bool,
    subscriptions: Vec<String>,
    message_queue: Vec<MQTTMessage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MQTTMessage {
    pub topic: String,
    pub payload: Vec<u8>,
    pub qos: QoS,
    pub retain: bool,
    pub timestamp: u64,
}

impl MQTTClient {
    pub fn new(config: MQTTConfig) -> Self {
        Self {
            config,
            connected: false,
            subscriptions: Vec::new(),
            message_queue: Vec::new(),
        }
    }

    pub fn connect(&mut self) -> Result<(), String> {
        self.connected = true;
        Ok(())
    }

    pub fn disconnect(&mut self) {
        self.connected = false;
        self.subscriptions.clear();
    }

    pub fn is_connected(&self) -> bool {
        self.connected
    }

    pub fn subscribe(&mut self, topic: &str) -> Result<(), String> {
        if !self.connected {
            return Err("Not connected".to_string());
        }

        self.subscriptions.push(topic.to_string());
        Ok(())
    }

    pub fn unsubscribe(&mut self, topic: &str) -> Result<(), String> {
        self.subscriptions.retain(|t| t != topic);
        Ok(())
    }

    pub fn publish(&mut self, topic: &str, payload: &[u8]) -> Result<(), String> {
        if !self.connected {
            return Err("Not connected".to_string());
        }

        let message = MQTTMessage {
            topic: topic.to_string(),
            payload: payload.to_vec(),
            qos: self.config.qos,
            retain: false,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
        };

        self.message_queue.push(message);
        Ok(())
    }

    pub fn poll_messages(&mut self) -> Vec<MQTTMessage> {
        let messages = self.message_queue.clone();
        self.message_queue.clear();
        messages
    }

    pub fn get_subscriptions(&self) -> &[String] {
        &self.subscriptions
    }

    pub fn get_config(&self) -> &MQTTConfig {
        &self.config
    }
}
