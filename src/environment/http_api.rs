//! HTTP API Client for RESTful Communication

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HTTPConfig {
    pub base_url: String,
    pub timeout_ms: u64,
    pub headers: HashMap<String, String>,
    pub auth_type: AuthType,
}

impl Default for HTTPConfig {
    fn default() -> Self {
        Self {
            base_url: "http://localhost:8080/api".to_string(),
            timeout_ms: 5000,
            headers: HashMap::new(),
            auth_type: AuthType::None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuthType {
    None,
    Bearer(String),
    Basic(String, String),
    ApiKey(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HTTPRequest {
    pub method: HTTPMethod,
    pub path: String,
    pub headers: HashMap<String, String>,
    pub body: Option<Vec<u8>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HTTPMethod {
    GET,
    POST,
    PUT,
    DELETE,
    PATCH,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HTTPResponse {
    pub status_code: u16,
    pub headers: HashMap<String, String>,
    pub body: Vec<u8>,
}

pub struct HTTPAPI {
    config: HTTPConfig,
    client: HTTPClient,
}

enum HTTPClient {
    #[cfg(feature = "reqwest")]
    Reqwest(reqwest::Client),
    Mock(MockClient),
}

struct MockClient;

impl MockClient {
    fn new() -> Self {
        MockClient
    }

    fn request(&self, _request: HTTPRequest) -> Result<HTTPResponse, String> {
        Ok(HTTPResponse {
            status_code: 200,
            headers: HashMap::new(),
            body: vec![],
        })
    }
}

impl HTTPAPI {
    pub fn new(config: HTTPConfig) -> Self {
        let client = HTTPClient::Mock(MockClient::new());
        Self { config, client }
    }

    pub fn get(&self, path: &str) -> Result<HTTPResponse, String> {
        let request = HTTPRequest {
            method: HTTPMethod::GET,
            path: path.to_string(),
            headers: self.config.headers.clone(),
            body: None,
        };
        self.request(request)
    }

    pub fn post<T: Serialize>(&self, path: &str, body: &T) -> Result<HTTPResponse, String> {
        let json = serde_json::to_vec(body).map_err(|e| format!("Serialization error: {}", e))?;

        let mut headers = self.config.headers.clone();
        headers.insert("Content-Type".to_string(), "application/json".to_string());

        let request = HTTPRequest {
            method: HTTPMethod::POST,
            path: path.to_string(),
            headers,
            body: Some(json),
        };

        self.request(request)
    }

    pub fn put<T: Serialize>(&self, path: &str, body: &T) -> Result<HTTPResponse, String> {
        let json = serde_json::to_vec(body).map_err(|e| format!("Serialization error: {}", e))?;

        let mut headers = self.config.headers.clone();
        headers.insert("Content-Type".to_string(), "application/json".to_string());

        let request = HTTPRequest {
            method: HTTPMethod::PUT,
            path: path.to_string(),
            headers,
            body: Some(json),
        };

        self.request(request)
    }

    pub fn delete(&self, path: &str) -> Result<HTTPResponse, String> {
        let request = HTTPRequest {
            method: HTTPMethod::DELETE,
            path: path.to_string(),
            headers: self.config.headers.clone(),
            body: None,
        };
        self.request(request)
    }

    fn request(&self, request: HTTPRequest) -> Result<HTTPResponse, String> {
        match &self.client {
            #[cfg(feature = "reqwest")]
            HTTPClient::Reqwest(client) => {
                let client = reqwest::Client::new();
                let url = format!("{}{}", self.config.base_url, request.path);

                let mut req = match request.method {
                    HTTPMethod::GET => client.get(&url),
                    HTTPMethod::POST => client.post(&url),
                    HTTPMethod::PUT => client.put(&url),
                    HTTPMethod::DELETE => client.delete(&url),
                    HTTPMethod::PATCH => client.patch(&url),
                };

                for (key, value) in &request.headers {
                    req = req.header(key, value);
                }

                if let Some(body) = request.body {
                    req = req.body(body);
                }

                let response = req.send().map_err(|e| format!("Request failed: {}", e))?;

                let status = response.status().as_u16();
                let body = response
                    .bytes()
                    .map_err(|e| format!("Failed to read body: {}", e))?
                    .to_vec();

                Ok(HTTPResponse {
                    status_code: status,
                    headers: HashMap::new(),
                    body,
                })
            }
            HTTPClient::Mock(client) => client.request(request),
        }
    }

    pub fn set_auth(&mut self, auth: AuthType) {
        self.config.auth_type = auth;

        match &self.config.auth_type {
            AuthType::Bearer(token) => {
                self.config
                    .headers
                    .insert("Authorization".to_string(), format!("Bearer {}", token));
            }
            AuthType::Basic(username, password) => {
                let credentials = base64::encode(&format!("{}:{}", username, password));
                self.config.headers.insert(
                    "Authorization".to_string(),
                    format!("Basic {}", credentials),
                );
            }
            AuthType::ApiKey(key) => {
                self.config
                    .headers
                    .insert("X-API-Key".to_string(), key.clone());
            }
            AuthType::None => {}
        }
    }

    pub fn get_config(&self) -> &HTTPConfig {
        &self.config
    }
}

mod base64 {
    pub fn encode(input: &str) -> String {
        let alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let bytes = input.as_bytes();
        let mut result = String::new();

        for chunk in bytes.chunks(3) {
            let mut n: u32 = 0;
            for (i, &byte) in chunk.iter().enumerate() {
                n |= (byte as u32) << (16 - i * 8);
            }

            result.push(
                alphabet
                    .chars()
                    .nth(((n >> 18) & 0x3F) as usize)
                    .unwrap_or('='),
            );
            result.push(
                alphabet
                    .chars()
                    .nth(((n >> 12) & 0x3F) as usize)
                    .unwrap_or('='),
            );

            if chunk.len() > 1 {
                result.push(
                    alphabet
                        .chars()
                        .nth(((n >> 6) & 0x3F) as usize)
                        .unwrap_or('='),
                );
            } else {
                result.push('=');
            }

            if chunk.len() > 2 {
                result.push(alphabet.chars().nth((n & 0x3F) as usize).unwrap_or('='));
            } else {
                result.push('=');
            }
        }

        result
    }
}
