//! Logger Module

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone)]
pub struct Logger {
    pub file_path: String,
    pub enabled: bool,
}

impl Logger {
    pub fn new(path: &str) -> Self {
        Self {
            file_path: path.to_string(),
            enabled: true,
        }
    }

    pub fn log(&self, message: &str) {
        if !self.enabled {
            return;
        }

        let timestamp = self.get_timestamp();
        let entry = format!("[{}] {}\n", timestamp, message);

        if let Ok(mut file) = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.file_path)
        {
            let _ = file.write_all(entry.as_bytes());
        }
    }

    pub fn log_info(&self, message: &str) {
        self.log(&format!("INFO: {}", message));
    }

    pub fn log_warn(&self, message: &str) {
        self.log(&format!("WARN: {}", message));
    }

    pub fn log_error(&self, message: &str) {
        self.log(&format!("ERROR: {}", message));
    }

    pub fn log_debug(&self, message: &str) {
        self.log(&format!("DEBUG: {}", message));
    }

    fn get_timestamp(&self) -> String {
        match SystemTime::now().duration_since(UNIX_EPOCH) {
            Ok(duration) => format!("{}", duration.as_secs()),
            Err(_) => "0".to_string(),
        }
    }

    pub fn clear(&self) {
        if let Ok(_) = File::create(&self.file_path) {
            // File cleared
        }
    }

    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
}

impl Default for Logger {
    fn default() -> Self {
        Self::new("agent.log")
    }
}
