//! Homeostasis Module
//!
//! Implements multi-drive homeostatic regulation for goal-directed behavior

pub mod allostasis;
pub mod drives;

pub use allostasis::Allostasis;
pub use drives::DriveSystem;
