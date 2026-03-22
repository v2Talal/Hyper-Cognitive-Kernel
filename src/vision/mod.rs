//! Visual Processing Module - Image and Video Processing for Cognitive Agents

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageConfig {
    pub width: usize,
    pub height: usize,
    pub channels: usize,
    pub normalize: bool,
    pub mean: Vec<f64>,
    pub std: Vec<f64>,
}

impl Default for ImageConfig {
    fn default() -> Self {
        Self {
            width: 224,
            height: 224,
            channels: 3,
            normalize: true,
            mean: vec![0.485, 0.456, 0.406],
            std: vec![0.229, 0.224, 0.225],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Image {
    pub data: Vec<f64>,
    pub width: usize,
    pub height: usize,
    pub channels: usize,
    pub timestamp: u64,
}

impl Image {
    pub fn new(width: usize, height: usize, channels: usize) -> Self {
        Self {
            data: vec![0.0; width * height * channels],
            width,
            height,
            channels,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
        }
    }

    pub fn from_raw(data: Vec<f64>, width: usize, height: usize, channels: usize) -> Self {
        Self {
            data,
            width,
            height,
            channels,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
        }
    }

    pub fn resize(&self, new_width: usize, new_height: usize) -> Self {
        let mut resized = Self::new(new_width, new_height, self.channels);

        let x_ratio = self.width as f64 / new_width as f64;
        let y_ratio = self.height as f64 / new_height as f64;

        for y in 0..new_height {
            for x in 0..new_width {
                let src_x = (x as f64 * x_ratio) as usize;
                let src_y = (y as f64 * y_ratio) as usize;

                let src_idx = (src_y * self.width + src_x) * self.channels;
                let dst_idx = (y * new_width + x) * self.channels;

                for c in 0..self.channels {
                    if src_idx + c < self.data.len() && dst_idx + c < resized.data.len() {
                        resized.data[dst_idx + c] = self.data[src_idx + c];
                    }
                }
            }
        }

        resized
    }

    pub fn normalize(&self, mean: &[f64], std: &[f64]) -> Vec<f64> {
        self.data
            .chunks(self.channels)
            .flat_map(|pixel| {
                pixel
                    .iter()
                    .zip(mean.iter())
                    .zip(std.iter())
                    .map(|((&p, &m), &s)| (p - m) / s)
                    .collect::<Vec<f64>>()
            })
            .collect()
    }

    pub fn to_grayscale(&self) -> Self {
        if self.channels < 3 {
            return self.clone();
        }

        let mut gray = Self::new(self.width, self.height, 1);

        for y in 0..self.height {
            for x in 0..self.width {
                let idx = (y * self.width + x) * self.channels;
                let r = self.data[idx];
                let g = self.data[idx + 1];
                let b = self.data[idx + 2];

                let gray_val = 0.299 * r + 0.587 * g + 0.114 * b;
                gray.data[y * self.width + x] = gray_val;
            }
        }

        gray
    }

    pub fn apply_gaussian_blur(&self, kernel_size: usize, sigma: f64) -> Self {
        let mut blurred = self.clone();
        let kernel = Self::create_gaussian_kernel(kernel_size, sigma);
        let half = kernel_size / 2;

        for y in half..self.height - half {
            for x in half..self.width - half {
                let mut sum = vec![0.0; self.channels];

                for ky in 0..kernel_size {
                    for kx in 0..kernel_size {
                        let px = x + kx - half;
                        let py = y + ky - half;
                        let pidx = (py * self.width + px) * self.channels;
                        let kidx = ky * kernel_size + kx;

                        for c in 0..self.channels {
                            sum[c] += self.data[pidx + c] * kernel[kidx];
                        }
                    }
                }

                let didx = (y * self.width + x) * self.channels;
                for c in 0..self.channels.min(sum.len()) {
                    blurred.data[didx + c] = sum[c];
                }
            }
        }

        blurred
    }

    fn create_gaussian_kernel(size: usize, sigma: f64) -> Vec<f64> {
        let mut kernel = Vec::with_capacity(size * size);
        let half = size as f64 / 2.0;
        let mut sum = 0.0;

        for y in 0..size {
            for x in 0..size {
                let dx = x as f64 - half + 0.5;
                let dy = y as f64 - half + 0.5;
                let value = (-(dx * dx + dy * dy) / (2.0 * sigma * sigma)).exp();
                kernel.push(value);
                sum += value;
            }
        }

        for v in &mut kernel {
            *v /= sum;
        }

        kernel
    }

    pub fn detect_edges(&self, threshold: f64) -> Self {
        let gray = self.to_grayscale();
        let mut edges = Self::new(gray.width, gray.height, 1);

        let sobel_x = [-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0];
        let sobel_y = [-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0];

        for y in 1..gray.height - 1 {
            for x in 1..gray.width - 1 {
                let mut gx = 0.0;
                let mut gy = 0.0;

                for ky in 0..3 {
                    for kx in 0..3 {
                        let px = x + kx - 1;
                        let py = y + ky - 1;
                        let idx = py * gray.width + px;
                        let kidx = ky * 3 + kx;

                        gx += gray.data[idx] * sobel_x[kidx];
                        gy += gray.data[idx] * sobel_y[kidx];
                    }
                }

                let magnitude = (gx * gx + gy * gy).sqrt();
                let idx = y * gray.width + x;
                edges.data[idx] = if magnitude > threshold { 1.0 } else { 0.0 };
            }
        }

        edges
    }
}

pub struct VideoStream {
    frames: VecDeque<Image>,
    max_frames: usize,
    fps: f64,
}

impl VideoStream {
    pub fn new(max_frames: usize) -> Self {
        Self {
            frames: VecDeque::with_capacity(max_frames),
            max_frames,
            fps: 30.0,
        }
    }

    pub fn push_frame(&mut self, image: Image) {
        if self.frames.len() >= self.max_frames {
            self.frames.pop_front();
        }
        self.frames.push_back(image);
    }

    pub fn get_frame(&self, idx: usize) -> Option<&Image> {
        self.frames.get(idx)
    }

    pub fn get_latest(&self) -> Option<&Image> {
        self.frames.back()
    }

    pub fn get_temporal_difference(&self) -> Option<Vec<f64>> {
        if self.frames.len() < 2 {
            return None;
        }

        let current = self.frames.back()?;
        let previous = self.frames.get(self.frames.len() - 2)?;

        let diff: Vec<f64> = current
            .data
            .iter()
            .zip(previous.data.iter())
            .map(|(c, p)| (c - p).abs())
            .collect();

        Some(diff)
    }

    pub fn optical_flow(&self) -> Option<Vec<(f64, f64)>> {
        if self.frames.len() < 2 {
            return None;
        }

        let mut flows = Vec::new();

        for i in 0..self.frames.len() - 1 {
            let frame1 = &self.frames[i];
            let frame2 = &self.frames[i + 1];

            for y in (0..frame1.height - 1).step_by(8) {
                for x in (0..frame1.width - 1).step_by(8) {
                    let idx1 = y * frame1.width + x;
                    let idx2 = (y + 1) * frame2.width + (x + 1);

                    if idx1 < frame1.data.len() && idx2 < frame2.data.len() {
                        let dx = frame2.data[idx2] - frame1.data[idx1];
                        let dy = frame2.data[idx2] - frame1.data[idx1];
                        flows.push((dx, dy));
                    }
                }
            }
        }

        Some(flows)
    }

    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }
}

pub struct FeatureExtractor {
    pub config: ImageConfig,
}

impl FeatureExtractor {
    pub fn new() -> Self {
        Self {
            config: ImageConfig::default(),
        }
    }

    pub fn extract_features(&self, image: &Image) -> Vec<f64> {
        let resized = image.resize(self.config.width, self.config.height);

        let features = if self.config.normalize {
            resized.normalize(&self.config.mean, &self.config.std)
        } else {
            resized.data.clone()
        };

        let pooled = self.global_average_pool(&features, self.config.channels);

        pooled
    }

    fn global_average_pool(&self, data: &[f64], channels: usize) -> Vec<f64> {
        let spatial_size = data.len() / channels;
        let mut pooled = vec![0.0; channels];

        for c in 0..channels {
            let channel_data = &data[c * spatial_size..(c + 1) * spatial_size];
            pooled[c] = channel_data.iter().sum::<f64>() / spatial_size as f64;
        }

        pooled
    }

    pub fn extract_spatial_features(&self, image: &Image) -> Vec<f64> {
        let gray = image.to_grayscale();

        let mut features = Vec::new();

        let regions = 4;
        let region_h = gray.height / regions;
        let region_w = gray.width / regions;

        for r in 0..regions {
            for c in 0..regions {
                let mut region_sum = 0.0;
                let mut count = 0;

                for y in r * region_h..(r + 1) * region_h {
                    for x in c * region_w..(c + 1) * region_w {
                        let idx = y * gray.width + x;
                        if idx < gray.data.len() {
                            region_sum += gray.data[idx];
                            count += 1;
                        }
                    }
                }

                features.push(region_sum / count as f64);
            }
        }

        features
    }

    pub fn extract_histogram(&self, image: &Image, bins: usize) -> Vec<f64> {
        let gray = image.to_grayscale();
        let mut histogram = vec![0.0; bins];
        let bin_size = 256.0 / bins as f64;

        for &pixel in &gray.data {
            let bin = ((pixel as f64 * 255.0) / bin_size) as usize;
            let bin = bin.min(bins - 1);
            histogram[bin] += 1.0;
        }

        let sum: f64 = histogram.iter().sum();
        if sum > 0.0 {
            for h in &mut histogram {
                *h /= sum;
            }
        }

        histogram
    }
}

impl Default for FeatureExtractor {
    fn default() -> Self {
        Self::new()
    }
}
