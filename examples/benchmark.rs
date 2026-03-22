//! Benchmarking for Hyper-Cognitive Kernel

use hyper_cognitive_kernel::continual::ContinualLearner;
use hyper_cognitive_kernel::neural::{ActivationFunction, NeuralNetwork};
use hyper_cognitive_kernel::{AgentConfig, CognitiveAgent};
use std::time::Instant;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║              HYPER-COGNITIVE KERNEL BENCHMARKS                   ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");
    println!();

    benchmark_agent_creation();
    benchmark_cognitive_cycle();
    benchmark_neural_network();
    benchmark_online_learning();
    benchmark_continual_learning();
    benchmark_memory_encoding();
    benchmark_throughput();

    println!();
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("                        BENCHMARKS COMPLETE                              ");
    println!("═══════════════════════════════════════════════════════════════════════");
}

fn benchmark_agent_creation() {
    println!("┌──────────────────────────────────────────────────────────────────────┐");
    println!("│ 1. AGENT CREATION TIME                                              │");
    println!("└──────────────────────────────────────────────────────────────────────┘");

    let config = AgentConfig::new()
        .with_learning_rate(0.01)
        .with_prediction_depth(3);

    let start = Instant::now();
    let agent = CognitiveAgent::new(1, config.clone());
    let agent2 = CognitiveAgent::new(2, config.clone());
    let agent3 = CognitiveAgent::new(3, config.clone());
    let duration = start.elapsed();

    let avg = duration.as_secs_f64() / 3.0 * 1000.0;
    println!(
        "    3 agents created in: {:.3} ms",
        duration.as_secs_f64() * 1000.0
    );
    println!("    Average per agent: {:.3} ms", avg);
    println!();
}

fn benchmark_cognitive_cycle() {
    println!("┌──────────────────────────────────────────────────────────────────────┐");
    println!("│ 2. COGNITIVE CYCLE LATENCY                                         │");
    println!("└──────────────────────────────────────────────────────────────────────┘");

    let config = AgentConfig::new()
        .with_learning_rate(0.01)
        .with_prediction_depth(3);

    let mut agent = CognitiveAgent::new(1, config);

    let sensors: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1).collect();

    // Warmup
    for _ in 0..100 {
        let _ = agent.cognitive_cycle(&sensors, 0.5);
    }

    // Benchmark
    let iterations = 10000;
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = agent.cognitive_cycle(&sensors, 0.5);
    }
    let duration = start.elapsed();

    let per_cycle = duration.as_secs_f64() / iterations as f64 * 1000.0;
    println!(
        "    {} cycles in: {:.3} ms",
        iterations,
        duration.as_secs_f64() * 1000.0
    );
    println!("    Per cycle: {:.6} ms", per_cycle);
    println!(
        "    Throughput: {:.0} cycles/sec",
        iterations as f64 / duration.as_secs_f64()
    );
    println!();
}

fn benchmark_neural_network() {
    println!("┌──────────────────────────────────────────────────────────────────────┐");
    println!("│ 3. NEURAL NETWORK INFERENCE                                        │");
    println!("└──────────────────────────────────────────────────────────────────────┘");

    let mut net = NeuralNetwork::new(4, 2);
    net.add_dense(64, ActivationFunction::ReLU);
    net.add_dense(32, ActivationFunction::ReLU);
    net.add_dense(2, ActivationFunction::Softmax);

    let input = vec![0.5, 0.3, 0.8, 0.2];

    // Warmup
    for _ in 0..1000 {
        let _ = net.predict(&input);
    }

    let iterations = 100000;
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = net.predict(&input);
    }
    let duration = start.elapsed();

    let per_inference = duration.as_secs_f64() / iterations as f64 * 1000.0;
    println!(
        "    {} inferences in: {:.3} ms",
        iterations,
        duration.as_secs_f64() * 1000.0
    );
    println!("    Per inference: {:.6} ms", per_inference);
    println!(
        "    Throughput: {:.0} inferences/sec",
        iterations as f64 / duration.as_secs_f64()
    );
    println!();
}

fn benchmark_online_learning() {
    println!("┌──────────────────────────────────────────────────────────────────────┐");
    println!("│ 4. ONLINE LEARNING (Single Sample)                                  │");
    println!("└──────────────────────────────────────────────────────────────────────┘");

    let mut net = NeuralNetwork::new(4, 2);
    net.add_dense(16, ActivationFunction::ReLU);
    net.add_dense(2, ActivationFunction::Softmax);

    let input = vec![0.5, 0.3, 0.8, 0.2];
    let target = vec![1.0, 0.0];

    // Warmup
    for _ in 0..100 {
        let _ = net.online_train(&input, &target);
    }

    let iterations = 10000;
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = net.online_train(&input, &target);
    }
    let duration = start.elapsed();

    let per_update = duration.as_secs_f64() / iterations as f64 * 1000.0;
    println!(
        "    {} updates in: {:.3} ms",
        iterations,
        duration.as_secs_f64() * 1000.0
    );
    println!("    Per update: {:.6} ms", per_update);
    println!(
        "    Throughput: {:.0} updates/sec",
        iterations as f64 / duration.as_secs_f64()
    );
    println!();
}

fn benchmark_continual_learning() {
    println!("┌──────────────────────────────────────────────────────────────────────┐");
    println!("│ 5. CONTINUAL LEARNING                                              │");
    println!("└──────────────────────────────────────────────────────────────────────┘");

    let mut learner = ContinualLearner::with_ewc(5000.0);

    let input = vec![0.1, 0.2, 0.3, 0.4];
    let target = vec![1.0, 0.0, 0.0];

    let iterations = 10000;
    let start = Instant::now();
    for i in 0..iterations {
        learner.add_replay_sample(input.clone(), target.clone(), i % 3);
    }
    let duration = start.elapsed();

    let per_sample = duration.as_secs_f64() / iterations as f64 * 1000.0;
    println!(
        "    {} samples in: {:.3} ms",
        iterations,
        duration.as_secs_f64() * 1000.0
    );
    println!("    Per sample: {:.6} ms", per_sample);
    println!(
        "    Throughput: {:.0} samples/sec",
        iterations as f64 / duration.as_secs_f64()
    );
    println!();
}

fn benchmark_memory_encoding() {
    println!("┌──────────────────────────────────────────────────────────────────────┐");
    println!("│ 6. MEMORY ENCODING                                                  │");
    println!("└──────────────────────────────────────────────────────────────────────┘");

    let config = AgentConfig::new()
        .with_learning_rate(0.01)
        .with_prediction_depth(3);

    let mut agent = CognitiveAgent::new(1, config);

    let input = vec![0.5; 8];

    let iterations = 10000;
    let start = Instant::now();
    for i in 0..iterations {
        let _ = agent.cognitive_cycle(&input, if i % 10 == 0 { 1.0 } else { 0.0 });
    }
    let duration = start.elapsed();

    let per_encode = duration.as_secs_f64() / iterations as f64 * 1000.0;
    println!(
        "    {} memories encoded in: {:.3} ms",
        iterations,
        duration.as_secs_f64() * 1000.0
    );
    println!("    Per memory: {:.6} ms", per_encode);
    println!();
}

fn benchmark_throughput() {
    println!("┌──────────────────────────────────────────────────────────────────────┐");
    println!("│ 7. SUSTAINED THROUGHPUT                                            │");
    println!("└──────────────────────────────────────────────────────────────────────┘");

    let config = AgentConfig::new()
        .with_learning_rate(0.01)
        .with_prediction_depth(3);

    let mut agent = CognitiveAgent::new(1, config);

    let sensors: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1).collect();

    let duration = Instant::now();
    let mut count = 0u64;

    while duration.elapsed().as_secs() < 3 {
        let _ = agent.cognitive_cycle(&sensors, 0.5);
        count += 1;
    }

    let elapsed = duration.elapsed().as_secs_f64();
    let throughput = count as f64 / elapsed;

    println!("    Samples processed in 3 seconds: {}", count);
    println!("    Throughput: {:.0} samples/sec", throughput);
    println!();
}
