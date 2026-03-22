//! Hyper-Cognitive Kernel - Demo

use hyper_cognitive_kernel::{
    continual::ContinualLearner,
    neural::{ActivationFunction, NeuralNetwork},
    nlp::{SentimentAnalyzer, Tokenizer},
    real_time::{RealTimeLearningController, Sample},
    rl_integration::{RLAgent, RLConfig, RLTransition},
    vision::{FeatureExtractor, Image},
    AgentConfig, CognitiveAgent,
};

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║          HYPER-COGNITIVE KERNEL v2.0 - ADVANCED EDITION            ║");
    println!("║                                                                      ║");
    println!("║  ✦ Real-time Online Learning (No Batch Training)                   ║");
    println!("║  ✦ Deep Neural Networks (LSTM, Attention, Reservoir)              ║");
    println!("║  ✦ Continual Learning (No Catastrophic Forgetting)                ║");
    println!("║  ✦ Distributed Multi-Agent Learning                               ║");
    println!("║  ✦ Vision Processing (CNN, Optical Flow)                         ║");
    println!("║  ✦ Natural Language Processing                                   ║");
    println!("║  ✦ Advanced RL + Active Inference                                ║");
    println!("║  ✦ Environment Integration (MQTT, WebSocket, API)               ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");
    println!();

    demo_neural_network();
    demo_realtime_learning();
    demo_continual_learning();
    demo_nlp();
    demo_vision();
    demo_rl_integration();
    demo_cognitive_agent();

    println!();
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║                    ALL DEMONSTRATIONS COMPLETE                      ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");
}

fn demo_neural_network() {
    println!("┌──────────────────────────────────────────────────────────────────────┐");
    println!("│ 1. NEURAL NETWORK - Online Learning                                │");
    println!("└──────────────────────────────────────────────────────────────────────┘");

    let mut net = NeuralNetwork::new(4, 2);
    net.add_dense(16, ActivationFunction::ReLU);
    net.add_dense(8, ActivationFunction::ReLU);
    net.add_dense(2, ActivationFunction::Softmax);

    println!("    Network: {}", net.summary());

    let input = vec![0.5, 0.3, 0.8, 0.2];
    let target = vec![1.0, 0.0];

    for i in 0..5 {
        let loss = net.online_train(&input, &target);
        println!(
            "    Step {}: Loss = {:.6}, Accuracy = {:.2}%",
            i + 1,
            loss,
            net.get_accuracy() * 100.0
        );
    }

    let stats = net.get_stats();
    println!(
        "    Stats: {} parameters, {} updates",
        stats.total_parameters, stats.total_updates
    );
    println!();
}

fn demo_realtime_learning() {
    println!("┌──────────────────────────────────────────────────────────────────────┐");
    println!("│ 2. REAL-TIME LEARNING CONTROLLER                                   │");
    println!("└──────────────────────────────────────────────────────────────────────┘");

    let mut controller = RealTimeLearningController::new();

    let input = vec![0.5, 0.3, 0.8, 0.2];
    let target = vec![1.0, 0.0];

    for i in 0..100 {
        let sample = Sample::new(input.clone(), target.clone());
        let decision = controller.process_sample(sample);

        let gradients = vec![0.1, 0.2, 0.3, 0.1, 0.2];
        let _clipped = controller.compute_adaptive_gradient(&gradients);
        let loss = 1.0 / (i as f64 + 1.0);
        let lr = controller.update_learning_rate(loss);

        if i % 25 == 0 {
            println!(
                "    Step {}: Decision = {:?}, Loss = {:.6}, LR = {:.6}",
                i, decision, loss, lr
            );
        }
    }

    let stats = controller.get_stats();
    println!(
        "    Samples: {}, Updates: {}, Throughput: {:.1} samples/sec",
        stats.samples_processed, stats.total_updates, stats.throughput_samples_per_sec
    );
    println!();
}

fn demo_continual_learning() {
    println!("┌──────────────────────────────────────────────────────────────────────┐");
    println!("│ 3. CONTINUAL LEARNING - No Catastrophic Forgetting!                  │");
    println!("└──────────────────────────────────────────────────────────────────────┘");

    let mut learner = ContinualLearner::with_ewc(5000.0);

    println!("    Method: Elastic Weight Consolidation (EWC)");
    println!("    Lambda: 5000.0");
    println!();

    println!("    Task 1: Learning pattern A...");
    for _ in 0..10 {
        learner.add_replay_sample(vec![0.1, 0.1, 0.1, 0.1], vec![1.0, 0.0, 0.0], 1);
    }

    println!("    Task 2: Learning pattern B (without forgetting A)...");
    for _ in 0..10 {
        learner.add_replay_sample(vec![0.9, 0.9, 0.9, 0.9], vec![0.0, 1.0, 0.0], 2);
    }

    let stats = learner.get_stats();
    println!("    Tasks learned: {}", stats.task_count);
    println!(
        "    Replay buffer: {}/{}",
        stats.replay_buffer_size, stats.replay_buffer_capacity
    );
    println!();
}

fn demo_nlp() {
    println!("┌──────────────────────────────────────────────────────────────────────┐");
    println!("│ 4. NATURAL LANGUAGE PROCESSING                                      │");
    println!("└──────────────────────────────────────────────────────────────────────┘");

    let mut tokenizer = Tokenizer::new(10000);
    let mut sentiment = SentimentAnalyzer::new();

    let texts = vec![
        "This is absolutely amazing! I love it so much!",
        "This is terrible, I hate it completely.",
        "It's okay, nothing special really.",
    ];

    for text in &texts {
        let ids = tokenizer.encode(text, 20);
        println!("    \"{}\"", text);
        println!("    Tokens: {} words", ids.len());

        let result = sentiment.analyze(text);
        println!(
            "    Sentiment: {:?} (confidence: {:.2}%)",
            result.sentiment,
            result.confidence * 100.0
        );
        println!();
    }
}

fn demo_vision() {
    println!("┌──────────────────────────────────────────────────────────────────────┐");
    println!("│ 5. VISION PROCESSING - Image Features                              │");
    println!("└──────────────────────────────────────────────────────────────────────┘");

    let extractor = FeatureExtractor::new();

    let image_data: Vec<f64> = (0..224 * 224 * 3)
        .map(|i| (i as f64 / 1000.0).sin())
        .collect();
    let image = Image::from_raw(image_data, 224, 224, 3);

    println!(
        "    Image: {}x{}x{}",
        image.width, image.height, image.channels
    );

    let features = extractor.extract_features(&image);
    println!("    Global features: {} dimensions", features.len());

    let spatial = extractor.extract_spatial_features(&image);
    println!("    Spatial features: {} dimensions", spatial.len());

    let histogram = extractor.extract_histogram(&image, 16);
    println!("    Histogram: {} bins", histogram.len());
    println!();
}

fn demo_rl_integration() {
    println!("┌──────────────────────────────────────────────────────────────────────┐");
    println!("│ 6. RL + ACTIVE INFERENCE - Intelligent Action Selection               │");
    println!("└──────────────────────────────────────────────────────────────────────┘");

    let config = RLConfig::default();
    let mut agent = RLAgent::new(config, 8, 4);

    let state = vec![0.5, 0.3, 0.8, 0.2, 0.6, 0.4, 0.7, 0.1];

    println!("    State: {:?} (8 dimensions)", &state[..4]);

    let action = agent.select_action(&state, false);
    println!("    Action: {:?} (4 values)", &action);

    let env_reward = 1.0;
    let prediction_error = 0.3;
    let combined_reward =
        agent.compute_active_inference_reward(env_reward, prediction_error, &state);

    println!(
        "    Env Reward: {:.2}, Prediction Error: {:.2}",
        env_reward, prediction_error
    );
    println!(
        "    Combined Reward: {:.2} (with Active Inference bonus)",
        combined_reward
    );

    let transition = RLTransition::new(
        state.clone(),
        action.clone(),
        combined_reward,
        state.clone(),
        false,
    );
    agent.store_transition(transition);

    println!();
}

fn demo_cognitive_agent() {
    println!("┌──────────────────────────────────────────────────────────────────────┐");
    println!("│ 7. FULL COGNITIVE AGENT - Complete Autonomous System                │");
    println!("└──────────────────────────────────────────────────────────────────────┘");

    let config = AgentConfig::new()
        .with_learning_rate(0.01)
        .with_prediction_depth(3)
        .with_surprise_threshold(0.5);

    let mut agent = CognitiveAgent::new(1, config);

    println!("    Agent ID: {}, Initial State: Active", agent.id);
    println!();

    let mut total_reward = 0.0;

    for step in 0..100 {
        let sensors = vec![
            (step as f64 / 10.0).sin(),
            (step as f64 / 5.0).cos(),
            ((step % 20) as f64 / 20.0),
            ((step % 15) as f64 / 15.0),
            0.5 + 0.3 * ((step / 7) as f64).sin(),
            0.5 + 0.3 * ((step / 11) as f64).cos(),
            ((step % 25) as f64 / 25.0),
            ((step % 30) as f64 / 30.0),
        ];

        let reward = if step % 10 == 0 { 1.0 } else { 0.0 };
        total_reward += reward;

        let _actions = agent.cognitive_cycle(&sensors, reward);

        if step % 25 == 0 {
            let report = agent.self_reflect();
            println!(
                "    Step {:3}: Free Energy = {:.4}, Accuracy = {:.1}%, Reward = {:.2}",
                step,
                report.free_energy,
                report.prediction_accuracy * 100.0,
                total_reward
            );
        }
    }

    let report = agent.self_reflect();
    println!();
    println!("    Final Report:");
    println!("    ├─ Age (Cycles): {}", report.age);
    println!("    ├─ Free Energy: {:.6}", report.free_energy);
    println!(
        "    ├─ Prediction Accuracy: {:.1}%",
        report.prediction_accuracy * 100.0
    );
    println!("    ├─ Total Reward: {:.2}", total_reward);
    println!("    └─ Memory Usage: {:.1}%", report.memory_usage * 100.0);
}
