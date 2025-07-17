# Local Deployment of Transformer-based Code Assistants

This project benchmarks lightweight transformer-based language models for offline Python code generation, focusing on practical trade-offs between speed, accuracy, and resource efficiency.

## Overview

Modern AI coding assistants like GitHub Copilot offer powerful support but raise concerns around privacy, latency, and cloud dependency. This study investigates offline deployment of two transformer models—**CodeT5** and **StarCoder**—integrated into a local VS Code environment to simulate real-world usage.

## Paper

**Title:** Local Deployment of Transformer-based Code Assistants: Balancing Speed, Accuracy, and Efficiency  
**Author:** Jaiveer Bassi  
**Preprint:** [TechRxiv link coming soon]  
**Institution:** Grand Canyon University

## Models Evaluated

- **CodeT5-small (60M parameters)**
- **CodeT5-base (220M parameters)**
- **StarCoder (15B parameters)**

## Benchmark Setup

- **Task:** Python code completion on the HumanEval benchmark
- **Metrics:**
  - Pass@1 and Pass@5 for functional correctness
  - Inference speed (tokens/sec)
- **Environment:** NVIDIA RTX 3090 GPU, Hugging Face Transformers, PyTorch, VS Code extension

## Key Results

| Model               | Params | Pass@1 | Pass@5 | Speed (tokens/s) |
|--------------------|--------|--------|--------|------------------|
| CodeT5-Small        | 60M    | 15.0%  | 28.5%  | 450              |
| CodeT5-Base         | 220M   | 23.0%  | 42.0%  | 300              |
| StarCoder (default) | 15B    | 33.6%  | 56.2%  | 55               |
| StarCoder (prompted)| 15B    | 40.8%  | 65.0%  | 50               |

## Highlights

- **StarCoder** provides the highest accuracy but requires significant GPU resources.
- **CodeT5-base** offers a good balance between accuracy and latency for local deployment.
- Offline coding assistants are feasible and beneficial for privacy-sensitive development.

## Implementation

- Built using Hugging Face Transformers
- Integrated into a VS Code extension via Language Server Protocol
- Models served through a local Python backend for in-editor code completions

## Future Work

- Extend benchmarks to other models (CodeLlama, QwenCoder)
- Add support for additional tasks (e.g., bug fixing, summarization)
- Explore compression techniques and real-time streaming in the IDE

## License

This project is released under an open license for academic and non-commercial use. See `LICENSE` for details.
