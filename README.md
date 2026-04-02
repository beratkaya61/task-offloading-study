# IoT Task Offloading Research Project

This project focuses on **"Semantic-Aware Task Offloading and Dynamic Resource Allocation in Next-Gen Edge Networks via LLM-Guided Deep Reinforcement Learning"**.

## Directory Structure

*   **`literature_review.md`**: Comprehensive review of 30+ papers, taxonomy comparison (2020-2025), and detailed thesis proposal with research gaps.
*   **`simulation_design.md`**: Technical specification of the simulation environment, including dynamic mathematical models (Shannon capacity, DVFS), battery models, fairness metrics, and dataset usage (Google Cluster, Didi Gaia).
*   **`roadmap.md`**: Visual project timeline (Gantt chart) and system architecture diagram.
*   **`concepts_and_keywords.md`**: Explanation of core concepts and list of search keywords used.

## Research Goal
To solve the "sparse reward" problem in DRL-based offloading by using Large Language Models (LLMs) for semantic task analysis and reward shaping, maximizing Quality of Experience (QoE) in dynamic IoT environments.

## Current Experiment Structure

- `configs/synthetic/` and `experiments/synthetic/`: simulated environment workflows
- `configs/trace/` and `experiments/trace/`: trace-driven workflows
- `models/`: reusable checkpoints that are loaded again by later experiments
- `results/`: logs, reports, and figures produced by training/evaluation runs

In short:
- `synthetic` means the model is trained or evaluated on the simulated environment
- `trace` means the model is trained or evaluated on trace-driven data

## How to Reproduce
1. Ensure dependencies are installed (e.g., `gymnasium`, `stable-baselines3`, `simpy`, `transformers`).
2. Run synthetic RL retraining: `python experiments/synthetic/train_rl_agents.py`
3. Run synthetic policy evaluation: `python experiments/synthetic/evaluate_policies.py`
4. Run trace-driven PPO training: `python experiments/trace/train_ppo.py`
5. Experiment seed (`seed`), specific configurations, and models are deterministic via `src/utils/reproducibility.py`.
6. Inspect logs: Detailed execution logs are written as workflow-specific CSV files under `results/raw/` and summarized in `results/tables/offloading_experiment_report.md`.
