# IoT Task Offloading Research Project

This project focuses on **"Semantic-Aware Task Offloading and Dynamic Resource Allocation in Next-Gen Edge Networks via LLM-Guided Deep Reinforcement Learning"**.

## Directory Structure

*   **`literature_review.md`**: Comprehensive review of 30+ papers, taxonomy comparison (2020-2025), and detailed thesis proposal with research gaps.
*   **`simulation_design.md`**: Technical specification of the simulation environment, including dynamic mathematical models (Shannon capacity, DVFS), battery models, fairness metrics, and dataset usage (Google Cluster, Didi Gaia).
*   **`roadmap.md`**: Visual project timeline (Gantt chart) and system architecture diagram.
*   **`concepts_and_keywords.md`**: Explanation of core concepts and list of search keywords used.

## Research Goal
To solve the "sparse reward" problem in DRL-based offloading by using Large Language Models (LLMs) for semantic task analysis and reward shaping, maximizing Quality of Experience (QoE) in dynamic IoT environments.

## How to Reproduce
1. Ensure dependencies are installed (e.g., `gymnasium`, `stable-baselines3`, `simpy`, `transformers`).
2. Run standard training: `python src/train_agent.py`
3. Experiment seed (`seed`), specific configurations, and models are deterministic via `src/utils/reproducibility.py`.
4. Inspect logs: Detailed execution logs and results are generated in JSON and a master CSV table directly inside `results/raw/master_experiments.csv`. Evaluater scripts like `src/evaluation.py` can be used to compare experiments out-of-the-box.
