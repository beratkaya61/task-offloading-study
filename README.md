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

- `configs/phase_5/` and `experiments/phase_5/`: sentetik RL, baseline ve ablation calismalari
- `configs/phase_6/` and `experiments/phase_6/`: synthetic-trace pipeline ve real-data recovery / real-composite trace akisi
- `configs/phase_7/` and `experiments/phase_7/`: oracle labeling, supervised warm-start ve staged training calismalari
- `configs/phase_8/` and `experiments/phase_8/`: graph-aware policy ve fusion karsilastirmalari
- `models/`: reusable checkpoints that are loaded again by later experiments
- `results/`: faz bazli ham CSV, gorsel ve ozet artefaktlari (`results/README.md`)

In short:
- `phase_*` folders tell you which stage of the study a config or experiment belongs to
- data regime is then made explicit in file names such as `synthetic_*`, `synthetic_trace_*`, or `real_composite_trace_*`

## How to Reproduce
1. Ensure dependencies are installed (e.g., `gymnasium`, `stable-baselines3`, `simpy`, `transformers`).
2. Run synthetic RL retraining: `python experiments/phase_5/run_synthetic_rl_retraining.py`
3. Run synthetic policy evaluation: `python experiments/phase_5/run_synthetic_policy_evaluation.py`
4. Run trace-driven PPO training: `python experiments/phase_6/train_synthetic_trace_ppo.py`
5. Experiment seed (`seed`), specific configurations, and models are deterministic via `src/utils/reproducibility.py`.
6. Inspect artefacts: workflow-specific CSV files phase bazli olarak `results/phase_*/metrics/` altina, gorseller ise `results/phase_*/figures/` altina yazilir. Faz 5 kanonik ozet raporu `v2_docs/phase_5/offloading_experiment_report.md` dosyasindadir.


