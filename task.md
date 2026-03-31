# AgentVNE-Inspired Task Offloading Upgrade Plan

Based on `TODO_ANTIGRAVITY_TASK_OFFLOADING_UPGRADE.md` and `AgentVNE`.

## Faz 1 - Reproducibility ve Kod Temizligi
- [x] 1.1 `src/baselines.py`, `src/evaluation.py`, `src/metrics.py`, `src/config.py`, `src/trace_loader.py`, `src/semantic_prior.py`, `src/pretrain_policy.py`, `src/utils/reproducibility.py`
- [x] 1.2 `configs/` and `results/` structure
- [x] 1.3 Seed helper for `numpy`, `random`, `torch`, `gymnasium`
- [x] 1.4 JSON/CSV logging infrastructure
- [x] 1.5 Phase 1 test and commit (user side)

## Faz 2 - Train Environment ve Gercek Simulasyon Hizalamasi
- [x] 2.1 Shared `src/state_builder.py`
- [x] 2.2 Simulator-backed reset in `rl_env.py`
- [x] 2.3 Reward components moved to `src/reward.py`
- [x] 2.4 Multi-step episode design
- [x] 2.5 Phase 2 test and commit (user side)

## Faz 3 - LLM Entegrasyonunu Gercek Katkiya Donustur
- [x] 3.1 6D action prior via `semantic_prior.py`
- [x] 3.2 Confidence integration
- [x] 3.3 Structured JSON output and logging
- [x] 3.4 Phase 3 test and commit (user side)

## Faz 4 - Baseline Ailesini Genislet
- [x] 4.1 LocalOnly, EdgeOnly, Random, GreedyLatency
- [x] 4.2 Genetic Algorithm baseline
- [x] 4.3 PPO v2, DQN, A2C scaffolding
- [x] 4.4 Common evaluation
- [x] 4.5 Phase 4 test/report

## Faz 5 - Sistematik Ablation Study
- [x] 5.1 Ablation config (`configs/ablation.yaml`)
- [x] 5.2 Ablation logs (90 run, Phase_5_Report.md)
- [x] 5.3 Phase 5 test/report

## Faz 6 - Gercek Veri / Trace-Driven Deney Paketi
- [x] 6.1 `trace_loader.py` exists in `src/core/trace_loader.py`
- [x] 6.2 Trace to task mapping
- [ ] 6.3 Domain shift (synthetic vs trace)
- [x] 6.4 Phase 6 test/report

## Faz 7 - Two-Stage Training (AgentVNE Concept)
- [ ] 7.1 Oracle / heuristic labels
- [ ] 7.2 Imitation / supervised pretraining
- [ ] 7.3 Fine-tune PPO vs scratch comparison
- [ ] 7.4 Phase 7 test and commit

## Faz 8 - Graph-Aware Policy Upgrade
- [ ] 8.1 Graph state node and edge features
- [ ] 8.2 GNN policy implementation
- [ ] 8.3 Semantic prior fusion
- [ ] 8.4 Phase 8 test and commit

## Faz 9 - Advanced Metrics and GUI
- [ ] 9.1 Jitter/Fairness/p95 in GUI
- [ ] 9.2 Result/Ablation/Reward Decomposition panels
- [ ] 9.3 Phase 9 test and commit

## Faz 10 - LLM Self-Reflection & Experience Replay
- [ ] 10.1 Post-hoc analysis for bad decisions
- [ ] 10.2 Explanation bank as RAG/few-shot memory
- [ ] 10.3 Final technical documentation
- [ ] 10.4 Final test and commit
