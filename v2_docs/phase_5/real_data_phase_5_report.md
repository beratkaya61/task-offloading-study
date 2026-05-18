Bkz. ortak kavram sozlugu: v2_docs/project_concepts_glossary.md

# Faz 5 - Real Data Report

Bu dosya, Faz 5'in `real_data` kolu icin tek kanonik rapordur.
Bu kol `real-world trace-driven hybrid benchmark` olarak adlandirilir; tam gercek MEC logu iddiasi tasimaz.
Sentetik Faz 5 sonuclari ayri olarak `v2_docs/phase_5/synthetic_phase_5_report.md` dosyasinda tutulur.

## Phase 5R Gate

- Ana basari tanimi degismez: task fiziksel gecikme altinda deadline'i tutturursa basarilidir.
- Oracle/feasibility audit uretilmeden real-data ablation sonuclari nihai bilimsel iddia olarak okunmaz.
- Politika tek aksiyona cokuyorsa, sonuc action-collapse diagnostigi gecmeden final iddia olarak kullanilmaz.

## Artefakt Haritasi

- `results/phase_5/metrics/real_data/rl_retraining/`
- `results/phase_5/metrics/real_data/policy_evaluation/`
- `results/phase_5/metrics/real_data/ablation/`
- `results/phase_5/metrics/real_data/oracle/`
- `results/phase_5/figures/real_data/ablation/`

Oracle/feasibility gate raporu mevcut:
- `v2_docs/phase_5/real_data_oracle_audit_service_class.md`

Benchmark validity diagnostic raporu mevcut:
- `v2_docs/phase_5/benchmark_validity_service_class.md`

## RL Retraining

| Algorithm | Seed | Success Rate | Miss Ratio | Avg Latency | P95 | P99 | Avg Energy | Energy/Success | Partial Ratio | Overhead ms | QoE | Dominant Action |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PPO | 42 | 66.20% | 33.80% | 0.6107 | 1.4940 | 2.7296 | 0.0462 | 0.0698 | 100.00% | 0.2538 | 58.73 | 3 |
| PPO | 43 | 66.40% | 33.60% | 0.6108 | 1.4935 | 2.7303 | 0.0463 | 0.0697 | 100.00% | 0.2476 | 58.93 | 3 |
| PPO | 44 | 66.60% | 33.40% | 0.6107 | 1.4932 | 2.7229 | 0.0462 | 0.0694 | 100.00% | 0.2445 | 59.13 | 3 |
| DQN | 42 | 24.00% | 76.00% | 0.7846 | 1.9457 | 3.5962 | 0.0178 | 0.0742 | 0.00% | 0.1531 | 14.27 | 4 |
| DQN | 43 | 49.00% | 51.00% | 0.6988 | 1.8229 | 3.4391 | 0.0748 | 0.1527 | 100.00% | 0.1683 | 39.89 | 2 |
| DQN | 44 | 66.40% | 33.60% | 0.6107 | 1.4931 | 2.7308 | 0.0463 | 0.0697 | 100.00% | 0.1761 | 58.93 | 3 |
| A2C | 42 | 66.20% | 33.80% | 0.6109 | 1.4938 | 2.7317 | 0.0464 | 0.0700 | 100.00% | 0.2435 | 58.73 | 3 |
| A2C | 43 | 66.20% | 33.80% | 0.6108 | 1.4940 | 2.7336 | 0.0463 | 0.0699 | 100.00% | 0.2621 | 58.73 | 3 |
| A2C | 44 | 66.20% | 33.80% | 0.6107 | 1.4938 | 2.7282 | 0.0462 | 0.0699 | 100.00% | 0.2523 | 58.73 | 3 |

### RL Retraining Action-Collapse Diagnostic

- `PPO` seed `42` tek aksiyona cokuyor: action `3` = 100.00%.
- `PPO` seed `43` tek aksiyona cokuyor: action `3` = 100.00%.
- `PPO` seed `44` tek aksiyona cokuyor: action `3` = 100.00%.
- `DQN` seed `42` tek aksiyona cokuyor: action `4` = 100.00%.
- `DQN` seed `43` tek aksiyona cokuyor: action `2` = 100.00%.
- `DQN` seed `44` tek aksiyona cokuyor: action `3` = 100.00%.
- `A2C` seed `42` tek aksiyona cokuyor: action `3` = 100.00%.
- `A2C` seed `43` tek aksiyona cokuyor: action `3` = 100.00%.
- `A2C` seed `44` tek aksiyona cokuyor: action `3` = 100.00%.

## Policy Evaluation

| Policy | Seed | Success Rate | Miss Ratio | Avg Latency | P95 | P99 | Avg Energy | Energy/Success | Partial Ratio | Overhead ms | QoE | Dominant Action |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LocalOnly | 42 | 21.00% | 79.00% | 1.3195 | 3.5580 | 6.8182 | 0.1319 | 0.6283 | 0.00% | 0.0006 | 3.21 | 0 |
| EdgeOnly | 42 | 24.00% | 76.00% | 0.7844 | 1.9470 | 3.5982 | 0.0177 | 0.0739 | 0.00% | 0.0004 | 14.27 | 4 |
| CloudOnly | 42 | 8.00% | 92.00% | 0.9519 | 1.4626 | 2.1583 | 0.0178 | 0.2224 | 0.00% | 0.0005 | 0.69 | 5 |
| Random | 42 | 34.60% | 65.40% | 0.8984 | 2.7597 | 5.1609 | 0.0688 | 0.1990 | 51.60% | 0.0014 | 20.80 | 3 |
| GreedyLatency | 42 | 76.20% | 23.80% | 0.5521 | 1.2328 | 1.9820 | 0.0355 | 0.0466 | 69.00% | 0.0218 | 70.04 | 3 |
| DeadlineAwareGreedy | 42 | 76.60% | 23.40% | 0.5523 | 1.2362 | 1.9872 | 0.0354 | 0.0462 | 68.00% | 0.0240 | 70.42 | 3 |
| GeneticAlgorithm | 42 | 75.40% | 24.60% | 0.5563 | 1.2341 | 1.9818 | 0.0358 | 0.0475 | 68.40% | 0.2298 | 69.23 | 3 |
| PPO | 42 | 66.40% | 33.60% | 0.6106 | 1.4940 | 2.7228 | 0.0462 | 0.0696 | 100.00% | 0.2524 | 58.93 | 3 |
| DQN | 42 | 24.00% | 76.00% | 0.7843 | 1.9445 | 3.5967 | 0.0177 | 0.0737 | 0.00% | 0.1342 | 14.28 | 4 |
| A2C | 42 | 66.40% | 33.60% | 0.6106 | 1.4933 | 2.7311 | 0.0462 | 0.0696 | 100.00% | 0.2316 | 58.93 | 3 |
| LocalOnly | 43 | 21.00% | 79.00% | 1.3195 | 3.5580 | 6.8182 | 0.1319 | 0.6283 | 0.00% | 0.0004 | 3.21 | 0 |
| EdgeOnly | 43 | 24.00% | 76.00% | 0.7844 | 1.9449 | 3.5899 | 0.0177 | 0.0739 | 0.00% | 0.0004 | 14.28 | 4 |
| CloudOnly | 43 | 8.00% | 92.00% | 0.9518 | 1.4628 | 2.1600 | 0.0177 | 0.2218 | 0.00% | 0.0004 | 0.69 | 5 |
| Random | 43 | 34.80% | 65.20% | 0.8527 | 2.2092 | 3.9546 | 0.0634 | 0.1822 | 48.00% | 0.0013 | 23.75 | 5 |
| GreedyLatency | 43 | 76.40% | 23.60% | 0.5519 | 1.2343 | 1.9761 | 0.0354 | 0.0463 | 68.80% | 0.0227 | 70.23 | 3 |
| DeadlineAwareGreedy | 43 | 76.40% | 23.60% | 0.5523 | 1.2365 | 1.9883 | 0.0354 | 0.0464 | 68.00% | 0.0253 | 70.22 | 3 |
| GeneticAlgorithm | 43 | 76.00% | 24.00% | 0.5585 | 1.2284 | 2.0506 | 0.0361 | 0.0475 | 68.20% | 0.2330 | 69.86 | 3 |
| PPO | 43 | 66.20% | 33.80% | 0.6108 | 1.4938 | 2.7283 | 0.0463 | 0.0699 | 100.00% | 0.2771 | 58.73 | 3 |
| DQN | 43 | 49.00% | 51.00% | 0.6988 | 1.8229 | 3.4391 | 0.0748 | 0.1527 | 100.00% | 0.1343 | 39.89 | 2 |
| A2C | 43 | 66.40% | 33.60% | 0.6108 | 1.4938 | 2.7309 | 0.0463 | 0.0697 | 100.00% | 0.2309 | 58.93 | 3 |
| LocalOnly | 44 | 21.00% | 79.00% | 1.3195 | 3.5580 | 6.8182 | 0.1319 | 0.6283 | 0.00% | 0.0004 | 3.21 | 0 |
| EdgeOnly | 44 | 24.00% | 76.00% | 0.7844 | 1.9451 | 3.5979 | 0.0178 | 0.0740 | 0.00% | 0.0004 | 14.27 | 4 |
| CloudOnly | 44 | 8.00% | 92.00% | 0.9517 | 1.4624 | 2.1577 | 0.0177 | 0.2216 | 0.00% | 0.0004 | 0.69 | 5 |
| Random | 44 | 33.60% | 66.40% | 0.8575 | 1.8824 | 4.0640 | 0.0646 | 0.1924 | 49.80% | 0.0013 | 24.19 | 5 |
| GreedyLatency | 44 | 76.20% | 23.80% | 0.5520 | 1.2336 | 1.9755 | 0.0355 | 0.0466 | 69.00% | 0.0216 | 70.03 | 3 |
| DeadlineAwareGreedy | 44 | 76.60% | 23.40% | 0.5523 | 1.2367 | 1.9827 | 0.0354 | 0.0462 | 68.00% | 0.0240 | 70.42 | 3 |
| GeneticAlgorithm | 44 | 75.00% | 25.00% | 0.5559 | 1.2393 | 1.9768 | 0.0359 | 0.0478 | 68.80% | 0.2274 | 68.80 | 3 |
| PPO | 44 | 66.20% | 33.80% | 0.6107 | 1.4935 | 2.7326 | 0.0462 | 0.0698 | 100.00% | 0.2357 | 58.73 | 3 |
| DQN | 44 | 66.60% | 33.40% | 0.6107 | 1.4935 | 2.7278 | 0.0463 | 0.0695 | 100.00% | 0.1377 | 59.13 | 3 |
| A2C | 44 | 66.40% | 33.60% | 0.6108 | 1.4943 | 2.7319 | 0.0463 | 0.0698 | 100.00% | 0.2300 | 58.93 | 3 |

### Policy Evaluation Action-Collapse Diagnostic

- `LocalOnly` seed `42` tek aksiyona cokuyor: action `0` = 100.00%.
- `EdgeOnly` seed `42` tek aksiyona cokuyor: action `4` = 100.00%.
- `CloudOnly` seed `42` tek aksiyona cokuyor: action `5` = 100.00%.
- `PPO` seed `42` tek aksiyona cokuyor: action `3` = 100.00%.
- `DQN` seed `42` tek aksiyona cokuyor: action `4` = 100.00%.
- `A2C` seed `42` tek aksiyona cokuyor: action `3` = 100.00%.
- `LocalOnly` seed `43` tek aksiyona cokuyor: action `0` = 100.00%.
- `EdgeOnly` seed `43` tek aksiyona cokuyor: action `4` = 100.00%.
- `CloudOnly` seed `43` tek aksiyona cokuyor: action `5` = 100.00%.
- `PPO` seed `43` tek aksiyona cokuyor: action `3` = 100.00%.
- `DQN` seed `43` tek aksiyona cokuyor: action `2` = 100.00%.
- `A2C` seed `43` tek aksiyona cokuyor: action `3` = 100.00%.
- `LocalOnly` seed `44` tek aksiyona cokuyor: action `0` = 100.00%.
- `EdgeOnly` seed `44` tek aksiyona cokuyor: action `4` = 100.00%.
- `CloudOnly` seed `44` tek aksiyona cokuyor: action `5` = 100.00%.
- `PPO` seed `44` tek aksiyona cokuyor: action `3` = 100.00%.
- `DQN` seed `44` tek aksiyona cokuyor: action `3` = 100.00%.
- `A2C` seed `44` tek aksiyona cokuyor: action `3` = 100.00%.

## Ablation

### a2c_multi_seed_evaluation

| Variant | Seed | Success Rate | Miss Ratio | Avg Latency | P95 | P99 | Avg Energy | Energy/Success | Partial Ratio | Overhead ms | QoE | Dominant Action |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full_model | 42 | 66.20% | 33.80% | 0.6107 | 1.4937 | 2.7276 | 0.0463 | 0.0699 | 100.00% | 0.3027 | 58.73 | 3 |
| full_model | 43 | 66.20% | 33.80% | 0.6107 | 1.4939 | 2.7263 | 0.0463 | 0.0699 | 100.00% | 0.2758 | 58.73 | 3 |
| full_model | 44 | 49.00% | 51.00% | 0.6988 | 1.8229 | 3.4391 | 0.0748 | 0.1527 | 100.00% | 0.3092 | 39.89 | 2 |
| w_o_semantics | 42 | 24.00% | 76.00% | 0.7845 | 1.9457 | 3.5900 | 0.0178 | 0.0741 | 0.00% | 0.1582 | 14.27 | 4 |
| w_o_semantics | 43 | 60.60% | 39.40% | 0.6213 | 1.5051 | 2.7318 | 0.0495 | 0.0817 | 100.00% | 0.1770 | 53.07 | 3 |
| w_o_semantics | 44 | 66.40% | 33.60% | 0.6107 | 1.4936 | 2.7265 | 0.0463 | 0.0697 | 100.00% | 0.1873 | 58.93 | 3 |
| w_o_reward_shaping | 42 | 49.00% | 51.00% | 0.6988 | 1.8229 | 3.4391 | 0.0748 | 0.1528 | 100.00% | 0.2579 | 39.89 | 2 |
| w_o_reward_shaping | 43 | 23.00% | 77.00% | 0.7253 | 1.8142 | 2.0414 | 0.0177 | 0.0771 | 0.00% | 0.2359 | 13.93 | 4 |
| w_o_reward_shaping | 44 | 24.00% | 76.00% | 0.7845 | 1.9458 | 3.5995 | 0.0178 | 0.0741 | 0.00% | 0.2342 | 14.27 | 4 |
| w_o_semantic_prior | 42 | 66.20% | 33.80% | 0.6107 | 1.4937 | 2.7228 | 0.0462 | 0.0699 | 100.00% | 0.3007 | 58.73 | 3 |
| w_o_semantic_prior | 43 | 49.00% | 51.00% | 0.6988 | 1.8229 | 3.4391 | 0.0748 | 0.1527 | 100.00% | 0.2539 | 39.89 | 2 |
| w_o_semantic_prior | 44 | 66.40% | 33.60% | 0.6107 | 1.4938 | 2.7228 | 0.0462 | 0.0696 | 100.00% | 0.2801 | 58.93 | 3 |
| w_o_confidence | 42 | 45.80% | 54.20% | 0.7017 | 1.8229 | 3.4391 | 0.0746 | 0.1629 | 86.60% | 0.2532 | 36.69 | 2 |
| w_o_confidence | 43 | 66.40% | 33.60% | 0.6108 | 1.4939 | 2.7318 | 0.0463 | 0.0698 | 100.00% | 0.2426 | 58.93 | 3 |
| w_o_confidence | 44 | 66.20% | 33.80% | 0.6109 | 1.4933 | 2.7272 | 0.0464 | 0.0700 | 100.00% | 0.2465 | 58.73 | 3 |
| w_o_partial_offloading | 42 | 24.20% | 75.80% | 0.7843 | 1.9457 | 3.6025 | 0.0177 | 0.0731 | 0.00% | 0.2426 | 14.47 | 2 |
| w_o_partial_offloading | 43 | 21.00% | 79.00% | 1.3195 | 3.5580 | 6.8182 | 0.1319 | 0.6283 | 0.00% | 0.2686 | 3.21 | 1 |
| w_o_partial_offloading | 44 | 24.20% | 75.80% | 0.7843 | 1.9466 | 3.6051 | 0.0177 | 0.0732 | 0.00% | 0.2879 | 14.47 | 2 |
| w_o_battery_awareness | 42 | 66.40% | 33.60% | 0.6107 | 1.4941 | 2.7245 | 0.0463 | 0.0697 | 100.00% | 0.2427 | 58.93 | 3 |
| w_o_battery_awareness | 43 | 49.00% | 51.00% | 0.6988 | 1.8229 | 3.4391 | 0.0748 | 0.1527 | 100.00% | 0.2608 | 39.89 | 2 |
| w_o_battery_awareness | 44 | 49.00% | 51.00% | 0.6988 | 1.8229 | 3.4391 | 0.0748 | 0.1527 | 100.00% | 0.3007 | 39.89 | 2 |
| w_o_queue_awareness | 42 | 66.40% | 33.60% | 0.6108 | 1.4942 | 2.7318 | 0.0463 | 0.0697 | 100.00% | 0.2970 | 58.93 | 3 |
| w_o_queue_awareness | 43 | 66.20% | 33.80% | 0.6108 | 1.4940 | 2.7299 | 0.0463 | 0.0700 | 100.00% | 0.2305 | 58.73 | 3 |
| w_o_queue_awareness | 44 | 66.00% | 34.00% | 0.6108 | 1.4937 | 2.7304 | 0.0463 | 0.0701 | 100.00% | 0.2548 | 58.53 | 3 |
| w_o_mobility_features | 42 | 61.80% | 38.20% | 0.6228 | 1.5045 | 2.7229 | 0.0506 | 0.0818 | 100.00% | 0.2783 | 54.28 | 3 |
| w_o_mobility_features | 43 | 66.40% | 33.60% | 0.6107 | 1.4938 | 2.7250 | 0.0463 | 0.0697 | 100.00% | 0.2854 | 58.93 | 3 |
| w_o_mobility_features | 44 | 66.60% | 33.40% | 0.6107 | 1.4933 | 2.7305 | 0.0462 | 0.0694 | 100.00% | 0.2885 | 59.13 | 3 |

### a2c_multi_seed_evaluation Action-Collapse Diagnostic

- `full_model` seed `42` tek aksiyona cokuyor: action `3` = 100.00%.
- `full_model` seed `43` tek aksiyona cokuyor: action `3` = 100.00%.
- `full_model` seed `44` tek aksiyona cokuyor: action `2` = 100.00%.
- `w_o_semantics` seed `42` tek aksiyona cokuyor: action `4` = 100.00%.
- `w_o_semantics` seed `44` tek aksiyona cokuyor: action `3` = 100.00%.
- `w_o_reward_shaping` seed `42` tek aksiyona cokuyor: action `2` = 100.00%.
- `w_o_reward_shaping` seed `43` tek aksiyona cokuyor: action `4` = 94.20%.
- `w_o_reward_shaping` seed `44` tek aksiyona cokuyor: action `4` = 100.00%.
- `w_o_semantic_prior` seed `42` tek aksiyona cokuyor: action `3` = 100.00%.
- `w_o_semantic_prior` seed `43` tek aksiyona cokuyor: action `2` = 100.00%.
- `w_o_semantic_prior` seed `44` tek aksiyona cokuyor: action `3` = 100.00%.
- `w_o_confidence` seed `43` tek aksiyona cokuyor: action `3` = 100.00%.
- `w_o_confidence` seed `44` tek aksiyona cokuyor: action `3` = 100.00%.
- `w_o_partial_offloading` seed `42` tek aksiyona cokuyor: action `2` = 100.00%.
- `w_o_partial_offloading` seed `43` tek aksiyona cokuyor: action `1` = 100.00%.
- `w_o_partial_offloading` seed `44` tek aksiyona cokuyor: action `2` = 100.00%.
- `w_o_battery_awareness` seed `42` tek aksiyona cokuyor: action `3` = 100.00%.
- `w_o_battery_awareness` seed `43` tek aksiyona cokuyor: action `2` = 100.00%.
- `w_o_battery_awareness` seed `44` tek aksiyona cokuyor: action `2` = 100.00%.
- `w_o_queue_awareness` seed `42` tek aksiyona cokuyor: action `3` = 100.00%.
- `w_o_queue_awareness` seed `43` tek aksiyona cokuyor: action `3` = 100.00%.
- `w_o_queue_awareness` seed `44` tek aksiyona cokuyor: action `3` = 100.00%.
- `w_o_mobility_features` seed `43` tek aksiyona cokuyor: action `3` = 100.00%.
- `w_o_mobility_features` seed `44` tek aksiyona cokuyor: action `3` = 100.00%.

### a2c_multi_seed_retraining

| Variant | Seed | Success Rate | Miss Ratio | Avg Latency | P95 | P99 | Avg Energy | Energy/Success | Partial Ratio | Overhead ms | QoE | Dominant Action |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full_model | 42 | 66.40% | 33.60% | 0.6106 | 1.4942 | 2.7227 | 0.0462 | 0.0696 | 100.00% | 0.3015 | 58.93 | 3 |
| full_model | 43 | 66.40% | 33.60% | 0.6107 | 1.4937 | 2.7328 | 0.0463 | 0.0697 | 100.00% | 0.3109 | 58.93 | 3 |
| full_model | 44 | 49.00% | 51.00% | 0.6988 | 1.8229 | 3.4391 | 0.0748 | 0.1527 | 100.00% | 0.2685 | 39.89 | 2 |
| w_o_semantics | 42 | 24.00% | 76.00% | 0.7845 | 1.9453 | 3.6000 | 0.0178 | 0.0741 | 0.00% | 0.1822 | 14.27 | 4 |
| w_o_semantics | 43 | 60.40% | 39.60% | 0.6211 | 1.5052 | 2.7328 | 0.0494 | 0.0818 | 100.00% | 0.1674 | 52.87 | 3 |
| w_o_semantics | 44 | 66.40% | 33.60% | 0.6107 | 1.4941 | 2.7284 | 0.0463 | 0.0697 | 100.00% | 0.2169 | 58.93 | 3 |
| w_o_reward_shaping | 42 | 49.00% | 51.00% | 0.6988 | 1.8229 | 3.4391 | 0.0748 | 0.1527 | 100.00% | 0.2572 | 39.89 | 2 |
| w_o_reward_shaping | 43 | 23.00% | 77.00% | 0.7252 | 1.8147 | 2.0417 | 0.0177 | 0.0770 | 0.00% | 0.2998 | 13.93 | 4 |
| w_o_reward_shaping | 44 | 24.00% | 76.00% | 0.7844 | 1.9452 | 3.6001 | 0.0177 | 0.0739 | 0.00% | 0.2291 | 14.27 | 4 |
| w_o_semantic_prior | 42 | 66.40% | 33.60% | 0.6109 | 1.4936 | 2.7311 | 0.0463 | 0.0698 | 100.00% | 0.2690 | 58.93 | 3 |
| w_o_semantic_prior | 43 | 49.00% | 51.00% | 0.6988 | 1.8229 | 3.4391 | 0.0748 | 0.1527 | 100.00% | 0.2999 | 39.89 | 2 |
| w_o_semantic_prior | 44 | 66.20% | 33.80% | 0.6109 | 1.4942 | 2.7231 | 0.0463 | 0.0700 | 100.00% | 0.2836 | 58.73 | 3 |
| w_o_confidence | 42 | 45.80% | 54.20% | 0.7017 | 1.8229 | 3.4391 | 0.0746 | 0.1629 | 86.60% | 0.2704 | 36.69 | 2 |
| w_o_confidence | 43 | 66.60% | 33.40% | 0.6106 | 1.4942 | 2.7237 | 0.0462 | 0.0694 | 100.00% | 0.2863 | 59.13 | 3 |
| w_o_confidence | 44 | 66.40% | 33.60% | 0.6106 | 1.4935 | 2.7281 | 0.0462 | 0.0696 | 100.00% | 0.2550 | 58.93 | 3 |
| w_o_partial_offloading | 42 | 24.00% | 76.00% | 0.7845 | 1.9456 | 3.5920 | 0.0178 | 0.0741 | 0.00% | 0.2703 | 14.27 | 2 |
| w_o_partial_offloading | 43 | 21.00% | 79.00% | 1.3195 | 3.5580 | 6.8182 | 0.1319 | 0.6283 | 0.00% | 0.2531 | 3.21 | 1 |
| w_o_partial_offloading | 44 | 24.20% | 75.80% | 0.7845 | 1.9450 | 3.6045 | 0.0178 | 0.0735 | 0.00% | 0.2382 | 14.47 | 2 |
| w_o_battery_awareness | 42 | 66.40% | 33.60% | 0.6107 | 1.4940 | 2.7299 | 0.0462 | 0.0696 | 100.00% | 0.2421 | 58.93 | 3 |
| w_o_battery_awareness | 43 | 49.00% | 51.00% | 0.6988 | 1.8229 | 3.4391 | 0.0749 | 0.1529 | 100.00% | 0.2618 | 39.89 | 2 |
| w_o_battery_awareness | 44 | 49.00% | 51.00% | 0.6988 | 1.8229 | 3.4391 | 0.0748 | 0.1527 | 100.00% | 0.2695 | 39.89 | 2 |
| w_o_queue_awareness | 42 | 66.20% | 33.80% | 0.6107 | 1.4937 | 2.7316 | 0.0462 | 0.0699 | 100.00% | 0.2533 | 58.73 | 3 |
| w_o_queue_awareness | 43 | 66.20% | 33.80% | 0.6108 | 1.4934 | 2.7246 | 0.0463 | 0.0700 | 100.00% | 0.2627 | 58.73 | 3 |
| w_o_queue_awareness | 44 | 66.20% | 33.80% | 0.6108 | 1.4941 | 2.7332 | 0.0463 | 0.0699 | 100.00% | 0.2387 | 58.73 | 3 |
| w_o_mobility_features | 42 | 62.00% | 38.00% | 0.6228 | 1.5044 | 2.7283 | 0.0505 | 0.0815 | 100.00% | 0.2459 | 54.48 | 3 |
| w_o_mobility_features | 43 | 66.20% | 33.80% | 0.6108 | 1.4936 | 2.7334 | 0.0463 | 0.0699 | 100.00% | 0.2713 | 58.73 | 3 |
| w_o_mobility_features | 44 | 66.60% | 33.40% | 0.6107 | 1.4934 | 2.7295 | 0.0462 | 0.0694 | 100.00% | 0.2599 | 59.13 | 3 |

### a2c_multi_seed_retraining Action-Collapse Diagnostic

- `full_model` seed `42` tek aksiyona cokuyor: action `3` = 100.00%.
- `full_model` seed `43` tek aksiyona cokuyor: action `3` = 100.00%.
- `full_model` seed `44` tek aksiyona cokuyor: action `2` = 100.00%.
- `w_o_semantics` seed `42` tek aksiyona cokuyor: action `4` = 100.00%.
- `w_o_semantics` seed `44` tek aksiyona cokuyor: action `3` = 100.00%.
- `w_o_reward_shaping` seed `42` tek aksiyona cokuyor: action `2` = 100.00%.
- `w_o_reward_shaping` seed `43` tek aksiyona cokuyor: action `4` = 94.20%.
- `w_o_reward_shaping` seed `44` tek aksiyona cokuyor: action `4` = 100.00%.
- `w_o_semantic_prior` seed `42` tek aksiyona cokuyor: action `3` = 100.00%.
- `w_o_semantic_prior` seed `43` tek aksiyona cokuyor: action `2` = 100.00%.
- `w_o_semantic_prior` seed `44` tek aksiyona cokuyor: action `3` = 100.00%.
- `w_o_confidence` seed `43` tek aksiyona cokuyor: action `3` = 100.00%.
- `w_o_confidence` seed `44` tek aksiyona cokuyor: action `3` = 100.00%.
- `w_o_partial_offloading` seed `42` tek aksiyona cokuyor: action `2` = 100.00%.
- `w_o_partial_offloading` seed `43` tek aksiyona cokuyor: action `1` = 100.00%.
- `w_o_partial_offloading` seed `44` tek aksiyona cokuyor: action `2` = 100.00%.
- `w_o_battery_awareness` seed `42` tek aksiyona cokuyor: action `3` = 100.00%.
- `w_o_battery_awareness` seed `43` tek aksiyona cokuyor: action `2` = 100.00%.
- `w_o_battery_awareness` seed `44` tek aksiyona cokuyor: action `2` = 100.00%.
- `w_o_queue_awareness` seed `42` tek aksiyona cokuyor: action `3` = 100.00%.
- `w_o_queue_awareness` seed `43` tek aksiyona cokuyor: action `3` = 100.00%.
- `w_o_queue_awareness` seed `44` tek aksiyona cokuyor: action `3` = 100.00%.
- `w_o_mobility_features` seed `43` tek aksiyona cokuyor: action `3` = 100.00%.
- `w_o_mobility_features` seed `44` tek aksiyona cokuyor: action `3` = 100.00%.

### dqn_multi_seed_evaluation

| Variant | Seed | Success Rate | Miss Ratio | Avg Latency | P95 | P99 | Avg Energy | Energy/Success | Partial Ratio | Overhead ms | QoE | Dominant Action |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full_model | 42 | 49.00% | 51.00% | 0.6988 | 1.8229 | 3.4391 | 0.0748 | 0.1528 | 100.00% | 0.1364 | 39.89 | 2 |
| full_model | 43 | 24.20% | 75.80% | 0.7842 | 1.9463 | 3.5934 | 0.0176 | 0.0728 | 0.00% | 0.1353 | 14.47 | 4 |
| full_model | 44 | 49.00% | 51.00% | 0.6988 | 1.8229 | 3.4391 | 0.0748 | 0.1527 | 100.00% | 0.1532 | 39.89 | 2 |
| w_o_semantics | 42 | 66.40% | 33.60% | 0.6106 | 1.4936 | 2.7307 | 0.0462 | 0.0696 | 100.00% | 0.0829 | 58.93 | 3 |
| w_o_semantics | 43 | 21.60% | 78.40% | 1.0249 | 2.7140 | 5.1436 | 0.1034 | 0.4788 | 100.00% | 0.0886 | 8.03 | 1 |
| w_o_semantics | 44 | 66.20% | 33.80% | 0.6107 | 1.4938 | 2.7277 | 0.0462 | 0.0699 | 100.00% | 0.0910 | 58.73 | 3 |
| w_o_reward_shaping | 42 | 66.20% | 33.80% | 0.6108 | 1.4940 | 2.7277 | 0.0463 | 0.0700 | 100.00% | 0.1392 | 58.73 | 3 |
| w_o_reward_shaping | 43 | 66.20% | 33.80% | 0.6108 | 1.4934 | 2.7274 | 0.0463 | 0.0699 | 100.00% | 0.1370 | 58.73 | 3 |
| w_o_reward_shaping | 44 | 66.20% | 33.80% | 0.6108 | 1.4939 | 2.7280 | 0.0463 | 0.0699 | 100.00% | 0.1446 | 58.73 | 3 |
| w_o_semantic_prior | 42 | 49.00% | 51.00% | 0.6988 | 1.8229 | 3.4391 | 0.0748 | 0.1527 | 100.00% | 0.1378 | 39.89 | 2 |
| w_o_semantic_prior | 43 | 66.40% | 33.60% | 0.6106 | 1.4936 | 2.7327 | 0.0462 | 0.0696 | 100.00% | 0.1390 | 58.93 | 3 |
| w_o_semantic_prior | 44 | 49.00% | 51.00% | 0.6988 | 1.8229 | 3.4391 | 0.0748 | 0.1527 | 100.00% | 0.1361 | 39.89 | 2 |
| w_o_confidence | 42 | 24.00% | 76.00% | 0.7845 | 1.9455 | 3.5973 | 0.0178 | 0.0740 | 0.00% | 0.1380 | 14.27 | 4 |
| w_o_confidence | 43 | 66.20% | 33.80% | 0.6107 | 1.4937 | 2.7236 | 0.0462 | 0.0699 | 100.00% | 0.1362 | 58.73 | 3 |
| w_o_confidence | 44 | 24.00% | 76.00% | 0.7845 | 1.9458 | 3.6020 | 0.0178 | 0.0741 | 0.00% | 0.1373 | 14.27 | 4 |
| w_o_partial_offloading | 42 | 24.00% | 76.00% | 0.7844 | 1.9453 | 3.5987 | 0.0178 | 0.0740 | 0.00% | 0.1399 | 14.27 | 2 |
| w_o_partial_offloading | 43 | 24.00% | 76.00% | 0.7844 | 1.9461 | 3.6037 | 0.0177 | 0.0739 | 0.00% | 0.1372 | 14.27 | 3 |
| w_o_partial_offloading | 44 | 24.00% | 76.00% | 0.7845 | 1.9453 | 3.6014 | 0.0178 | 0.0741 | 0.00% | 0.1355 | 14.27 | 2 |
| w_o_battery_awareness | 42 | 66.20% | 33.80% | 0.6108 | 1.4934 | 2.7225 | 0.0463 | 0.0700 | 100.00% | 0.1364 | 58.73 | 3 |
| w_o_battery_awareness | 43 | 66.20% | 33.80% | 0.6107 | 1.4935 | 2.7291 | 0.0462 | 0.0699 | 100.00% | 0.1369 | 58.73 | 3 |
| w_o_battery_awareness | 44 | 49.00% | 51.00% | 0.6988 | 1.8229 | 3.4391 | 0.0749 | 0.1528 | 100.00% | 0.1371 | 39.89 | 2 |
| w_o_queue_awareness | 42 | 49.00% | 51.00% | 0.6988 | 1.8229 | 3.4391 | 0.0748 | 0.1527 | 100.00% | 0.1357 | 39.89 | 2 |
| w_o_queue_awareness | 43 | 49.00% | 51.00% | 0.6988 | 1.8229 | 3.4391 | 0.0748 | 0.1527 | 100.00% | 0.1378 | 39.89 | 2 |
| w_o_queue_awareness | 44 | 49.00% | 51.00% | 0.6988 | 1.8229 | 3.4391 | 0.0748 | 0.1527 | 100.00% | 0.1362 | 39.89 | 2 |
| w_o_mobility_features | 42 | 66.20% | 33.80% | 0.6107 | 1.4940 | 2.7252 | 0.0463 | 0.0699 | 100.00% | 0.1375 | 58.73 | 3 |
| w_o_mobility_features | 43 | 24.00% | 76.00% | 0.7845 | 1.9462 | 3.6010 | 0.0178 | 0.0740 | 0.00% | 0.1386 | 14.27 | 4 |
| w_o_mobility_features | 44 | 66.20% | 33.80% | 0.6108 | 1.4934 | 2.7242 | 0.0463 | 0.0699 | 100.00% | 0.1362 | 58.73 | 3 |

### dqn_multi_seed_evaluation Action-Collapse Diagnostic

- `full_model` seed `42` tek aksiyona cokuyor: action `2` = 100.00%.
- `full_model` seed `43` tek aksiyona cokuyor: action `4` = 100.00%.
- `full_model` seed `44` tek aksiyona cokuyor: action `2` = 100.00%.
- `w_o_semantics` seed `42` tek aksiyona cokuyor: action `3` = 100.00%.
- `w_o_semantics` seed `43` tek aksiyona cokuyor: action `1` = 100.00%.
- `w_o_semantics` seed `44` tek aksiyona cokuyor: action `3` = 100.00%.
- `w_o_reward_shaping` seed `42` tek aksiyona cokuyor: action `3` = 100.00%.
- `w_o_reward_shaping` seed `43` tek aksiyona cokuyor: action `3` = 100.00%.
- `w_o_reward_shaping` seed `44` tek aksiyona cokuyor: action `3` = 100.00%.
- `w_o_semantic_prior` seed `42` tek aksiyona cokuyor: action `2` = 100.00%.
- `w_o_semantic_prior` seed `43` tek aksiyona cokuyor: action `3` = 100.00%.
- `w_o_semantic_prior` seed `44` tek aksiyona cokuyor: action `2` = 100.00%.
- `w_o_confidence` seed `42` tek aksiyona cokuyor: action `4` = 100.00%.
- `w_o_confidence` seed `43` tek aksiyona cokuyor: action `3` = 100.00%.
- `w_o_confidence` seed `44` tek aksiyona cokuyor: action `4` = 100.00%.
- `w_o_partial_offloading` seed `42` tek aksiyona cokuyor: action `2` = 100.00%.
- `w_o_partial_offloading` seed `43` tek aksiyona cokuyor: action `3` = 100.00%.
- `w_o_partial_offloading` seed `44` tek aksiyona cokuyor: action `2` = 100.00%.
- `w_o_battery_awareness` seed `42` tek aksiyona cokuyor: action `3` = 100.00%.
- `w_o_battery_awareness` seed `43` tek aksiyona cokuyor: action `3` = 100.00%.
- `w_o_battery_awareness` seed `44` tek aksiyona cokuyor: action `2` = 100.00%.
- `w_o_queue_awareness` seed `42` tek aksiyona cokuyor: action `2` = 100.00%.
- `w_o_queue_awareness` seed `43` tek aksiyona cokuyor: action `2` = 100.00%.
- `w_o_queue_awareness` seed `44` tek aksiyona cokuyor: action `2` = 100.00%.
- `w_o_mobility_features` seed `42` tek aksiyona cokuyor: action `3` = 100.00%.
- `w_o_mobility_features` seed `43` tek aksiyona cokuyor: action `4` = 100.00%.
- `w_o_mobility_features` seed `44` tek aksiyona cokuyor: action `3` = 100.00%.

### dqn_multi_seed_retraining

| Variant | Seed | Success Rate | Miss Ratio | Avg Latency | P95 | P99 | Avg Energy | Energy/Success | Partial Ratio | Overhead ms | QoE | Dominant Action |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full_model | 42 | 49.00% | 51.00% | 0.6988 | 1.8229 | 3.4391 | 0.0748 | 0.1528 | 100.00% | 0.1369 | 39.89 | 2 |
| full_model | 43 | 24.00% | 76.00% | 0.7843 | 1.9451 | 3.5954 | 0.0177 | 0.0737 | 0.00% | 0.1352 | 14.27 | 4 |
| full_model | 44 | 49.00% | 51.00% | 0.6988 | 1.8229 | 3.4391 | 0.0748 | 0.1527 | 100.00% | 0.1326 | 39.89 | 2 |
| w_o_semantics | 42 | 66.20% | 33.80% | 0.6109 | 1.4943 | 2.7330 | 0.0464 | 0.0700 | 100.00% | 0.0958 | 58.73 | 3 |
| w_o_semantics | 43 | 21.60% | 78.40% | 1.0249 | 2.7140 | 5.1436 | 0.1034 | 0.4786 | 100.00% | 0.1019 | 8.03 | 1 |
| w_o_semantics | 44 | 66.40% | 33.60% | 0.6107 | 1.4938 | 2.7298 | 0.0463 | 0.0697 | 100.00% | 0.1055 | 58.93 | 3 |
| w_o_reward_shaping | 42 | 66.40% | 33.60% | 0.6107 | 1.4939 | 2.7323 | 0.0463 | 0.0697 | 100.00% | 0.1370 | 58.93 | 3 |
| w_o_reward_shaping | 43 | 66.20% | 33.80% | 0.6108 | 1.4934 | 2.7257 | 0.0463 | 0.0699 | 100.00% | 0.1367 | 58.73 | 3 |
| w_o_reward_shaping | 44 | 66.40% | 33.60% | 0.6107 | 1.4943 | 2.7229 | 0.0463 | 0.0697 | 100.00% | 0.1400 | 58.93 | 3 |
| w_o_semantic_prior | 42 | 49.00% | 51.00% | 0.6988 | 1.8229 | 3.4391 | 0.0749 | 0.1528 | 100.00% | 0.1354 | 39.89 | 2 |
| w_o_semantic_prior | 43 | 66.20% | 33.80% | 0.6108 | 1.4938 | 2.7335 | 0.0463 | 0.0699 | 100.00% | 0.1381 | 58.73 | 3 |
| w_o_semantic_prior | 44 | 49.00% | 51.00% | 0.6988 | 1.8229 | 3.4391 | 0.0748 | 0.1526 | 100.00% | 0.1349 | 39.89 | 2 |
| w_o_confidence | 42 | 24.20% | 75.80% | 0.7844 | 1.9463 | 3.5941 | 0.0177 | 0.0732 | 0.00% | 0.1344 | 14.47 | 4 |
| w_o_confidence | 43 | 66.40% | 33.60% | 0.6107 | 1.4935 | 2.7295 | 0.0463 | 0.0697 | 100.00% | 0.1365 | 58.93 | 3 |
| w_o_confidence | 44 | 24.00% | 76.00% | 0.7844 | 1.9466 | 3.6022 | 0.0177 | 0.0739 | 0.00% | 0.1459 | 14.27 | 4 |
| w_o_partial_offloading | 42 | 24.20% | 75.80% | 0.7844 | 1.9448 | 3.5917 | 0.0177 | 0.0732 | 0.00% | 0.1959 | 14.48 | 2 |
| w_o_partial_offloading | 43 | 24.20% | 75.80% | 0.7845 | 1.9470 | 3.5933 | 0.0178 | 0.0734 | 0.00% | 0.1338 | 14.46 | 3 |
| w_o_partial_offloading | 44 | 24.20% | 75.80% | 0.7844 | 1.9471 | 3.6049 | 0.0177 | 0.0733 | 0.00% | 0.1461 | 14.46 | 2 |
| w_o_battery_awareness | 42 | 66.40% | 33.60% | 0.6107 | 1.4931 | 2.7234 | 0.0463 | 0.0697 | 100.00% | 0.1352 | 58.93 | 3 |
| w_o_battery_awareness | 43 | 66.20% | 33.80% | 0.6106 | 1.4936 | 2.7287 | 0.0462 | 0.0698 | 100.00% | 0.1339 | 58.73 | 3 |
| w_o_battery_awareness | 44 | 49.00% | 51.00% | 0.6988 | 1.8229 | 3.4391 | 0.0748 | 0.1526 | 100.00% | 0.1360 | 39.89 | 2 |
| w_o_queue_awareness | 42 | 49.00% | 51.00% | 0.6988 | 1.8229 | 3.4391 | 0.0749 | 0.1528 | 100.00% | 0.1365 | 39.89 | 2 |
| w_o_queue_awareness | 43 | 49.00% | 51.00% | 0.6988 | 1.8229 | 3.4391 | 0.0748 | 0.1527 | 100.00% | 0.1366 | 39.89 | 2 |
| w_o_queue_awareness | 44 | 49.00% | 51.00% | 0.6988 | 1.8229 | 3.4391 | 0.0749 | 0.1528 | 100.00% | 0.1370 | 39.89 | 2 |
| w_o_mobility_features | 42 | 66.20% | 33.80% | 0.6108 | 1.4938 | 2.7247 | 0.0463 | 0.0699 | 100.00% | 0.1348 | 58.73 | 3 |
| w_o_mobility_features | 43 | 24.00% | 76.00% | 0.7846 | 1.9466 | 3.6006 | 0.0178 | 0.0743 | 0.00% | 0.1359 | 14.27 | 4 |
| w_o_mobility_features | 44 | 66.40% | 33.60% | 0.6108 | 1.4938 | 2.7238 | 0.0463 | 0.0697 | 100.00% | 0.1374 | 58.93 | 3 |

### dqn_multi_seed_retraining Action-Collapse Diagnostic

- `full_model` seed `42` tek aksiyona cokuyor: action `2` = 100.00%.
- `full_model` seed `43` tek aksiyona cokuyor: action `4` = 100.00%.
- `full_model` seed `44` tek aksiyona cokuyor: action `2` = 100.00%.
- `w_o_semantics` seed `42` tek aksiyona cokuyor: action `3` = 100.00%.
- `w_o_semantics` seed `43` tek aksiyona cokuyor: action `1` = 100.00%.
- `w_o_semantics` seed `44` tek aksiyona cokuyor: action `3` = 100.00%.
- `w_o_reward_shaping` seed `42` tek aksiyona cokuyor: action `3` = 100.00%.
- `w_o_reward_shaping` seed `43` tek aksiyona cokuyor: action `3` = 100.00%.
- `w_o_reward_shaping` seed `44` tek aksiyona cokuyor: action `3` = 100.00%.
- `w_o_semantic_prior` seed `42` tek aksiyona cokuyor: action `2` = 100.00%.
- `w_o_semantic_prior` seed `43` tek aksiyona cokuyor: action `3` = 100.00%.
- `w_o_semantic_prior` seed `44` tek aksiyona cokuyor: action `2` = 100.00%.
- `w_o_confidence` seed `42` tek aksiyona cokuyor: action `4` = 100.00%.
- `w_o_confidence` seed `43` tek aksiyona cokuyor: action `3` = 100.00%.
- `w_o_confidence` seed `44` tek aksiyona cokuyor: action `4` = 100.00%.
- `w_o_partial_offloading` seed `42` tek aksiyona cokuyor: action `2` = 100.00%.
- `w_o_partial_offloading` seed `43` tek aksiyona cokuyor: action `3` = 100.00%.
- `w_o_partial_offloading` seed `44` tek aksiyona cokuyor: action `2` = 100.00%.
- `w_o_battery_awareness` seed `42` tek aksiyona cokuyor: action `3` = 100.00%.
- `w_o_battery_awareness` seed `43` tek aksiyona cokuyor: action `3` = 100.00%.
- `w_o_battery_awareness` seed `44` tek aksiyona cokuyor: action `2` = 100.00%.
- `w_o_queue_awareness` seed `42` tek aksiyona cokuyor: action `2` = 100.00%.
- `w_o_queue_awareness` seed `43` tek aksiyona cokuyor: action `2` = 100.00%.
- `w_o_queue_awareness` seed `44` tek aksiyona cokuyor: action `2` = 100.00%.
- `w_o_mobility_features` seed `42` tek aksiyona cokuyor: action `3` = 100.00%.
- `w_o_mobility_features` seed `43` tek aksiyona cokuyor: action `4` = 100.00%.
- `w_o_mobility_features` seed `44` tek aksiyona cokuyor: action `3` = 100.00%.

### ppo_multi_seed_evaluation

| Variant | Seed | Success Rate | Miss Ratio | Avg Latency | P95 | P99 | Avg Energy | Energy/Success | Partial Ratio | Overhead ms | QoE | Dominant Action |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full_model | 42 | 66.40% | 33.60% | 0.6108 | 1.4939 | 2.7324 | 0.0463 | 0.0698 | 100.00% | 0.2857 | 58.93 | 3 |
| full_model | 43 | 66.40% | 33.60% | 0.6107 | 1.4935 | 2.7254 | 0.0463 | 0.0697 | 100.00% | 0.2591 | 58.93 | 3 |
| full_model | 44 | 66.40% | 33.60% | 0.6107 | 1.4937 | 2.7287 | 0.0462 | 0.0696 | 100.00% | 0.2679 | 58.93 | 3 |
| w_o_semantics | 42 | 66.20% | 33.80% | 0.6107 | 1.4933 | 2.7326 | 0.0463 | 0.0699 | 100.00% | 0.1584 | 58.73 | 3 |
| w_o_semantics | 43 | 66.40% | 33.60% | 0.6108 | 1.4937 | 2.7230 | 0.0463 | 0.0697 | 100.00% | 0.2201 | 58.93 | 3 |
| w_o_semantics | 44 | 66.20% | 33.80% | 0.6108 | 1.4934 | 2.7327 | 0.0463 | 0.0699 | 100.00% | 0.1965 | 58.73 | 3 |
| w_o_reward_shaping | 42 | 66.40% | 33.60% | 0.6108 | 1.4939 | 2.7299 | 0.0463 | 0.0698 | 100.00% | 0.2496 | 58.93 | 3 |
| w_o_reward_shaping | 43 | 18.00% | 82.00% | 0.8708 | 1.4484 | 2.1403 | 0.0179 | 0.0993 | 10.00% | 0.3049 | 10.76 | 5 |
| w_o_reward_shaping | 44 | 49.00% | 51.00% | 0.6988 | 1.8229 | 3.4391 | 0.0749 | 0.1528 | 100.00% | 0.2851 | 39.89 | 2 |
| w_o_semantic_prior | 42 | 66.60% | 33.40% | 0.6107 | 1.4942 | 2.7231 | 0.0463 | 0.0694 | 100.00% | 0.2644 | 59.13 | 3 |
| w_o_semantic_prior | 43 | 66.40% | 33.60% | 0.6108 | 1.4933 | 2.7270 | 0.0463 | 0.0697 | 100.00% | 0.2980 | 58.93 | 3 |
| w_o_semantic_prior | 44 | 66.20% | 33.80% | 0.6108 | 1.4942 | 2.7322 | 0.0463 | 0.0700 | 100.00% | 0.2870 | 58.73 | 3 |
| w_o_confidence | 42 | 66.20% | 33.80% | 0.6107 | 1.4939 | 2.7221 | 0.0462 | 0.0699 | 100.00% | 0.2994 | 58.73 | 3 |
| w_o_confidence | 43 | 66.20% | 33.80% | 0.6107 | 1.4942 | 2.7276 | 0.0462 | 0.0699 | 100.00% | 0.2392 | 58.73 | 3 |
| w_o_confidence | 44 | 66.40% | 33.60% | 0.6107 | 1.4932 | 2.7223 | 0.0462 | 0.0696 | 100.00% | 0.2472 | 58.93 | 3 |
| w_o_partial_offloading | 42 | 24.20% | 75.80% | 0.7843 | 1.9463 | 3.5897 | 0.0177 | 0.0731 | 0.00% | 0.2303 | 14.47 | 3 |
| w_o_partial_offloading | 43 | 24.00% | 76.00% | 0.7844 | 1.9463 | 3.5996 | 0.0177 | 0.0739 | 0.00% | 0.2541 | 14.27 | 2 |
| w_o_partial_offloading | 44 | 24.00% | 76.00% | 0.7845 | 1.9464 | 3.5925 | 0.0178 | 0.0740 | 0.00% | 0.2555 | 14.27 | 2 |
| w_o_battery_awareness | 42 | 66.40% | 33.60% | 0.6105 | 1.4940 | 2.7262 | 0.0462 | 0.0695 | 100.00% | 0.2756 | 58.93 | 3 |
| w_o_battery_awareness | 43 | 66.20% | 33.80% | 0.6108 | 1.4939 | 2.7253 | 0.0463 | 0.0699 | 100.00% | 0.2441 | 58.73 | 3 |
| w_o_battery_awareness | 44 | 66.40% | 33.60% | 0.6108 | 1.4943 | 2.7330 | 0.0463 | 0.0697 | 100.00% | 0.2346 | 58.93 | 3 |
| w_o_queue_awareness | 42 | 66.20% | 33.80% | 0.6109 | 1.4933 | 2.7309 | 0.0464 | 0.0701 | 100.00% | 0.2462 | 58.73 | 3 |
| w_o_queue_awareness | 43 | 66.20% | 33.80% | 0.6106 | 1.4933 | 2.7333 | 0.0462 | 0.0698 | 100.00% | 0.2490 | 58.73 | 3 |
| w_o_queue_awareness | 44 | 66.20% | 33.80% | 0.6107 | 1.4940 | 2.7281 | 0.0463 | 0.0699 | 100.00% | 0.3062 | 58.73 | 3 |
| w_o_mobility_features | 42 | 66.20% | 33.80% | 0.6108 | 1.4938 | 2.7331 | 0.0463 | 0.0699 | 100.00% | 0.2634 | 58.73 | 3 |
| w_o_mobility_features | 43 | 66.20% | 33.80% | 0.6108 | 1.4943 | 2.7325 | 0.0463 | 0.0699 | 100.00% | 0.2657 | 58.73 | 3 |
| w_o_mobility_features | 44 | 66.40% | 33.60% | 0.6107 | 1.4931 | 2.7288 | 0.0462 | 0.0697 | 100.00% | 0.2340 | 58.93 | 3 |

### ppo_multi_seed_evaluation Action-Collapse Diagnostic

- `full_model` seed `42` tek aksiyona cokuyor: action `3` = 100.00%.
- `full_model` seed `43` tek aksiyona cokuyor: action `3` = 100.00%.
- `full_model` seed `44` tek aksiyona cokuyor: action `3` = 100.00%.
- `w_o_semantics` seed `42` tek aksiyona cokuyor: action `3` = 100.00%.
- `w_o_semantics` seed `43` tek aksiyona cokuyor: action `3` = 100.00%.
- `w_o_semantics` seed `44` tek aksiyona cokuyor: action `3` = 100.00%.
- `w_o_reward_shaping` seed `42` tek aksiyona cokuyor: action `3` = 100.00%.
- `w_o_reward_shaping` seed `43` tek aksiyona cokuyor: action `5` = 90.00%.
- `w_o_reward_shaping` seed `44` tek aksiyona cokuyor: action `2` = 100.00%.
- `w_o_semantic_prior` seed `42` tek aksiyona cokuyor: action `3` = 100.00%.
- `w_o_semantic_prior` seed `43` tek aksiyona cokuyor: action `3` = 100.00%.
- `w_o_semantic_prior` seed `44` tek aksiyona cokuyor: action `3` = 100.00%.
- `w_o_confidence` seed `42` tek aksiyona cokuyor: action `3` = 100.00%.
- `w_o_confidence` seed `43` tek aksiyona cokuyor: action `3` = 100.00%.
- `w_o_confidence` seed `44` tek aksiyona cokuyor: action `3` = 100.00%.
- `w_o_partial_offloading` seed `42` tek aksiyona cokuyor: action `3` = 100.00%.
- `w_o_partial_offloading` seed `43` tek aksiyona cokuyor: action `2` = 100.00%.
- `w_o_partial_offloading` seed `44` tek aksiyona cokuyor: action `2` = 100.00%.
- `w_o_battery_awareness` seed `42` tek aksiyona cokuyor: action `3` = 100.00%.
- `w_o_battery_awareness` seed `43` tek aksiyona cokuyor: action `3` = 100.00%.
- `w_o_battery_awareness` seed `44` tek aksiyona cokuyor: action `3` = 100.00%.
- `w_o_queue_awareness` seed `42` tek aksiyona cokuyor: action `3` = 100.00%.
- `w_o_queue_awareness` seed `43` tek aksiyona cokuyor: action `3` = 100.00%.
- `w_o_queue_awareness` seed `44` tek aksiyona cokuyor: action `3` = 100.00%.
- `w_o_mobility_features` seed `42` tek aksiyona cokuyor: action `3` = 100.00%.
- `w_o_mobility_features` seed `43` tek aksiyona cokuyor: action `3` = 100.00%.
- `w_o_mobility_features` seed `44` tek aksiyona cokuyor: action `3` = 100.00%.

### ppo_multi_seed_retraining

| Variant | Seed | Success Rate | Miss Ratio | Avg Latency | P95 | P99 | Avg Energy | Energy/Success | Partial Ratio | Overhead ms | QoE | Dominant Action |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full_model | 42 | 66.20% | 33.80% | 0.6107 | 1.4941 | 2.7323 | 0.0463 | 0.0699 | 100.00% | 0.2643 | 58.73 | 3 |
| full_model | 43 | 66.20% | 33.80% | 0.6107 | 1.4941 | 2.7237 | 0.0463 | 0.0699 | 100.00% | 0.2472 | 58.73 | 3 |
| full_model | 44 | 66.20% | 33.80% | 0.6108 | 1.4938 | 2.7334 | 0.0463 | 0.0699 | 100.00% | 0.2264 | 58.73 | 3 |
| w_o_semantics | 42 | 66.20% | 33.80% | 0.6107 | 1.4944 | 2.7295 | 0.0462 | 0.0699 | 100.00% | 0.1921 | 58.73 | 3 |
| w_o_semantics | 43 | 66.20% | 33.80% | 0.6108 | 1.4933 | 2.7249 | 0.0463 | 0.0700 | 100.00% | 0.1817 | 58.73 | 3 |
| w_o_semantics | 44 | 66.20% | 33.80% | 0.6107 | 1.4935 | 2.7242 | 0.0462 | 0.0698 | 100.00% | 0.2396 | 58.73 | 3 |
| w_o_reward_shaping | 42 | 66.40% | 33.60% | 0.6108 | 1.4937 | 2.7333 | 0.0463 | 0.0697 | 100.00% | 0.2503 | 58.93 | 3 |
| w_o_reward_shaping | 43 | 18.00% | 82.00% | 0.8707 | 1.4477 | 2.1395 | 0.0178 | 0.0991 | 10.00% | 0.2648 | 10.76 | 5 |
| w_o_reward_shaping | 44 | 49.00% | 51.00% | 0.6988 | 1.8229 | 3.4391 | 0.0748 | 0.1527 | 100.00% | 0.2524 | 39.89 | 2 |
| w_o_semantic_prior | 42 | 66.20% | 33.80% | 0.6108 | 1.4941 | 2.7292 | 0.0463 | 0.0699 | 100.00% | 0.2725 | 58.73 | 3 |
| w_o_semantic_prior | 43 | 66.40% | 33.60% | 0.6107 | 1.4936 | 2.7229 | 0.0463 | 0.0697 | 100.00% | 0.2831 | 58.93 | 3 |
| w_o_semantic_prior | 44 | 66.40% | 33.60% | 0.6106 | 1.4935 | 2.7320 | 0.0462 | 0.0696 | 100.00% | 0.2514 | 58.93 | 3 |
| w_o_confidence | 42 | 66.40% | 33.60% | 0.6107 | 1.4934 | 2.7335 | 0.0463 | 0.0697 | 100.00% | 0.2651 | 58.93 | 3 |
| w_o_confidence | 43 | 66.20% | 33.80% | 0.6107 | 1.4941 | 2.7276 | 0.0463 | 0.0699 | 100.00% | 0.2571 | 58.73 | 3 |
| w_o_confidence | 44 | 66.20% | 33.80% | 0.6108 | 1.4942 | 2.7224 | 0.0463 | 0.0700 | 100.00% | 0.2592 | 58.73 | 3 |
| w_o_partial_offloading | 42 | 24.20% | 75.80% | 0.7843 | 1.9451 | 3.6013 | 0.0177 | 0.0731 | 0.00% | 0.2461 | 14.47 | 3 |
| w_o_partial_offloading | 43 | 24.00% | 76.00% | 0.7842 | 1.9453 | 3.5930 | 0.0177 | 0.0735 | 0.00% | 0.2711 | 14.27 | 2 |
| w_o_partial_offloading | 44 | 24.00% | 76.00% | 0.7845 | 1.9457 | 3.5899 | 0.0178 | 0.0740 | 0.00% | 0.2626 | 14.27 | 2 |
| w_o_battery_awareness | 42 | 66.20% | 33.80% | 0.6108 | 1.4935 | 2.7233 | 0.0463 | 0.0699 | 100.00% | 0.2539 | 58.73 | 3 |
| w_o_battery_awareness | 43 | 66.20% | 33.80% | 0.6108 | 1.4939 | 2.7315 | 0.0463 | 0.0699 | 100.00% | 0.2696 | 58.73 | 3 |
| w_o_battery_awareness | 44 | 66.40% | 33.60% | 0.6106 | 1.4940 | 2.7305 | 0.0462 | 0.0696 | 100.00% | 0.2604 | 58.93 | 3 |
| w_o_queue_awareness | 42 | 66.40% | 33.60% | 0.6108 | 1.4937 | 2.7315 | 0.0463 | 0.0697 | 100.00% | 0.2545 | 58.93 | 3 |
| w_o_queue_awareness | 43 | 66.20% | 33.80% | 0.6108 | 1.4942 | 2.7279 | 0.0463 | 0.0699 | 100.00% | 0.2927 | 58.73 | 3 |
| w_o_queue_awareness | 44 | 66.20% | 33.80% | 0.6108 | 1.4941 | 2.7232 | 0.0463 | 0.0699 | 100.00% | 0.2521 | 58.73 | 3 |
| w_o_mobility_features | 42 | 66.60% | 33.40% | 0.6108 | 1.4937 | 2.7274 | 0.0463 | 0.0696 | 100.00% | 0.2788 | 59.13 | 3 |
| w_o_mobility_features | 43 | 66.20% | 33.80% | 0.6108 | 1.4934 | 2.7298 | 0.0463 | 0.0700 | 100.00% | 0.2737 | 58.73 | 3 |
| w_o_mobility_features | 44 | 66.20% | 33.80% | 0.6107 | 1.4940 | 2.7279 | 0.0462 | 0.0699 | 100.00% | 0.2878 | 58.73 | 3 |

### ppo_multi_seed_retraining Action-Collapse Diagnostic

- `full_model` seed `42` tek aksiyona cokuyor: action `3` = 100.00%.
- `full_model` seed `43` tek aksiyona cokuyor: action `3` = 100.00%.
- `full_model` seed `44` tek aksiyona cokuyor: action `3` = 100.00%.
- `w_o_semantics` seed `42` tek aksiyona cokuyor: action `3` = 100.00%.
- `w_o_semantics` seed `43` tek aksiyona cokuyor: action `3` = 100.00%.
- `w_o_semantics` seed `44` tek aksiyona cokuyor: action `3` = 100.00%.
- `w_o_reward_shaping` seed `42` tek aksiyona cokuyor: action `3` = 100.00%.
- `w_o_reward_shaping` seed `43` tek aksiyona cokuyor: action `5` = 90.00%.
- `w_o_reward_shaping` seed `44` tek aksiyona cokuyor: action `2` = 100.00%.
- `w_o_semantic_prior` seed `42` tek aksiyona cokuyor: action `3` = 100.00%.
- `w_o_semantic_prior` seed `43` tek aksiyona cokuyor: action `3` = 100.00%.
- `w_o_semantic_prior` seed `44` tek aksiyona cokuyor: action `3` = 100.00%.
- `w_o_confidence` seed `42` tek aksiyona cokuyor: action `3` = 100.00%.
- `w_o_confidence` seed `43` tek aksiyona cokuyor: action `3` = 100.00%.
- `w_o_confidence` seed `44` tek aksiyona cokuyor: action `3` = 100.00%.
- `w_o_partial_offloading` seed `42` tek aksiyona cokuyor: action `3` = 100.00%.
- `w_o_partial_offloading` seed `43` tek aksiyona cokuyor: action `2` = 100.00%.
- `w_o_partial_offloading` seed `44` tek aksiyona cokuyor: action `2` = 100.00%.
- `w_o_battery_awareness` seed `42` tek aksiyona cokuyor: action `3` = 100.00%.
- `w_o_battery_awareness` seed `43` tek aksiyona cokuyor: action `3` = 100.00%.
- `w_o_battery_awareness` seed `44` tek aksiyona cokuyor: action `3` = 100.00%.
- `w_o_queue_awareness` seed `42` tek aksiyona cokuyor: action `3` = 100.00%.
- `w_o_queue_awareness` seed `43` tek aksiyona cokuyor: action `3` = 100.00%.
- `w_o_queue_awareness` seed `44` tek aksiyona cokuyor: action `3` = 100.00%.
- `w_o_mobility_features` seed `42` tek aksiyona cokuyor: action `3` = 100.00%.
- `w_o_mobility_features` seed `43` tek aksiyona cokuyor: action `3` = 100.00%.
- `w_o_mobility_features` seed `44` tek aksiyona cokuyor: action `3` = 100.00%.

