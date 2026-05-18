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
- `v2_docs/phase_5/real_data_oracle_audit.md`

## RL Retraining

| Algorithm | Seed | Success Rate | Miss Ratio | Avg Latency | P95 | P99 | Avg Energy | Energy/Success | Partial Ratio | Overhead ms | QoE | Dominant Action |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PPO | 42 | 42.00% | 0.00% | 0.0000 | 2.0395 | 0.0000 | 0.0594 | 0.0000 | 0.00% | 0.0000 | 31.80 | 3 |
| PPO | 43 | 42.00% | 0.00% | 0.0000 | 2.0393 | 0.0000 | 0.0594 | 0.0000 | 0.00% | 0.0000 | 31.80 | 3 |
| PPO | 44 | 42.00% | 0.00% | 0.0000 | 2.0397 | 0.0000 | 0.0594 | 0.0000 | 0.00% | 0.0000 | 31.80 | 3 |

### RL Retraining Action-Collapse Diagnostic

- `PPO` seed `42` tek aksiyona cokuyor: action `3` = 100.00%.
- `PPO` seed `43` tek aksiyona cokuyor: action `3` = 100.00%.
- `PPO` seed `44` tek aksiyona cokuyor: action `3` = 100.00%.

## Policy Evaluation

| Policy | Seed | Success Rate | Miss Ratio | Avg Latency | P95 | P99 | Avg Energy | Energy/Success | Partial Ratio | Overhead ms | QoE | Dominant Action |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LocalOnly | 42 | 0.40% | 99.60% | 2.1285 | 5.3009 | 6.2949 | 0.2128 | 53.2120 | 0.00% | 0.0004 | -26.10 | 0 |
| EdgeOnly | 42 | 12.20% | 87.80% | 1.1669 | 2.7752 | 3.2384 | 0.0083 | 0.0680 | 0.00% | 0.0004 | -1.68 | 4 |
| CloudOnly | 42 | 38.80% | 61.20% | 0.8447 | 1.5101 | 1.7195 | 0.0083 | 0.0213 | 0.00% | 0.0003 | 31.25 | 5 |
| Random | 42 | 27.20% | 72.80% | 1.2788 | 3.9216 | 5.2528 | 0.0986 | 0.3623 | 51.40% | 0.0012 | 7.59 | 1 |
| GreedyLatency | 42 | 42.40% | 57.60% | 0.8105 | 1.4905 | 1.7066 | 0.0110 | 0.0259 | 0.00% | 0.0029 | 34.95 | 5 |
| GeneticAlgorithm | 42 | 49.20% | 50.80% | 0.7840 | 1.4897 | 1.7515 | 0.0130 | 0.0265 | 12.60% | 0.1114 | 41.75 | 5 |
| PPO | 42 | 42.00% | 58.00% | 0.8959 | 2.1059 | 2.4580 | 0.0594 | 0.1414 | 100.00% | 0.2982 | 31.47 | 3 |
| DQN | 42 | 38.80% | 61.20% | 0.8447 | 1.5100 | 1.7180 | 0.0083 | 0.0214 | 0.00% | 0.1438 | 31.25 | 5 |
| A2C | 42 | 38.80% | 61.20% | 0.8447 | 1.5105 | 1.7192 | 0.0083 | 0.0214 | 0.00% | 0.2417 | 31.25 | 5 |
| LocalOnly | 43 | 0.40% | 99.60% | 2.1285 | 5.3009 | 6.2949 | 0.2128 | 53.2120 | 0.00% | 0.0003 | -26.10 | 0 |
| EdgeOnly | 43 | 12.20% | 87.80% | 1.1668 | 2.7755 | 3.2386 | 0.0083 | 0.0677 | 0.00% | 0.0003 | -1.68 | 4 |
| CloudOnly | 43 | 38.80% | 61.20% | 0.8446 | 1.5104 | 1.7181 | 0.0083 | 0.0213 | 0.00% | 0.0003 | 31.25 | 5 |
| Random | 43 | 26.60% | 73.40% | 1.2174 | 3.3808 | 5.1699 | 0.0884 | 0.3323 | 51.00% | 0.0011 | 9.70 | 1 |
| GreedyLatency | 43 | 42.40% | 57.60% | 0.8106 | 1.4895 | 1.7071 | 0.0110 | 0.0260 | 0.00% | 0.0028 | 34.95 | 5 |
| GeneticAlgorithm | 43 | 48.60% | 51.40% | 0.7845 | 1.4931 | 1.7076 | 0.0130 | 0.0267 | 13.20% | 0.1111 | 41.13 | 5 |
| PPO | 43 | 42.00% | 58.00% | 0.8960 | 2.1060 | 2.4580 | 0.0594 | 0.1415 | 100.00% | 0.2333 | 31.47 | 3 |
| DQN | 43 | 38.80% | 61.20% | 0.8447 | 1.5100 | 1.7194 | 0.0083 | 0.0213 | 0.00% | 0.1375 | 31.25 | 5 |
| A2C | 43 | 38.80% | 61.20% | 0.8446 | 1.5102 | 1.7197 | 0.0083 | 0.0213 | 0.00% | 0.2326 | 31.25 | 5 |
| LocalOnly | 44 | 0.40% | 99.60% | 2.1285 | 5.3009 | 6.2949 | 0.2128 | 53.2120 | 0.00% | 0.0003 | -26.10 | 0 |
| EdgeOnly | 44 | 12.20% | 87.80% | 1.1668 | 2.7753 | 3.2386 | 0.0083 | 0.0678 | 0.00% | 0.0004 | -1.68 | 4 |
| CloudOnly | 44 | 38.80% | 61.20% | 0.8447 | 1.5104 | 1.7193 | 0.0083 | 0.0213 | 0.00% | 0.0003 | 31.25 | 5 |
| Random | 44 | 28.00% | 72.00% | 1.2716 | 3.9216 | 5.3727 | 0.0981 | 0.3504 | 55.60% | 0.0012 | 8.39 | 2 |
| GreedyLatency | 44 | 42.40% | 57.60% | 0.8106 | 1.4893 | 1.7071 | 0.0110 | 0.0260 | 0.00% | 0.0029 | 34.95 | 5 |
| GeneticAlgorithm | 44 | 48.80% | 51.20% | 0.7867 | 1.5049 | 1.7503 | 0.0132 | 0.0271 | 12.80% | 0.1140 | 41.28 | 5 |
| PPO | 44 | 42.00% | 58.00% | 0.8960 | 2.1057 | 2.4582 | 0.0594 | 0.1414 | 100.00% | 0.2406 | 31.47 | 3 |
| DQN | 44 | 38.80% | 61.20% | 0.8447 | 1.5099 | 1.7196 | 0.0083 | 0.0213 | 0.00% | 0.1444 | 31.25 | 5 |
| A2C | 44 | 38.80% | 61.20% | 0.8446 | 1.5102 | 1.7192 | 0.0083 | 0.0213 | 0.00% | 0.2395 | 31.25 | 5 |

### Policy Evaluation Action-Collapse Diagnostic

- `LocalOnly` seed `42` tek aksiyona cokuyor: action `0` = 100.00%.
- `EdgeOnly` seed `42` tek aksiyona cokuyor: action `4` = 100.00%.
- `CloudOnly` seed `42` tek aksiyona cokuyor: action `5` = 100.00%.
- `GreedyLatency` seed `42` tek aksiyona cokuyor: action `5` = 90.20%.
- `PPO` seed `42` tek aksiyona cokuyor: action `3` = 100.00%.
- `DQN` seed `42` tek aksiyona cokuyor: action `5` = 100.00%.
- `A2C` seed `42` tek aksiyona cokuyor: action `5` = 100.00%.
- `LocalOnly` seed `43` tek aksiyona cokuyor: action `0` = 100.00%.
- `EdgeOnly` seed `43` tek aksiyona cokuyor: action `4` = 100.00%.
- `CloudOnly` seed `43` tek aksiyona cokuyor: action `5` = 100.00%.
- `GreedyLatency` seed `43` tek aksiyona cokuyor: action `5` = 90.20%.
- `PPO` seed `43` tek aksiyona cokuyor: action `3` = 100.00%.
- `DQN` seed `43` tek aksiyona cokuyor: action `5` = 100.00%.
- `A2C` seed `43` tek aksiyona cokuyor: action `5` = 100.00%.
- `LocalOnly` seed `44` tek aksiyona cokuyor: action `0` = 100.00%.
- `EdgeOnly` seed `44` tek aksiyona cokuyor: action `4` = 100.00%.
- `CloudOnly` seed `44` tek aksiyona cokuyor: action `5` = 100.00%.
- `GreedyLatency` seed `44` tek aksiyona cokuyor: action `5` = 90.20%.
- `PPO` seed `44` tek aksiyona cokuyor: action `3` = 100.00%.
- `DQN` seed `44` tek aksiyona cokuyor: action `5` = 100.00%.
- `A2C` seed `44` tek aksiyona cokuyor: action `5` = 100.00%.

## Ablation

Henuz ablation sonucu uretilmedi.

