# Benchmark Validity Diagnostic - service_class_mec_mixed

Bu rapor, hibrit trace-driven benchmark'in dogal korelasyonlari ne kadar korudugunu ve policy sonuclarini yorumlamak icin yeterince dengeli olup olmadigini denetler.

## Gate Result

- fusion mode: `correlation_aware`
- record count: `5000`
- status: `pass`
- warning: yok

## Key Correlations

| Relation | Correlation |
|---|---:|
| difficulty_vs_data_size | 0.826 |
| difficulty_vs_deadline | 0.665 |
| difficulty_vs_execution_time | 0.286 |
| difficulty_vs_server_load | 0.443 |
| arrival_vs_mobility_rank | 0.952 |
| arrival_vs_server_load | 0.807 |
| location_x_vs_server_load | 0.012 |

## Oracle / Action Feasibility

- oracle success ceiling: `91.80%`
- impossible task ratio: `8.20%`
- dominant oracle action: `edge_75` (`70.15%`)
- oracle action diversity: `3`

| Action | Feasible Rate | Avg Delay (s) |
|---|---:|---:|
| local | 21.00% | 1.319 |
| edge_25 | 23.60% | 1.018 |
| edge_50 | 52.20% | 0.688 |
| edge_75 | 88.00% | 0.550 |
| edge_100 | 49.40% | 0.705 |
| cloud | 39.40% | 0.649 |

## Priority / Proxy Summary

| Priority | Tasks | CPU Mean | Size Mean | Deadline Mean | Difficulty Mean | Server Load Mean |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1658 | 1451610629.622 | 957.809 | 1.121 | 0.386 | 0.556 |
| 1 | 451 | 841812480.614 | 794.623 | 0.500 | 0.456 | 0.546 |
| 2 | 342 | 853260933.965 | 756.871 | 0.437 | 0.469 | 0.546 |
| 3 | 2549 | 1642417503.281 | 2145.690 | 0.746 | 0.586 | 0.579 |

## Policy Snapshot

| Model | Success Mean | P95 Mean | Partial Ratio | Cloud Rate |
|---|---:|---:|---:|---:|
| DeadlineAwareGreedy | 76.53% | 1.236 | 68.00% | 12.40% |
| GreedyLatency | 76.27% | 1.234 | 68.93% | 12.07% |
| GeneticAlgorithm | 75.47% | 1.234 | 68.47% | 13.00% |
| A2C | 66.40% | 1.494 | 100.00% | 0.00% |
| PPO | 66.27% | 1.494 | 100.00% | 0.00% |
| DQN | 46.53% | 1.754 | 66.67% | 0.00% |
| Random | 34.33% | 2.284 | 49.80% | 18.33% |
| EdgeOnly | 24.00% | 1.946 | 0.00% | 0.00% |
| LocalOnly | 21.00% | 3.558 | 0.00% | 0.00% |
| CloudOnly | 8.00% | 1.463 | 0.00% | 100.00% |

## Interpretation

- `pass` sonucu benchmark'in Faz 5R policy ve ablation yorumlari icin cozulur-zor bir rejim sundugunu gosterir.
- Oracle ceiling yuksek ama PPO/DQN/A2C dusuk veya tek aksiyona cokuyorsa sorun benchmark'tan once policy hizalama, reward/state temsil veya action-collapse tarafinda aranir.
- Sabit LocalOnly/EdgeOnly/CloudOnly baselines dusuk kalirsa bu beklenen bir kontrol sonucudur; env-aware heuristics ile RL/MLP modelleri asil rekabetci karsilastirma grubudur.
- `needs_revision` sonucu gorulurse bu model basarisizligi degil, benchmark-policy hizalama kapisinin gecilmedigi anlamina gelir.
