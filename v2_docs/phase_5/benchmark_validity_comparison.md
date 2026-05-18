# Phase 5R Benchmark Validity Comparison

Bu rapor, real-world trace-driven hybrid benchmark varyantlarini karsilastirir. Amac model kosularindan once benchmark kapisinin gecilip gecilmedigini belirlemektir.

| Benchmark | Status | Oracle Ceiling | Impossible | Dominant Oracle Action | Dominant Share | Diversity | Local Feas. | Edge25 | Edge50 | Edge75 | Edge100 | Cloud |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| random | needs_revision | 94.80% | 5.20% | cloud | 89.66% | 2 | 0.40% | 4.20% | 23.60% | 58.60% | 23.60% | 92.60% |
| correlation_aware_cloud035 | pass | 88.00% | 12.00% | edge_75 | 53.86% | 3 | 0.00% | 2.40% | 46.60% | 85.80% | 39.80% | 66.60% |
| service_class_mec_mixed | pass | 91.80% | 8.20% | edge_75 | 70.15% | 3 | 21.00% | 23.60% | 52.20% | 88.00% | 49.40% | 39.20% |

## Correlation Checks

| Benchmark | Diff-Server Corr | Arrival-Mobility Corr | Arrival-Server Corr |
|---|---:|---:|---:|
| random | 0.009 | 0.000 | 0.012 |
| correlation_aware_cloud035 | 0.443 | 0.952 | 0.807 |
| service_class_mec_mixed | 0.443 | 0.952 | 0.807 |

## Decision

- `random` build korunacak ama yalnizca independent-pairing baseline olarak okunacak.
- `correlation_aware_cloud035` dogal korelasyon ve cloud dominance sorununu iyilestirdi, fakat local/edge25/full-edge coverage halen zayifti.
- `service_class_mec_mixed` Faz 5R icin yeni ana benchmark adayidir: oracle ceiling `%91.80`, impossible task `%8.20`, local feasible `%21.00`, edge25 `%23.60`, edge50 `%52.20`, edge75 `%88.00`, edge100 `%49.40`, cloud `%39.20`.

## Next Gate

Bundan sonra model kosulari once `configs/phase_5/real_data_rl_training_service_class.yaml` uzerinden heuristic-only sanity check ile baslamali. PPO/DQN/A2C veya ablation retraining ancak bu benchmark uzerinde policy sonuclari tekrar okunduktan sonra calistirilmelidir.
