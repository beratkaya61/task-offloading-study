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
| cloud | 39.20% | 0.649 |

## Priority / Proxy Summary

| Priority | Tasks | CPU Mean | Size Mean | Deadline Mean | Difficulty Mean | Server Load Mean |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1658 | 1451610629.622 | 957.809 | 1.121 | 0.386 | 0.556 |
| 1 | 451 | 841812480.614 | 794.623 | 0.500 | 0.456 | 0.546 |
| 2 | 342 | 853260933.965 | 756.871 | 0.437 | 0.469 | 0.546 |
| 3 | 2549 | 1642417503.281 | 2145.690 | 0.746 | 0.586 | 0.579 |

## Policy Snapshot

Policy evaluation CSV bulunmadigi icin model snapshot okunmadi.

## Interpretation

- `needs_revision` sonucu model basarisizligi degil, benchmark-policy hizalama kapisinin gecilmedigi anlamina gelir.
- Cloud dominance yuksekse benchmark zengin offloading karar probleminden cok cloud-agirlikli probleme donusmus olabilir.
- Workload-server-location korelasyonlari zayifsa hibrit pairing dogal korelasyonlari korumuyor demektir.
