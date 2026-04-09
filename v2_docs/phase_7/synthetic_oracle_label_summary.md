Bkz. ortak kavram sozlugu: v2_docs/project_concepts_glossary.md

# Synthetic Oracle Label Summary

Bu rapor, Faz 7 icin uretilen oracle label dataset'inin ilk ozetini verir.
Dataset, supervised pretraining oncesi observation -> action etiketi ureten ogretmen mekanizmanin cikisidir.

## Uretim Ayarlari

- Seed: `42`
- Episode sayisi: `60`
- Teacher policies: `teacher_latency_greedy, teacher_energy_greedy, teacher_balanced_semantic, teacher_contextual_reward_aligned`

## Split Dagilimi

| Split | Sample Count |
|---|---:|
| train | 8400 |
| val | 1800 |
| test | 1800 |

## Teacher Policy Bazli Action Dagilimi

### teacher_latency_greedy

| Action | Count | Ratio |
|---|---:|---:|
| cloud | 1238 | 41.27% |
| edge_100 | 269 | 8.97% |
| edge_25 | 269 | 8.97% |
| edge_50 | 357 | 11.90% |
| edge_75 | 574 | 19.13% |
| local | 293 | 9.77% |

### teacher_energy_greedy

| Action | Count | Ratio |
|---|---:|---:|
| cloud | 1035 | 34.50% |
| edge_100 | 327 | 10.90% |
| edge_25 | 276 | 9.20% |
| edge_50 | 424 | 14.13% |
| edge_75 | 607 | 20.23% |
| local | 331 | 11.03% |

### teacher_balanced_semantic

| Action | Count | Ratio |
|---|---:|---:|
| cloud | 932 | 31.07% |
| edge_100 | 302 | 10.07% |
| edge_25 | 293 | 9.77% |
| edge_50 | 428 | 14.27% |
| edge_75 | 697 | 23.23% |
| local | 348 | 11.60% |

### teacher_contextual_reward_aligned

| Action | Count | Ratio |
|---|---:|---:|
| cloud | 878 | 29.27% |
| edge_100 | 252 | 8.40% |
| edge_25 | 253 | 8.43% |
| edge_50 | 302 | 10.07% |
| edge_75 | 1049 | 34.97% |
| local | 266 | 8.87% |
