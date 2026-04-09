Bkz. ortak kavram sozlugu: v2_docs/project_concepts_glossary.md

# Synthetic Oracle Label Summary

Bu rapor, Faz 7 icin uretilen oracle label dataset'inin ilk ozetini verir.
Dataset, supervised pretraining oncesi observation -> action etiketi ureten ogretmen mekanizmanin cikisidir.

## Uretim Ayarlari

- Seed: `42`
- Episode sayisi: `60`
- Objective'ler: `latency_oracle, energy_oracle, weighted_objective_oracle, reward_aligned_oracle`

## Split Dagilimi

| Split | Sample Count |
|---|---:|
| train | 8400 |
| val | 1800 |
| test | 1800 |

## Objective Bazli Action Dagilimi

### latency_oracle

| Action | Count | Ratio |
|---|---:|---:|
| cloud | 2925 | 97.50% |
| edge_100 | 39 | 1.30% |
| edge_50 | 13 | 0.43% |
| edge_75 | 17 | 0.57% |
| local | 6 | 0.20% |

### energy_oracle

| Action | Count | Ratio |
|---|---:|---:|
| cloud | 2931 | 97.70% |
| edge_100 | 39 | 1.30% |
| edge_50 | 13 | 0.43% |
| edge_75 | 8 | 0.27% |
| local | 9 | 0.30% |

### weighted_objective_oracle

| Action | Count | Ratio |
|---|---:|---:|
| cloud | 1181 | 39.37% |
| edge_25 | 7 | 0.23% |
| edge_50 | 7 | 0.23% |
| edge_75 | 1787 | 59.57% |
| local | 18 | 0.60% |

### reward_aligned_oracle

| Action | Count | Ratio |
|---|---:|---:|
| cloud | 1017 | 33.90% |
| edge_100 | 3 | 0.10% |
| edge_50 | 21 | 0.70% |
| edge_75 | 1905 | 63.50% |
| local | 54 | 1.80% |

