Bkz. ortak kavram sozlugu: v2_docs/project_concepts_glossary.md

# Staged Training Comparison Report

- Batch ID: `synthetic_staged_training_20260409_000314`
- Pretrained checkpoint: `models/ppo/pretrained/ppo_reward_aligned_pretrained.zip`
- Comparison: `PPO from scratch` vs `Pretrained + PPO`
- Faz 7.3 ara metrikleri: deadline miss ratio, energy per success, step-to-75% success, best success, success AUC, success 95% CI

## Action Mapping

| Action ID | Meaning |
|---:|---|
| 0 | local |
| 1 | edge_25 |
| 2 | edge_50 |
| 3 | edge_75 |
| 4 | edge_full |
| 5 | cloud |

## Final Karsilastirma

| Model | Success Rate (mean +- std) | Success 95% CI | Deadline Miss Ratio | Avg Reward | P95 Latency | Avg Energy | Energy per Success | QoE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PPO_from_scratch | 83.67% +- 0.76 | +- 0.86% | 16.33% +- 0.76 | 846.49 +- 118.84 | 2.006 +- 0.029 | 0.0140 +- 0.0016 | 0.0168 +- 0.0019 | 73.64 +- 0.90 |
| PPO_pretrained_finetuned | 83.67% +- 0.76 | +- 0.86% | 16.33% +- 0.76 | 846.49 +- 118.84 | 2.006 +- 0.029 | 0.0140 +- 0.0016 | 0.0168 +- 0.0019 | 73.64 +- 0.90 |

## Convergence Ara Metrikleri

| Init Mode | Step to 75% Success | Best Success During Training | Success Curve AUC |
|---|---:|---:|---:|
| pretrained | 10000 +- 8660 | 87.60% +- 1.83 | 0.6710 +- 0.0423 |
| scratch | 18333 +- 2887 | 87.60% +- 1.83 | 0.6443 +- 0.0159 |

## Learning Curve Snapshot

| Init Mode | Training Step | Success Rate | Avg Reward | P95 Latency | Avg Energy | QoE |
|---|---:|---:|---:|---:|---:|---:|
| pretrained | 5000 | 78.40% | 986.85 | 2.527 | 0.0599 | 65.76 |
| pretrained | 10000 | 77.60% | 1049.35 | 2.515 | 0.0544 | 65.03 |
| pretrained | 15000 | 73.60% | 1257.82 | 3.013 | 0.0986 | 58.53 |
| pretrained | 20000 | 85.20% | 875.11 | 1.983 | 0.0168 | 75.28 |
| pretrained | 25000 | 83.60% | 773.24 | 2.027 | 0.0161 | 73.46 |
| pretrained | 30000 | 86.80% | 1045.57 | 1.986 | 0.0170 | 76.87 |
| scratch | 5000 | 68.00% | 1370.03 | 3.517 | 0.1325 | 50.41 |
| scratch | 10000 | 64.80% | 1314.75 | 3.542 | 0.1371 | 47.09 |
| scratch | 15000 | 75.60% | 1258.39 | 3.021 | 0.0995 | 60.49 |
| scratch | 20000 | 85.20% | 875.11 | 1.983 | 0.0168 | 75.28 |
| scratch | 25000 | 83.60% | 773.24 | 2.027 | 0.0161 | 73.46 |
| scratch | 30000 | 86.80% | 1045.57 | 1.986 | 0.0170 | 76.87 |

## Action Profili

| Model | Dominant Action Profile | Local | Edge %25 | Edge %50 | Edge %75 | Full Edge | Full Cloud |
|---|---|---:|---:|---:|---:|---:|---:|
| PPO_from_scratch | Full Cloud (100.0%) | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 100.00% |
| PPO_pretrained_finetuned | Full Cloud (100.0%) | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 100.00% |

## Kisa Yorum

- Success delta (`Pretrained + PPO` - `Scratch PPO`): `+0.00` puan
- QoE delta (`Pretrained + PPO` - `Scratch PPO`): `+0.00`
- P95 latency delta (`Pretrained + PPO` - `Scratch PPO`): `+0.000` s
- Deadline miss ratio delta (`Pretrained + PPO` - `Scratch PPO`): `+0.00` puan
- Energy per success delta (`Pretrained + PPO` - `Scratch PPO`): `+0.0000`
- `Pretrained + PPO`, `%75` success esigine ortalama `10000` stepte ulasir; `Scratch PPO` ise `18333` stepte ulasir.
- Success curve AUC karsilastirmasi: scratch `0.6443`, pretrained `0.6710`.
- Yorum: pretrained baslangic hizini iyilestiriyor; ancak bu batchte scratch PPO orta-ileri asamada daha guclu final dengeye oturuyor.
