Bkz. ortak kavram sozlugu: v2_docs/project_concepts_glossary.md

# Supervised Pretraining Report

## Kanonik Teacher Secimi

Bu raporun kanonik pretraining sonucu, Faz 7'nin son staged-training karsilastirmasinda kullanilan teacher varyantini temsil eder.

- Selected teacher objective: `reward_aligned_oracle`
- Min margin filter: `10.0`
- Selected checkpoint: `models/ppo/pretrained/ppo_reward_aligned_pretrained.zip`
- Faz 7 staged-training branch icinde kullanilan final schedule:
  - `retention stage`: `10000` step, `learning_rate = 5e-05`, hafif `policy anchoring`
  - `refinement stage`: `20000` step, `learning_rate = 1.5e-04`, anchoring kapali

## Guncel Reward-Aligned Pretraining Sonucu

- Objective: `reward_aligned_oracle`
- Min margin filter: `10.0`
- Configured epoch count: `30`
- Executed epoch count: `6`
- Early stopping patience: `5`
- Early stopping triggered: `yes`
- Best epoch: `1`
- Best validation accuracy: `59.51%`
- Final train accuracy: `60.33%`
- Final validation accuracy: `59.51%`
- Final test accuracy: `61.03%`
- Metrics CSV: `results/raw/synthetic/pretraining/supervised_pretraining_metrics.csv`
- Checkpoint: `models/ppo/pretrained/ppo_reward_aligned_pretrained.zip`

## Teacher Varyantlari Karsilastirmasi

| Teacher objective | Min margin | Best val acc | Test acc | Faz 7 yorumu |
|---|---:|---:|---:|---|
| `weighted_objective_oracle` | `0.0` | `78.00%` | `80.22%` | supervised imitation acisindan daha kolay |
| `reward_aligned_oracle` | `10.0` | `59.51%` | `61.03%` | downstream RL reward ile daha hizali |

## Neden Son Kazanan Branch Reward-Aligned Teacher Kullandi

Ilk bakista `weighted_objective_oracle` daha iyi gorunur, cunku supervised accuracy daha yuksektir.
Ancak Faz 7'de ogrendigimiz sey su oldu:

- daha kolay taklit edilen teacher, her zaman daha iyi RL warm-start uretmez
- `reward_aligned_oracle`, supervised olarak daha zor olsa da downstream PPO objective ile daha uyumlu bir baslangic sagladi
- son `retention -> refinement` schedule ile final staged-training sonucu `Pretrained + PPO = 84.60%` ve `Scratch PPO = 83.67%` seviyesine geldi

Bu nedenle Faz 7 icin kanonik teacher secimi su an `reward_aligned_oracle` olarak tutuluyor.

