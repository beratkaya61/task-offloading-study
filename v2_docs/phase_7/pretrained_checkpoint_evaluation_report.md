Bkz. ortak kavram sozlugu: v2_docs/project_concepts_glossary.md

# Pretrained Checkpoint Evaluation Report

Bu rapor, supervised pretraining sonrasi elde edilen checkpointlerin fine-tuning oncesi environment icindeki gercek basarisini olcer.
Amac, `supervised accuracy` ile `fine-tuned RL success` arasindaki farkin teacher bilgisinin unutulmasindan mi yoksa farkli metriklerden mi kaynaklandigini daha net gostermektir.

| Teacher Policy | Supervised Test Acc | Pretrained-Only Success | Fine-Tuned Success | Scratch Success | Delta (Fine-Tuned - Pretrained-Only) | Dominant Action |
|---|---:|---:|---:|---:|---:|---|
| teacher_latency_greedy | 77.56% | 76.93% | 77.33% | 63.00% | +0.40 puan | Full Cloud (53.1%) |
| teacher_contextual_reward_aligned | 83.11% | 74.47% | 75.20% | 63.00% | +0.73 puan | Edge %75 (58.7%) |
| teacher_balanced_semantic | 83.56% | 67.87% | 73.27% | 63.00% | +5.40 puan | Full Cloud (38.0%) |
| teacher_energy_greedy | 79.33% | 68.40% | 72.13% | 63.00% | +3.73 puan | Full Cloud (35.4%) |

Yorum anahtari:
- `Fine-Tuned Success > Pretrained-Only Success` ise RL fine-tuning faydali okunur.
- `Fine-Tuned Success ~= Pretrained-Only Success` ise RL fine-tuning sinirli katkili okunur.
- `Fine-Tuned Success < Pretrained-Only Success` ise teacher retention problemi guclu bir sinyal verir.

