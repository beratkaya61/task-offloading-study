Bkz. ortak kavram sozlugu: v2_docs/project_concepts_glossary.md

# Teacher Policy Sensitivity Report

Bu rapor, Faz 7 kapsaminda her teacher policy icin uretilen supervised pretraining ve staged PPO fine-tuning sonuclarini tek yerde toplar.
Ayrica ayri teacher markdown dosyalari tutulmaz; Faz 7 teacher bazli tum ozet burada korunur.

## Kanonik Secim

Kanonik Faz 7 teacher policy olarak `teacher_contextual_reward_aligned` ile devam edilmistir.
Secim gerekcesi sadece en yuksek success degil, ayni zamanda daha savunulabilir decision structure elde edilmesidir.
`teacher_latency_greedy` daha yuksek success delta verse de final politika `Full Cloud` tarafina kaymistir.
`teacher_contextual_reward_aligned` ise `Pretrained + PPO > Scratch PPO` sonucunu korurken dominant davranisi `Edge %75` ekseninde tutmustur.

## Final Karsilastirma

| Teacher Policy | Coverage (train) | Pretrain Val Acc | Pretrain Test Acc | Scratch Success | Pretrained Success | Delta Success | Scratch Dominant | Pretrained Dominant | Delta Step-to-75 | Delta AUC |
|---|---:|---:|---:|---:|---:|---:|---|---|---:|---:|
| teacher_latency_greedy | 252/252/294/462/252/588 | 78.67% | 77.56% | 63.00% | 77.33% | +14.33 puan | Edge %75 (100.0%) | Full Cloud (58.8%) | n/a | +0.0882 |
| teacher_contextual_reward_aligned | 252/252/294/462/252/588 | 82.67% | 83.11% | 63.00% | 75.20% | +12.20 puan | Edge %75 (100.0%) | Edge %75 (56.5%) | n/a | +0.0587 |
| teacher_balanced_semantic | 252/252/294/462/252/588 | 78.00% | 83.56% | 63.00% | 73.27% | +10.27 puan | Edge %75 (100.0%) | Full Cloud (45.8%) | n/a | +0.0482 |
| teacher_energy_greedy | 252/252/294/462/252/588 | 78.44% | 79.33% | 63.00% | 72.13% | +9.13 puan | Edge %75 (100.0%) | Full Cloud (42.4%) | n/a | +0.0337 |

Coverage (train) sirasi: `local / edge_25 / edge_50 / edge_75 / edge_100 / cloud`.
Delta kolonlari `Pretrained + PPO - Scratch PPO` olarak hesaplanmistir.

## Teacher Bazli Ozetler

### Latency Greedy

- Teacher policy: `teacher_latency_greedy`
- Teacher config seti: `configs/synthetic/teacher_policy_configs/supervised_pretraining_latency_greedy.yaml` ve `configs/synthetic/teacher_policy_configs/staged_training_latency_greedy.yaml`
- Pretrained checkpoint: `models/ppo/teacher_policy_pretrained/latency_greedy/ppo_pretrained.zip`
- Pretraining metrics CSV: `results/raw/synthetic/teacher_policy_sensitivity/latency_greedy/supervised_pretraining_metrics.csv`
- Staged final CSV: `results/raw/synthetic/teacher_policy_sensitivity/latency_greedy/staged_training_comparison.csv`
- Staged progress CSV: `results/raw/synthetic/teacher_policy_sensitivity/latency_greedy/staged_training_progress.csv`
- Coverage total (`local / edge_25 / edge_50 / edge_75 / edge_100 / cloud`): `293/269/357/574/269/1238`
- Coverage train (`local / edge_25 / edge_50 / edge_75 / edge_100 / cloud`): `252/252/294/462/252/588`
- Supervised pretraining: best epoch `8`, val acc `78.67%`, test acc `77.56%`
- Scratch PPO success: `63.00%`
- Pretrained + PPO success: `77.33%`
- Delta success: `+14.33 puan`
- Scratch dominant action: `Edge %75 (100.0%)`
- Pretrained dominant action: `Full Cloud (58.8%)`
- Delta p95 latency: `-1.2560`
- Delta energy: `-0.0913`
- Delta QoE: `20.61`
- Delta deadline miss: `-14.33 puan`
- Delta energy per success: `-0.1591`
- Delta success AUC: `0.0882`

Yorum:
Bu teacher en yuksek final success artisini verdi, ancak final politika `Full Cloud` agirlikli hale geldi. Bu nedenle Faz 8 oncesi davranissal risk tasidigi kabul edildi.

### Contextual Reward Aligned

- Teacher policy: `teacher_contextual_reward_aligned`
- Teacher config seti: `configs/synthetic/teacher_policy_configs/supervised_pretraining_contextual_reward_aligned.yaml` ve `configs/synthetic/teacher_policy_configs/staged_training_contextual_reward_aligned.yaml`
- Pretrained checkpoint: `models/ppo/teacher_policy_pretrained/contextual_reward_aligned/ppo_pretrained.zip`
- Pretraining metrics CSV: `results/raw/synthetic/teacher_policy_sensitivity/contextual_reward_aligned/supervised_pretraining_metrics.csv`
- Staged final CSV: `results/raw/synthetic/teacher_policy_sensitivity/contextual_reward_aligned/staged_training_comparison.csv`
- Staged progress CSV: `results/raw/synthetic/teacher_policy_sensitivity/contextual_reward_aligned/staged_training_progress.csv`
- Coverage total (`local / edge_25 / edge_50 / edge_75 / edge_100 / cloud`): `266/253/302/1049/252/878`
- Coverage train (`local / edge_25 / edge_50 / edge_75 / edge_100 / cloud`): `252/252/294/462/252/588`
- Supervised pretraining: best epoch `17`, val acc `82.67%`, test acc `83.11%`
- Scratch PPO success: `63.00%`
- Pretrained + PPO success: `75.20%`
- Delta success: `+12.20 puan`
- Scratch dominant action: `Edge %75 (100.0%)`
- Pretrained dominant action: `Edge %75 (56.5%)`
- Delta p95 latency: `-0.6961`
- Delta energy: `-0.0631`
- Delta QoE: `15.68`
- Delta deadline miss: `-12.20 puan`
- Delta energy per success: `-0.1197`
- Delta success AUC: `0.0587`

Yorum:
Bu teacher, Faz 7 icin kanonik secim olarak sabitlendi. Success artisi, dominant action'in `Full Cloud`a kaymadan korunmasi ve davranissal olarak daha savunulabilir bir warm-start saglamasi nedeniyle tercih edildi.

### Balanced Semantic

- Teacher policy: `teacher_balanced_semantic`
- Teacher config seti: `configs/synthetic/teacher_policy_configs/supervised_pretraining_balanced_semantic.yaml` ve `configs/synthetic/teacher_policy_configs/staged_training_balanced_semantic.yaml`
- Pretrained checkpoint: `models/ppo/teacher_policy_pretrained/balanced_semantic/ppo_pretrained.zip`
- Pretraining metrics CSV: `results/raw/synthetic/teacher_policy_sensitivity/balanced_semantic/supervised_pretraining_metrics.csv`
- Staged final CSV: `results/raw/synthetic/teacher_policy_sensitivity/balanced_semantic/staged_training_comparison.csv`
- Staged progress CSV: `results/raw/synthetic/teacher_policy_sensitivity/balanced_semantic/staged_training_progress.csv`
- Coverage total (`local / edge_25 / edge_50 / edge_75 / edge_100 / cloud`): `348/293/428/697/302/932`
- Coverage train (`local / edge_25 / edge_50 / edge_75 / edge_100 / cloud`): `252/252/294/462/252/588`
- Supervised pretraining: best epoch `21`, val acc `78.00%`, test acc `83.56%`
- Scratch PPO success: `63.00%`
- Pretrained + PPO success: `73.27%`
- Delta success: `+10.27 puan`
- Scratch dominant action: `Edge %75 (100.0%)`
- Pretrained dominant action: `Full Cloud (45.8%)`
- Delta p95 latency: `-0.4375`
- Delta energy: `-0.0598`
- Delta QoE: `12.45`
- Delta deadline miss: `-10.27 puan`
- Delta energy per success: `-0.1126`
- Delta success AUC: `0.0482`

Yorum:
Bu teacher dengeli bir ara nokta sundu, ancak final politika yine `Full Cloud` baskinligina yaklasti. Bu nedenle kanonik secim olarak alinmadi.

### Energy Greedy

- Teacher policy: `teacher_energy_greedy`
- Teacher config seti: `configs/synthetic/teacher_policy_configs/supervised_pretraining_energy_greedy.yaml` ve `configs/synthetic/teacher_policy_configs/staged_training_energy_greedy.yaml`
- Pretrained checkpoint: `models/ppo/teacher_policy_pretrained/energy_greedy/ppo_pretrained.zip`
- Pretraining metrics CSV: `results/raw/synthetic/teacher_policy_sensitivity/energy_greedy/supervised_pretraining_metrics.csv`
- Staged final CSV: `results/raw/synthetic/teacher_policy_sensitivity/energy_greedy/staged_training_comparison.csv`
- Staged progress CSV: `results/raw/synthetic/teacher_policy_sensitivity/energy_greedy/staged_training_progress.csv`
- Coverage total (`local / edge_25 / edge_50 / edge_75 / edge_100 / cloud`): `331/276/424/607/327/1035`
- Coverage train (`local / edge_25 / edge_50 / edge_75 / edge_100 / cloud`): `252/252/294/462/252/588`
- Supervised pretraining: best epoch `30`, val acc `78.44%`, test acc `79.33%`
- Scratch PPO success: `63.00%`
- Pretrained + PPO success: `72.13%`
- Delta success: `+9.13 puan`
- Scratch dominant action: `Edge %75 (100.0%)`
- Pretrained dominant action: `Full Cloud (42.4%)`
- Delta p95 latency: `-0.4301`
- Delta energy: `-0.0699`
- Delta QoE: `11.28`
- Delta deadline miss: `-9.13 puan`
- Delta energy per success: `-0.1250`
- Delta success AUC: `0.0337`

Yorum:
Bu teacher success artisi saglasa da final karar yapisi yine `Full Cloud` tarafina kaydi. Energy perspektifi yararli kaldi, ancak Faz 7'nin context-sensitive behavior hedefi icin yeterli gorulmedi.

