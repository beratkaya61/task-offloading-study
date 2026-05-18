Bkz. ortak kavram sozlugu: v2_docs/project_concepts_glossary.md

# Faz 5 - Real Data Report

Bu dosya, Faz 5'in `real_data` kolu icin tek kanonik rapordur.
Sentetik Faz 5 sonuclari ayri olarak `v2_docs/phase_5/synthetic_phase_5_report.md` dosyasinda tutulur.

## Artefakt Haritasi

- `results/phase_5/metrics/real_data/rl_retraining/`
- `results/phase_5/metrics/real_data/policy_evaluation/`
- `results/phase_5/metrics/real_data/ablation/`
- `results/phase_5/figures/real_data/ablation/`

## RL Retraining

| Algorithm | Seed | Success Rate | Avg Reward | P95 Latency | Avg Energy | QoE | Dominant Action |
|---|---:|---:|---:|---:|---:|---:|---:|
| PPO | 42 | 42.00% | 4275.56 | 2.0395 | 0.0594 | 31.80 | 3 |
| PPO | 43 | 42.00% | 4275.42 | 2.0393 | 0.0594 | 31.80 | 3 |
| PPO | 44 | 42.00% | 4275.59 | 2.0397 | 0.0594 | 31.80 | 3 |

Ara yorum:
- Kalibrasyon ve env-contract duzeltmesi sonrasi `PPO` artik eski `%2.4` seviyesinde degil.
- Ancak yeni tabloda farkli bir sorun var: tum seedler neredeyse ayni noktaya geliyor ve hepsi `action=3` etrafinda kilitleniyor.
- Bu nedenle bu tablo su an `benchmark tamamen bozuk`tan cok `PPO reward/prior kaynakli local optimuma sabitleniyor` olarak okunmalidir.

## Policy Evaluation

Kanonik policy-evaluation kosusu benchmark revizyonundan sonra henuz yeniden kosturulmadi.
Benchmark sanity-check protokolunde elde edilen heuristic referanslar ise sunlardir:

| Policy | Mean Success | Mean P95 Latency | Dominant Action |
|---|---:|---:|---:|
| GeneticAlgorithm | 48.73% | 1.4505 | 5 |
| GreedyLatency | 42.40% | 1.4497 | 5 |
| CloudOnly | 38.80% | 1.4713 | 5 |
| Random | 24.93% | 3.3610 | degisken |
| EdgeOnly | 12.20% | 2.6865 | 4 |
| LocalOnly | 0.40% | 5.1387 | 0 |

Bu tablo benchmark'in ilk bozuk haline gore toparlandigini, fakat hala cloud agirlikli zor bir karar rejimi urettigini gosteriyor.

## Ablation

Kanonik `real_data` ablation artefaktlari benchmark ve env-contract revizyonundan sonra henuz yeniden kosturulmadi.
Bir sonraki dogru sira:
1. PPO'nun `action=3` kilitlenmesini azaltmak
2. sonra `policy_evaluation`
3. sonra `ablation`
kosularini yeni contract uzerinde yeniden uretmek

