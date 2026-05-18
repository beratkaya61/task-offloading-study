# Phase 5R Real-Data Oracle Audit

Bu rapor, mevcut veri setini `real-world trace-driven hybrid benchmark` olarak okur; tam gercek MEC logu iddiasi tasimaz.
Amac, algoritma sonuclarindan once benchmark'in fiziksel olarak cozulur olup olmadigini gormektir.

## Gate Result

- split: `test`
- task count: `500`
- oracle ceiling / success rate: `91.80%`
- impossible task ratio: `8.20%`
- dominant oracle action: `edge_75` (`71.80%`)
- status: `usable`

Karar: Benchmark, Faz 5R policy ve ablation kosulari icin makul bir cozulur-zor rejimde gorunuyor.

## Per-Action Feasibility

| Action | Name | Feasible Rate | Avg Delay (s) | P95 Delay (s) |
|---:|---|---:|---:|---:|
| 0 | local | 21.00% | 1.3195 | 3.5580 |
| 1 | edge_25 | 23.60% | 1.0178 | 2.7140 |
| 2 | edge_50 | 52.20% | 0.6879 | 1.8229 |
| 3 | edge_75 | 88.00% | 0.5495 | 1.4107 |
| 4 | edge_100 | 49.40% | 0.7050 | 1.8448 |
| 5 | cloud | 39.20% | 0.6492 | 1.1218 |

## Distribution Summaries

| Field | Mean | Std | Min | P50 | P95 | Max |
|---|---:|---:|---:|---:|---:|---:|
| deadline_window_s | 0.7222 | 0.8901 | 0.1200 | 0.6026 | 1.4668 | 6.7481 |
| oracle_deadline_tightness | 0.7013 | 0.2634 | 0.2022 | 0.7776 | 1.0326 | 1.1756 |
| cpu_cycles | 1319474179.4280 | 1348094457.3938 | 40000000.0000 | 1155553979.5000 | 3558031897.2000 | 9500000000.0000 |
| data_size_kb | 1489.8960 | 1082.1589 | 33.0000 | 1755.0000 | 2893.0000 | 4039.0000 |

## Interpretation Rule

- Oracle ceiling dusukse: dusuk PPO/heuristic sonucu algoritma basarisizligi olarak yorumlanmaz.
- Oracle ceiling yuksek ama modeller dusukse: reward, semantic prior, state veya action-collapse sorunu aranir.
- Tek aksiyona cok yogunlasan politika final bilimsel iddia olarak kullanilmadan once action-collapse diagnostiginden gecmelidir.

Detailed action CSV: `results\phase_5\metrics\real_data\oracle\service_class_oracle_action_audit.csv`
