# 📊 Faz 5: Kapsamlı Ablation Analizi (Genişletilmiş Veri Seti)

| Ablation Model | Success Rate (± StdDev) | Avg Energy (J) | P95 Latency (s) | QoE Score | Delta vs Baseline |
| --- | --- | --- | --- | --- | --- |
| w_o_battery_awareness | **88.00%** (±0.00) | 0.042 | 1.308 | 81.46 | +8.00% |
| w_o_queue_awareness | **88.00%** (±0.00) | 0.036 | 1.228 | 81.86 | +8.00% |
| w_o_semantics | **88.00%** (±0.00) | 0.035 | 1.204 | 81.98 | +8.00% |
| w_o_mobility_features | **84.00%** (±0.00) | 0.048 | 1.268 | 77.66 | +4.00% |
| w_o_semantic_prior | **84.00%** (±0.00) | 0.019 | 1.417 | 76.92 | +4.00% |
| full_model | **80.00%** (±0.00) | 0.029 | 1.194 | 74.03 | 0.00% (Baseline) |
| w_o_confidence | **80.00%** (±0.00) | 0.025 | 1.246 | 73.77 | +0.00% |
| w_o_partial_offloading | **76.00%** (±0.00) | 0.015 | 1.434 | 68.83 | -4.00% |
| w_o_reward_shaping | **76.00%** (±0.00) | 0.037 | 1.288 | 69.56 | -4.00% |

_Not: Tüm veriler 25 bağımsız bölüm (episode) üzerinden varyans hesaplanarak türetilmiştir._
