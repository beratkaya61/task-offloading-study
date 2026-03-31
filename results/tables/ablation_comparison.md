# 📊 Faz 5: Ablation Study Karşılaştırması

## Semantic Bileşenlerin Bireysel Katkısı

**Hedef:** Her ablation senaryosu, Full Model (%62.67) ile kıyaslanarak, 
bileşenlerin toplam başarıya katkısını ölçmek.

| config_model_type      |   metric_success_rate |   metric_avg_reward |
|:-----------------------|----------------------:|--------------------:|
| PPO_v2_Retrained       |               35.67   |              nan    |
| w_o_mobility_features  |                0.676  |             3127.41 |
| w_o_battery_awareness  |                0.65   |             3104.98 |
| w_o_confidence         |                0.646  |             3099.32 |
| w_o_queue_awareness    |                0.634  |             3034.32 |
| w_o_semantics          |                0.632  |             3072.99 |
| w_o_semantic_prior     |                0.628  |             2989.66 |
| PPO_v2                 |                0.6267 |             3011.61 |
| full_model             |                0.624  |             3062.65 |
| GeneticAlgorithm       |                0.5    |             2540.72 |
| GreedyLatency          |                0.4667 |             2345.87 |
| CloudOnly              |                0.46   |             2337.3  |
| Random                 |                0.4267 |             1482.53 |
| w_o_partial_offloading |                0.39   |             1666.38 |
| EdgeOnly               |                0.38   |             1506.82 |
| LocalOnly              |                0.1667 |            -1409.29 |
| w_o_reward_shaping     |                0      |              -98.81 |

---

## İstatistiksel Analiz

**Baseline (Full Model):** 62.40%

| Ablation | Başarı % | Delta | Katkı |
|----------|---------|-------|-------|
| PPO_v2_Retrained | 3567.00% | +3504.60% | -3504.60% |
| w_o_mobility_features | 67.60% | +5.20% | -5.20% |
| w_o_battery_awareness | 65.00% | +2.60% | -2.60% |
| w_o_confidence | 64.60% | +2.20% | -2.20% |
| w_o_queue_awareness | 63.40% | +1.00% | -1.00% |
| w_o_semantics | 63.20% | +0.80% | -0.80% |
| w_o_semantic_prior | 62.80% | +0.40% | -0.40% |
| PPO_v2 | 62.67% | +0.27% | -0.27% |
| full_model | 62.40% | +0.00% | 0.00% |
| GeneticAlgorithm | 50.00% | -12.40% | 12.40% |
| GreedyLatency | 46.67% | -15.73% | 15.73% |
| CloudOnly | 46.00% | -16.40% | 16.40% |
| Random | 42.67% | -19.73% | 19.73% |
| w_o_partial_offloading | 39.00% | -23.40% | 23.40% |
| EdgeOnly | 38.00% | -24.40% | 24.40% |
| LocalOnly | 16.67% | -45.73% | 45.73% |
| w_o_reward_shaping | 0.00% | -62.40% | 62.40% |

---
*Güncellenme: 2026-03-31T09:49:11.077189*
