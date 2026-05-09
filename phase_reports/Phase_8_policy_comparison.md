Bkz. ortak kavram sozlugu: v2_docs/project_concepts_glossary.md

# Phase 8 Policy Comparison

Bu rapor Faz 8.4 kapsaminda ayni evaluator mantigi altinda vector-state PPO ve graph-aware policy karsilastirmasini toplar.

## Seed Aggregated Summary

| Model | Seeds | Success Mean | Success 95% CI | P95 Latency Mean | Avg Energy Mean | QoE Mean | Dominant Action |
|---|---:|---:|---:|---:|---:|---:|---|
| GraphPolicy_late | 3 | 74.87% | +/- 2.38% | 2.671 | 0.0694 | 61.51 | edge_75 |
| GraphPolicy_none | 3 | 75.00% | +/- 3.77% | 2.529 | 0.0705 | 62.36 | edge_75 |
| MLP-PPO | 3 | 66.53% | +/- 0.47% | 3.617 | 0.1321 | 48.45 | edge_75 |
| Pretrained MLP-PPO | 3 | 75.40% | +/- 0.60% | 2.892 | 0.0702 | 60.94 | edge_75 |

## Per-Seed Details

| Model | Seed | Success | P95 Latency | Avg Energy | QoE | Dominant Action |
|---|---:|---:|---:|---:|---:|---|
| MLP-PPO | 42 | 66.20% | 3.622 | 0.1309 | 48.09 | edge_75 |
| Pretrained MLP-PPO | 42 | 74.80% | 2.833 | 0.0705 | 60.63 | edge_75 |
| GraphPolicy_none | 42 | 71.20% | 2.777 | 0.0840 | 57.32 | edge_75 |
| GraphPolicy_late | 42 | 74.80% | 2.797 | 0.0723 | 60.82 | edge_75 |
| MLP-PPO | 43 | 67.00% | 3.662 | 0.1323 | 48.69 | edge_75 |
| Pretrained MLP-PPO | 43 | 75.60% | 2.948 | 0.0674 | 60.86 | edge_75 |
| GraphPolicy_none | 43 | 76.40% | 2.397 | 0.0635 | 64.41 | edge_75 |
| GraphPolicy_late | 43 | 72.80% | 2.619 | 0.0692 | 59.71 | edge_75 |
| MLP-PPO | 44 | 66.40% | 3.568 | 0.1330 | 48.56 | edge_75 |
| Pretrained MLP-PPO | 44 | 75.80% | 2.896 | 0.0728 | 61.32 | edge_75 |
| GraphPolicy_none | 44 | 77.40% | 2.413 | 0.0639 | 65.34 | edge_75 |
| GraphPolicy_late | 44 | 77.00% | 2.598 | 0.0666 | 64.01 | edge_75 |

## Yorum

- `GraphPolicy_none`, 3 ortak seed uzerinde ortalama `75.00%` success ile `MLP-PPO`yu net sekilde gecti ve `Pretrained MLP-PPO` ile ayni banda geldi.
- `GraphPolicy_late`, ortalama `74.87%` success ile yine `MLP-PPO`dan daha iyi, ancak bu final environment kosusunda `GraphPolicy_none`u gecemedi.
- `Pretrained MLP-PPO`, ortalama `75.40%` success ile en guclu vector-state baseline olarak kaldi.
- Tum modellerde dominant aksiyon `edge_75` oldugu icin Faz 7'den devreden `Edge %75` attractor problemi Faz 8 sonunda tamamen kirilmis sayilamaz.

## Dort Net Cevap

1. `MLP-PPO + semantic prior` iyi mi?

Evet, fakat sinirli. `MLP-PPO` hatti calisir ve anlamli bir baseline verir; ancak bu final Faz 8 kosusunda ortalama `66.53%` success, `3.617 s` p95 latency ve `48.45` QoE ile hem graph policy'lerin hem de `Pretrained MLP-PPO`nun gerisinde kalmistir.

2. `Graph policy + semantic prior` daha iyi mi?

Klasik `MLP-PPO`ya gore evet, `Pretrained MLP-PPO`ya gore hayir. `GraphPolicy_late`, `MLP-PPO`yu acik farkla gecerken (`74.87%` vs `66.53%`), `Pretrained MLP-PPO`nun gerisine cok az farkla dusmustur (`74.87%` vs `75.40%`).

3. `Graph policy`, semantic prior olmadan ne yapiyor?

Oldukca guclu bir performans veriyor. `GraphPolicy_none`, ortalama `75.00%` success, `2.529 s` p95 latency ve `62.36` QoE ile bu karsilastirmadaki en guclu graph varyanti oldu. Bu bulgu, graph yapisinin tek basina da ciddi bir karar sinyali tasidigini gosteriyor.

4. `Semantic prior`, gercekten katki sagliyor mu?

Kismen ve asamaya bagli. Supervised graph warm-start tarafinda 5-seed full protokolde `late` fusion, `none`a gore belirgin accuracy kazanci sagladi (`82.76%` vs `74.04%`). Ancak final environment evaluation'da ayni katki net ve tutarli bicimde korunmadi; `GraphPolicy_late`, `GraphPolicy_none`un ortalama env sonucunu gecemedi. Bu nedenle Faz 8 sonu icin en dogru yorum su: semantic prior, graph policy icin ogrenme asamasinda faydali bir rehber sinyaldir; fakat end-to-end environment performansina katkisi henuz kesinlesmis degildir.
