# Real-Data Benchmark Sanity Check

Bu raporun amaci, kalibre edilmis `real_composite_trace` benchmark'inin environment icinde tamamen bozuk olup olmadigini ayirmaktir.
Burada hedef PPO'yu degil, benchmark + env kombinasyonunu heuristic politikalarla sinamaktir.

- Kaynak CSV: `results/phase_6/metrics/real_data/benchmark_sanity/real_data_benchmark_sanity.csv`
- Split: `test`
- Seedler: `42, 43, 44`
- Episode sayisi: her seed icin `10`

## Heuristic Sonuclari

| Policy | Mean Success | Std | Mean Reward | Mean P95 Latency (s) | Mean Energy | Mean QoE | Dominant Action |
|---|---:|---:|---:|---:|---:|---:|---|
| GeneticAlgorithm | 48.73% | 0.0042 | 2456.60 | 1.4369 | 0.0132 | 41.55 | 5 (3/3) |
| GreedyLatency | 42.40% | 0.0000 | 1764.39 | 1.4499 | 0.0110 | 35.15 | 5 (3/3) |
| CloudOnly | 38.80% | 0.0000 | 1200.61 | 1.4711 | 0.0083 | 31.44 | 5 (3/3) |
| Random | 26.93% | 0.0076 | 2107.91 | 3.3896 | 0.0916 | 9.98 | 5 (2/3) |
| EdgeOnly | 12.13% | 0.0012 | 1732.40 | 2.6864 | 0.0083 | -1.30 | 4 (3/3) |
| LocalOnly | 0.40% | 0.0000 | -816.15 | 5.1387 | 0.2128 | -25.29 | 0 (3/3) |

## PPO Referans Notu

- Mevcut PPO real-data retraining ortalamasi: `42.00%` (`std=0.0000`)
- En iyi heuristic ortalamasi: `GeneticAlgorithm` ile `48.73%`

Seed bazli PPO tablo okumasinda yeni contract altinda farkli bir sorun goruluyor:
- PPO artik yuksek varyansla ikiye ayrilmiyor;
- bunun yerine tum seedlerde neredeyse ayni noktaya sabitleniyor (`42.00%`);
- dominant aksiyon da sistematik bicimde `action=3` olarak kaliyor.

Bu tablo su yorumu destekliyor:
- benchmark tamamen fiziksel olarak bozuk degil, cunku heuristic aile anlamli ayrisiyor;
- PPO ise artik kararsiz degil, ama semantic/reward tarafinin ittiği zayif bir local optimuma kilitlenmis gorunuyor;
- dolayisiyla bundan sonraki mudahale noktasi benchmark'i tekrar bozmak degil, reward geometry / semantic prior bias / PPO aksiyon kilitlenmesidir.

## Karar

Bu benchmark, kalibrasyon sonrasi haliyle `imkansiz gorev coplugu` degil.
Fakat kolay bir benchmark da degil; cloud-benzeri kararlarin avantajli oldugu sert bir deadline rejimi var.
Bu nedenle Faz 5 real-data PPO kosulari devam etmeden once, PPO'nun neden partial-edge moduna cokebildigi ayrica incelenmelidir.
