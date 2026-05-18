# Real-Data Benchmark Sanity Check

Bu raporun amaci, kalibre edilmis `real_composite_trace` benchmark'inin environment icinde tamamen bozuk olup olmadigini ayirmaktir.
Burada hedef PPO'yu degil, benchmark + env kombinasyonunu heuristic politikalarla sinamaktir.

- Kaynak CSV: `results/phase_5/metrics/real_data/service_class_sanity/service_class_heuristic_sanity.csv`
- Split: `test`
- Seedler: `42, 43, 44`
- Episode sayisi: her seed icin `10`

## Heuristic Sonuclari

| Policy | Mean Success | Std | Mean Reward | Mean P95 Latency (s) | Mean Energy | Mean QoE | Dominant Action |
|---|---:|---:|---:|---:|---:|---:|---|
| Random | 35.53% | 0.0258 | 3376.60 | 2.2449 | 0.0664 | 24.31 | 2 (2/3) |
| GeneticAlgorithm | 24.40% | 0.0092 | 19.05 | 1.4307 | 0.0193 | 17.25 | 5 (3/3) |
| EdgeOnly | 24.00% | 0.0000 | 3175.45 | 1.9456 | 0.0177 | 14.27 | 4 (3/3) |
| GreedyLatency | 22.00% | 0.0000 | -256.25 | 1.4407 | 0.0182 | 14.80 | 5 (3/3) |
| LocalOnly | 21.00% | 0.0000 | 1582.33 | 3.5580 | 0.1319 | 3.21 | 0 (3/3) |
| CloudOnly | 8.00% | 0.0000 | -1633.74 | 1.4617 | 0.0177 | 0.69 | 5 (3/3) |

## PPO Referans Notu

- Mevcut PPO real-data retraining ortalamasi: `42.00%` (`std=0.0000`)
- En iyi heuristic ortalamasi: `Random` ile `35.53%`

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
