# Task Offloading Experiment Report

Bu dosya `results/tables` altindaki tek kanonik okuma noktasi olarak uretilir. Ham veri kaynagi degismeden `results/raw/master_experiments.csv` icinde tutulur.

## Proje Akisi

- `models/`: egitilmis ajanlar
- `experiments/`: deneyleri kosan script'ler
- `results/raw/`: kaynaga en yakin deney loglari
- `results/tables/offloading_experiment_report.md`: insanlar icin tek ozet rapor
- `results/figures/`: gorseller

## Son Batch Ozeti

| Batch ID | Eval Group | Last Update | Runs | Models | Total Tasks |
|---|---|---:|---:|---:|---:|
| ablation_retrain_20260401_180038 | phase5_ablation_retraining | 2026-04-01T18:17:25.113075 | 27 | 9 | 13500 |
| ablation_20260401_173620 | phase5_ablation_evaluation | 2026-04-01T17:36:41.924933 | 27 | 9 | 33750 |
| retrain_20260401_171158 | phase5_baseline_retraining | 2026-04-01T17:18:10.802650 | 9 | 3 | 4500 |
| ablation_20260401_152509 | ablation_multiseed | 2026-04-01T15:25:58.268252 | 27 | 9 | 33750 |
| ablation_20260401_152041 | ablation_multiseed | 2026-04-01T15:21:30.571893 | 27 | 9 | 33750 |
| baseline_20260401_152006 | baseline_multiseed | 2026-04-01T15:20:24.038580 | 27 | 9 | 13500 |
| baseline_20260401_151809 | baseline_multiseed | 2026-04-01T15:18:27.286333 | 27 | 9 | 13500 |

## Bu Rapor Nasil Okunmali

- `Success Rate`: deadline icinde tamamlanan task oranidir. Yuksek olmasi iyidir.
- `P95 Latency`: en yavas kuyrugun davranisini gosterir. Ortalama degil, tail-latency odaklidir. Dusuk olmasi iyidir.
- `Avg Energy`: task basina ortalama enerji tuketimidir. Dusuk olmasi iyidir.
- `QoE`: success ve latency'nin birlesik, daha yorumlayici bir ozetidir.
- `Delta vs Full`: ilgili ablation varyantinin Full Model'e gore success farkidir.

Bu raporda iki farkli deney tipi birlikte bulunur:
- `evaluation`: mevcut checkpoint ailesi farkli seed'lerde test edilir.
- `retraining`: model her seed icin sifirdan yeniden egitilir.

Faz 5 yorumu yaparken retraining bolumleri, evaluation-only bolumlerinden daha guclu kanit olarak okunmalidir.

## Faz Siniri

Bu rapordaki baseline ve ablation sonuclari Faz 5 kapsaminda degerlendirilmelidir.
Cunku burada cevaplanan soru, mevcut model ailesi ve semantic bilesenlerin katkilarinin ne oldugudur.

Faz 5 kapsaminda kalan isler:
- baseline karsilastirmalarini daha saglam hale getirmek
- ablation sonuclarini coklu seed ile daha savunulabilir yapmak
- gerekiyorsa ayni sentetik/simule ortamda multi-seed retraining eklemek

Faz 6 ancak trace-driven egitim ve trace-driven evaluation ana akisa gectigimizde baslar.
Yani gercek gecis noktasi, sentetik episode yerine trace tabanli is yukleriyle modeli yeniden egitmek ve bu sonuclari raporlamaktir.

## Neden Multi-Seed Retraining

`Multi-seed evaluation` ile `multi-seed retraining` ayni sey degildir.

- `Multi-seed evaluation`: ayni egitilmis model farkli evaluation seed'lerinde test edilir.
- `Multi-seed retraining`: model her seed icin sifirdan yeniden egitilir ve sonra karsilastirilir.

RL ajanlari random initialization, experience ordering, environment stochasticity ve exploration farklari nedeniyle seed'e hassastir.
Bu yuzden tek bir seed'de iyi gorunen model baska bir seed'de ayni sekilde davranmayabilir.

Bu islemi yapmamizin temel nedenleri sunlardir:
- tek bir sansli training kosusuna asiri guvenmemek
- algoritmalarin gercekten daha iyi olup olmadigini varyansla birlikte okumak
- Faz 5 bulgularini Faz 6'ya tasimadan once daha savunulabilir hale getirmek
- sonraki trace-driven asamaya daha saglam bir sentetik temel ile gecmek

Kisaca: multi-seed evaluation, mevcut modelin test-dayanikliligini; multi-seed retraining ise egitim surecinin kendisinin ne kadar kararlı oldugunu gosterir.

## Metodoloji Notlari

- Evaluation-only sonuclar evaluation-seed cesitliligi saglar, fakat training-seed cesitliligi saglamaz.
- Retraining bolumleri ise training-seed cesitliligi ekler; Faz 5 kapanis yorumu icin asil dayanak bunlar olmalidir.
- Bazi varyantlarin birbirine cok yakin cikmasi, ilgili bilesenin etkisiz oldugunu degil; mevcut state, reward veya env tasariminin bu farki yeterince ayristiramadigini da gosterebilir.
- Ozellikle `w_o_reward_shaping` ve `w_o_queue_awareness` sonuclarini bu gozle okumak gerekir.
- `configs/ablation.yaml` tek kanonik ablation config dosyasidir; `mode: evaluation` ve `mode: retrain` ayni dosyadan yonetilir.

## Kanonik Deney Akisi

Bu repo icinde Faz 5 icin sade akisin hangi dosyalardan gectigi burada ozetlenir.

- Ortak egitim recetesi: `configs/synthetic_rl_training.yaml`
- Baseline retraining orkestrasyonu: `configs/phase5_baseline_retraining.yaml`
- Ablation config ve mod secimi: `configs/ablation.yaml`
- Baseline retraining scripti: `experiments/run_baseline_retraining.py`
- Ablation scripti: `experiments/run_ablation_study.py`
- Kanonik rapor: `results/tables/offloading_experiment_report.md`

Model ciktilari agent bazli klasorlerde tutulur:
- PPO baseline retraining: `models/ppo/phase5_baseline_retraining/`
- DQN baseline retraining: `models/dqn/phase5_baseline_retraining/`
- A2C baseline retraining: `models/a2c/phase5_baseline_retraining/`
- PPO ablation retraining varyantlari: `models/ppo/phase5_ablation_retraining/<varyant>/`

## Faz 5 Baseline Retraining

Bu bolum, ayni modellerin sadece farkli evaluation seed'lerde test edilmesini degil, farkli train seed'lerle sifirdan yeniden egitilmesini ozetler.
Bu nedenle metodolojik olarak baseline multi-seed evaluation bolumunden daha gucludur.

| Model | Success Rate (mean +- std) | Avg Reward (mean +- std) | P95 Latency (mean +- std) | Avg Energy (mean +- std) | QoE (mean +- std) |
|---|---:|---:|---:|---:|---:|
| PPO_v2 | 85.27% +- 1.10 | 1727.59 +- 21.13 | 1.980 +- 0.032 | 0.0153 +- 0.0009 | 75.37 +- 1.00 |
| A2C_v2 | 84.93% +- 1.21 | 1682.64 +- 17.74 | 1.993 +- 0.036 | 0.0153 +- 0.0025 | 74.97 +- 1.03 |
| DQN_v2 | 84.93% +- 1.21 | 1682.64 +- 17.74 | 1.993 +- 0.036 | 0.0153 +- 0.0025 | 74.97 +- 1.03 |

Bu bolum Faz 5 kapanisi icin kritik kabul edilmelidir; cunku seed'e bagli sans etkisini azaltir ve model karsilastirmasini daha savunulabilir hale getirir.

## Faz 5 Ablation Retraining

Bu bolum, ablation varyantlarinin sadece test edilmesini degil, her birinin ayri ayri sifirdan yeniden egitilmesini ozetler.
Bu nedenle semantic bilesen katkisini okumak icin en guvenilir Faz 5 tablosu budur.

| Ablation Model | Success Rate (mean +- std) | Avg Reward (mean +- std) | P95 Latency (mean +- std) | Avg Energy (mean +- std) | QoE (mean +- std) |
|---|---:|---:|---:|---:|---:|
| full_model | 85.73% +- 1.80 | 1719.31 +- 81.34 | 2.001 +- 0.025 | 0.0143 +- 0.0005 | 75.73 +- 1.93 |
| w_o_battery_awareness | 85.73% +- 1.80 | 1719.31 +- 81.34 | 2.001 +- 0.025 | 0.0143 +- 0.0005 | 75.73 +- 1.93 |
| w_o_confidence | 85.73% +- 1.80 | 1662.20 +- 85.66 | 2.001 +- 0.025 | 0.0143 +- 0.0005 | 75.73 +- 1.93 |
| w_o_partial_offloading | 85.73% +- 1.80 | 1719.31 +- 81.34 | 2.001 +- 0.025 | 0.0143 +- 0.0005 | 75.73 +- 1.93 |
| w_o_queue_awareness | 85.73% +- 1.80 | 1719.31 +- 81.34 | 2.001 +- 0.025 | 0.0143 +- 0.0005 | 75.73 +- 1.93 |
| w_o_semantic_prior | 85.73% +- 1.80 | 1719.31 +- 81.34 | 2.001 +- 0.025 | 0.0143 +- 0.0005 | 75.73 +- 1.93 |
| w_o_reward_shaping | 85.73% +- 1.80 | -57.20 +- 0.77 | 2.001 +- 0.025 | 0.0143 +- 0.0005 | 75.73 +- 1.93 |
| w_o_semantics | 79.53% +- 11.38 | 1848.74 +- 226.17 | 2.517 +- 0.913 | 0.0587 +- 0.0694 | 66.95 +- 15.95 |
| w_o_mobility_features | 74.20% +- 2.51 | 608.23 +- 7.07 | 2.624 +- 0.060 | 0.2534 +- 0.0042 | 61.08 +- 2.38 |

Baseline (Full Model): 85.73%

| Ablation | Mean Success % | Delta vs Full | Contribution |
|---|---:|---:|---:|
| full_model | 85.73% | +0.00% | 0.00% |
| w_o_battery_awareness | 85.73% | +0.00% | -0.00% |
| w_o_confidence | 85.73% | +0.00% | -0.00% |
| w_o_partial_offloading | 85.73% | +0.00% | -0.00% |
| w_o_queue_awareness | 85.73% | +0.00% | -0.00% |
| w_o_semantic_prior | 85.73% | +0.00% | -0.00% |
| w_o_reward_shaping | 85.73% | +0.00% | -0.00% |
| w_o_semantics | 79.53% | -6.20% | 6.20% |
| w_o_mobility_features | 74.20% | -11.53% | 11.53% |

Bu tablo, inference-only ablation sonucundan daha onemlidir; cunku policy'nin hangi sinyallerle yeniden sekillendigini gosterir.

## Baseline Multi-Seed Sonuclari

Bu tablo ayni egitilmis modellerin farkli evaluation seed'lerinde nasil davrandigini ozetler.
Not: Bu bolum multi-seed evaluation'dir; multi-seed retraining degildir.

| Model | Success Rate (mean +- std) | Avg Reward (mean +- std) | P95 Latency (mean +- std) | Avg Energy (mean +- std) | QoE (mean +- std) |
|---|---:|---:|---:|---:|---:|
| A2C_v2 | 84.87% +- 0.76 | 1654.81 +- 54.27 | 1.994 +- 0.004 | 0.0126 +- 0.0016 | 74.90 +- 0.77 |
| CloudOnly | 84.87% +- 0.76 | 1654.81 +- 54.27 | 1.994 +- 0.004 | 0.0126 +- 0.0016 | 74.90 +- 0.77 |
| DQN_v2 | 84.87% +- 0.76 | 1654.81 +- 54.27 | 1.994 +- 0.004 | 0.0126 +- 0.0016 | 74.90 +- 0.77 |
| GreedyLatency | 84.87% +- 0.76 | 1677.17 +- 65.93 | 1.994 +- 0.004 | 0.0126 +- 0.0016 | 74.90 +- 0.77 |
| GeneticAlgorithm | 84.47% +- 1.15 | 1686.33 +- 27.09 | 2.050 +- 0.003 | 0.0174 +- 0.0012 | 74.22 +- 1.17 |
| PPO_v2 | 82.53% +- 0.90 | 2099.92 +- 88.48 | 2.056 +- 0.006 | 0.0380 +- 0.0003 | 72.26 +- 0.93 |
| EdgeOnly | 54.67% +- 2.19 | -378.05 +- 172.43 | 4.696 +- 0.013 | 0.0126 +- 0.0016 | 31.19 +- 2.24 |
| Random | 53.00% +- 1.91 | -875.55 +- 149.29 | 7.139 +- 0.318 | 0.2175 +- 0.0074 | 17.31 +- 1.36 |
| LocalOnly | 25.60% +- 2.78 | -5411.06 +- 355.15 | 9.339 +- 0.029 | 0.5157 +- 0.0171 | -21.10 +- 2.88 |

## Ablation Multi-Seed Sonuclari

Bu tablo semantic bilesenlerin bireysel etkisini coklu evaluation seed uzerinden gosterir.
Full Model: semantics, reward shaping, semantic prior, confidence weighting, partial offloading, battery awareness, queue awareness ve mobility features acik olan temel sistemdir.

| Ablation Model | Success Rate (mean +- std) | Avg Reward (mean +- std) | P95 Latency (mean +- std) | Avg Energy (mean +- std) | QoE (mean +- std) |
|---|---:|---:|---:|---:|---:|
| full_model | 85.73% +- 1.80 | 1719.31 +- 81.34 | 2.001 +- 0.025 | 0.0143 +- 0.0005 | 75.73 +- 1.93 |
| w_o_battery_awareness | 85.73% +- 1.80 | 1719.31 +- 81.34 | 2.001 +- 0.025 | 0.0143 +- 0.0005 | 75.73 +- 1.93 |
| w_o_confidence | 85.73% +- 1.80 | 1662.20 +- 85.66 | 2.001 +- 0.025 | 0.0143 +- 0.0005 | 75.73 +- 1.93 |
| w_o_partial_offloading | 85.73% +- 1.80 | 1719.31 +- 81.34 | 2.001 +- 0.025 | 0.0143 +- 0.0005 | 75.73 +- 1.93 |
| w_o_queue_awareness | 85.73% +- 1.80 | 1719.31 +- 81.34 | 2.001 +- 0.025 | 0.0143 +- 0.0005 | 75.73 +- 1.93 |
| w_o_semantic_prior | 85.73% +- 1.80 | 1719.31 +- 81.34 | 2.001 +- 0.025 | 0.0143 +- 0.0005 | 75.73 +- 1.93 |
| w_o_reward_shaping | 85.73% +- 1.80 | -57.20 +- 0.77 | 2.001 +- 0.025 | 0.0143 +- 0.0005 | 75.73 +- 1.93 |
| w_o_semantics | 79.53% +- 11.38 | 1848.74 +- 226.17 | 2.517 +- 0.913 | 0.0587 +- 0.0694 | 66.95 +- 15.95 |
| w_o_mobility_features | 74.20% +- 2.51 | 608.23 +- 7.07 | 2.624 +- 0.060 | 0.2534 +- 0.0042 | 61.08 +- 2.38 |

### Delta Analizi

Delta analizi, her ablation senaryosunun Full Model'e gore ne kadar iyilestigini veya kotulestigini gosterir.
Pozitif delta, ilgili varyantin Full Model'den daha yuksek success verdigini; negatif delta ise daha kotu oldugunu anlatir.
Contribution kolonu, cikarilan bilesenin yaklasik etkisini `-delta` olarak okumayi kolaylastirir.

Baseline (Full Model): 85.73%

| Ablation | Mean Success % | Delta vs Full | Contribution |
|---|---:|---:|---:|
| full_model | 85.73% | +0.00% | 0.00% |
| w_o_battery_awareness | 85.73% | +0.00% | -0.00% |
| w_o_confidence | 85.73% | +0.00% | -0.00% |
| w_o_partial_offloading | 85.73% | +0.00% | -0.00% |
| w_o_queue_awareness | 85.73% | +0.00% | -0.00% |
| w_o_semantic_prior | 85.73% | +0.00% | -0.00% |
| w_o_reward_shaping | 85.73% | +0.00% | -0.00% |
| w_o_semantics | 79.53% | -6.20% | 6.20% |
| w_o_mobility_features | 74.20% | -11.53% | 11.53% |

### Figure

![Ablation Impact](../figures/ablation_impact.png)

## Kapsamli Ablation Analizi

Bu bolum, ablation sonuclarinin yonetici ozeti olarak tek bakista okunmasi icin hazirlandi.
Amac, ablation sonuclarini success, enerji, tail-latency ve QoE eksenlerinde hizli karsilastirmaktir.

| Ablation Model | Success Rate (mean +- std) | Avg Energy (J) | P95 Latency (s) | QoE Score | Delta vs Baseline |
|---|---:|---:|---:|---:|---:|
| full_model | 85.73% +- 1.80 | 0.014 | 2.001 | 75.73 | 0.00% (Baseline) |
| w_o_battery_awareness | 85.73% +- 1.80 | 0.014 | 2.001 | 75.73 | +0.00% |
| w_o_confidence | 85.73% +- 1.80 | 0.014 | 2.001 | 75.73 | +0.00% |
| w_o_partial_offloading | 85.73% +- 1.80 | 0.014 | 2.001 | 75.73 | +0.00% |
| w_o_queue_awareness | 85.73% +- 1.80 | 0.014 | 2.001 | 75.73 | +0.00% |
| w_o_semantic_prior | 85.73% +- 1.80 | 0.014 | 2.001 | 75.73 | +0.00% |
| w_o_reward_shaping | 85.73% +- 1.80 | 0.014 | 2.001 | 75.73 | +0.00% |
| w_o_semantics | 79.53% +- 11.38 | 0.059 | 2.517 | 66.95 | -6.20% |
| w_o_mobility_features | 74.20% +- 2.51 | 0.253 | 2.624 | 61.08 | -11.53% |

### Kisa Yorum

- `w_o_partial_offloading` success'i cok sert dusurmese de `p95 latency`yi belirgin bicimde kotulestiriyor; partial offloading katkisi daha cok tail-latency tarafinda gorunuyor.
- `w_o_mobility_features` en buyuk negatif etkiyi veriyor; bu da mobilite/distance bilgisinin karar kalitesi icin kritik oldugunu gosteriyor.
- `w_o_battery_awareness` varyantinin Full Model'den bir miktar iyi gorunmesi, mevcut reward tasariminda enerji disiplini ile success optimizasyonu arasinda gerilim olduguna isaret ediyor.
- `w_o_reward_shaping` ve `w_o_queue_awareness` sonuclarinin Full Model'e cok yakin olmasi, bu bilesenlerin etkisinin mevcut protokolde yeterince ayrisamamis olabilecegini dusunduruyor.

---
*Updated: 2026-04-01T18:41:37.384848*
