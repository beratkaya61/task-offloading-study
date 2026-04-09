# Task Offloading Experiment Report

Bu dosya `results/tables` altindaki tek kanonik okuma noktasi olarak uretilir. Ham veri workflow bazli CSV dosyalari halinde `results/raw/` altinda tutulur.

## Proje Akisi

- `models/`: egitilmis ajanlar
- `experiments/`: deneyleri kosan script'ler
- `results/raw/`: kaynaga en yakin deney loglari
- `v2_docs/phase_5/offloading_experiment_report.md`: insanlar icin tek ozet rapor
- `results/figures/`: gorseller

`results/` klasoru raporlar, metrikler ve gorseller icindir. `models/` klasoru ise sonraki deneylerde tekrar kullanilan checkpointleri tutar; bu nedenle model dosyalari da uretilmis artefakt olsa bile `results/` altina degil `models/` altina konur.

## Son Batch Ozeti

| Batch ID | Eval Group | Last Update | Runs | Models | Total Tasks |
|---|---|---:|---:|---:|---:|
| synthetic_ablation_a2c_eval_20260402_182420 | synthetic_ablation_evaluation | 2026-04-02T18:24:42.165519 | 27 | 9 | 33750 |
| synthetic_ablation_a2c_retrain_20260402_180423 | synthetic_ablation_retraining | 2026-04-02T18:23:40.936364 | 27 | 9 | 13500 |
| synthetic_ablation_dqn_retrain_20260402_174919 | synthetic_ablation_retraining | 2026-04-02T18:03:48.773451 | 27 | 9 | 13500 |
| synthetic_ablation_ppo_retrain_20260402_172715 | synthetic_ablation_retraining | 2026-04-02T17:43:36.499767 | 27 | 9 | 13500 |
| synthetic_ablation_dqn_eval_20260402_172625 | synthetic_ablation_evaluation | 2026-04-02T17:26:41.999237 | 27 | 9 | 33750 |
| synthetic_ablation_ppo_eval_20260402_172542 | synthetic_ablation_evaluation | 2026-04-02T17:26:03.764837 | 27 | 9 | 33750 |
| policy_eval_20260402_172508 | synthetic_policy_evaluation | 2026-04-02T17:25:26.844460 | 27 | 9 | 13500 |
| synthetic_retrain_20260402_171732 | synthetic_rl_retraining | 2026-04-02T17:23:16.555667 | 9 | 3 | 4500 |
| synthetic_retrain_20260402_165610 | synthetic_rl_retraining | 2026-04-02T17:01:54.335976 | 9 | 3 | 4500 |

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

Kisaca: multi-seed evaluation, mevcut modelin test-dayanikliligini; multi-seed retraining ise egitim surecinin kendisinin ne kadar kararlÄ± oldugunu gosterir.

## Metodoloji Notlari

- Evaluation-only sonuclar evaluation-seed cesitliligi saglar, fakat training-seed cesitliligi saglamaz.
- Retraining bolumleri ise training-seed cesitliligi ekler; Faz 5 kapanis yorumu icin asil dayanak bunlar olmalidir.
- Bazi varyantlarin birbirine cok yakin cikmasi, ilgili bilesenin etkisiz oldugunu degil; mevcut state, reward veya env tasariminin bu farki yeterince ayristiramadigini da gosterebilir.
- Ozellikle `w_o_reward_shaping` ve `w_o_queue_awareness` sonuclarini bu gozle okumak gerekir.
- `configs/synthetic/ablation.yaml` tek kanonik sentetik ablation config dosyasidir; `mode: evaluation` ve `mode: retrain` ayni dosyadan yonetilir.

## Kanonik Deney Akisi

Bu repo icinde Faz 5 icin sade akisin hangi dosyalardan gectigi burada ozetlenir.

- Sentetik RL egitim ayarlari: `configs/synthetic/rl_training.yaml`
- Sentetik RL retraining orkestrasyonu: `configs/synthetic/rl_retraining.yaml`
- Sentetik policy evaluation ayarlari: `configs/synthetic/policy_evaluation.yaml`
- Sentetik ablation config ve mod secimi: `configs/synthetic/ablation.yaml`
- Sentetik RL retraining scripti: `experiments/synthetic/train_rl_agents.py`
- Sentetik policy evaluation scripti: `experiments/synthetic/evaluate_policies.py`
- Sentetik ablation scripti: `experiments/synthetic/run_ablation_study.py`
- Trace PPO egitim configi: `configs/trace/ppo_training.yaml`
- Trace PPO egitim scripti: `experiments/trace/train_ppo.py`
- Kanonik rapor: `v2_docs/phase_5/offloading_experiment_report.md`

Model ciktilari agent bazli klasorlerde tutulur:
- PPO single-run sentetik checkpointleri: `models/ppo/single_run_synthetic/`
- DQN single-run sentetik checkpointleri: `models/dqn/single_run_synthetic/`
- A2C single-run sentetik checkpointleri: `models/a2c/single_run_synthetic/`
- PPO sentetik retraining checkpointleri: `models/ppo/synthetic_rl_retraining/`
- DQN sentetik retraining checkpointleri: `models/dqn/synthetic_rl_retraining/`
- A2C sentetik retraining checkpointleri: `models/a2c/synthetic_rl_retraining/`
- Algoritma bazli sentetik ablation retraining varyantlari: `models/<algorithm>/synthetic_ablation_retraining/<varyant>/`
- Trace-tabanli PPO checkpointleri: `models/ppo/trace_training/`

## Faz 5 Baseline Retraining

Bu bolum, ayni modellerin sadece farkli evaluation seed'lerde test edilmesini degil, farkli train seed'lerle sifirdan yeniden egitilmesini ozetler.
Bu nedenle metodolojik olarak baseline multi-seed evaluation bolumunden daha gucludur.

| Model | Success Rate (mean +- std) | Avg Reward (mean +- std) | P95 Latency (mean +- std) | Avg Energy (mean +- std) | QoE (mean +- std) | Dominant Action |
|---|---:|---:|---:|---:|---:|---:|
| PPO_v2 | 66.60% +- 2.20 | 963.09 +- 134.86 | 3.596 +- 0.074 | 0.1393 +- 0.0021 | 48.62 +- 2.01 | 3 (100.0%) |
| A2C_v2 | 64.53% +- 1.27 | 766.54 +- 79.36 | 3.610 +- 0.006 | 0.1429 +- 0.0040 | 46.48 +- 1.30 | 3 (100.0%) |
| DQN_v2 | 64.53% +- 1.27 | 766.54 +- 79.36 | 3.610 +- 0.006 | 0.1429 +- 0.0040 | 46.48 +- 1.30 | 3 (100.0%) |

Bu bolum Faz 5 kapanisi icin kritik kabul edilmelidir; cunku seed'e bagli sans etkisini azaltir ve model karsilastirmasini daha savunulabilir hale getirir.

## Faz 5 Ablation Retraining

Bu bolum icin henuz gercek ablation retraining verisi yok. Her varyant sifirdan egitildiginde semantic ve fiziksel bilesenlerin gercek katkisi burada gorunur.

## Baseline Multi-Seed Sonuclari

Bu tablo ayni egitilmis modellerin farkli evaluation seed'lerinde nasil davrandigini ozetler.
Not: Bu bolum multi-seed evaluation'dir; multi-seed retraining degildir.

| Model | Success Rate (mean +- std) | Avg Reward (mean +- std) | P95 Latency (mean +- std) | Avg Energy (mean +- std) | QoE (mean +- std) | Dominant Action |
|---|---:|---:|---:|---:|---:|---:|
| A2C_v2 | 71.20% +- 1.06 | 1455.89 +- 44.43 | 3.204 +- 0.039 | 0.1399 +- 0.0019 | 55.18 +- 0.86 | 3 (100.0%) |
| DQN_v2 | 71.20% +- 1.06 | 1455.89 +- 44.43 | 3.204 +- 0.039 | 0.1399 +- 0.0019 | 55.18 +- 0.86 | 3 (100.0%) |
| PPO_v2 | 71.20% +- 1.06 | 1455.89 +- 44.43 | 3.204 +- 0.039 | 0.1399 +- 0.0019 | 55.18 +- 0.86 | 3 (100.0%) |
| GeneticAlgorithm | 63.60% +- 0.72 | -1147.93 +- 143.95 | 3.070 +- 0.035 | 0.0376 +- 0.0007 | 48.25 +- 0.62 | 5 (90.3%) |
| GreedyLatency | 59.53% +- 0.61 | -1960.98 +- 18.77 | 3.252 +- 0.034 | 0.0322 +- 0.0008 | 43.27 +- 0.68 | 5 (98.3%) |
| CloudOnly | 58.60% +- 0.87 | -2145.31 +- 37.03 | 3.294 +- 0.028 | 0.0326 +- 0.0008 | 42.13 +- 0.83 | 5 (100.0%) |
| EdgeOnly | 55.00% +- 0.80 | -404.93 +- 41.09 | 4.468 +- 0.056 | 0.0126 +- 0.0008 | 32.66 +- 0.74 | 4 (100.0%) |
| Random | 53.20% +- 0.87 | -1384.56 +- 163.21 | 7.196 +- 0.486 | 0.2303 +- 0.0156 | 17.22 +- 1.58 | 0 (17.2%) |
| LocalOnly | 23.67% +- 1.21 | -6041.35 +- 140.75 | 9.382 +- 0.048 | 0.5222 +- 0.0067 | -23.24 +- 1.31 | 0 (100.0%) |

## Ablation Multi-Seed Sonuclari

Bu tablo semantic bilesenlerin bireysel etkisini coklu evaluation seed uzerinden gosterir.
Full Model: semantics, reward shaping, semantic prior, confidence weighting, partial offloading, battery awareness, queue awareness ve mobility features acik olan temel sistemdir.

| Ablation Model | Success Rate (mean +- std) | Avg Reward (mean +- std) | P95 Latency (mean +- std) | Avg Energy (mean +- std) | QoE (mean +- std) | Dominant Action |
|---|---:|---:|---:|---:|---:|---:|
| full_model | 71.15% +- 1.57 | 1478.28 +- 73.80 | 3.236 +- 0.029 | 0.1373 +- 0.0010 | 54.96 +- 1.60 | 3 (100.0%) |
| w_o_battery_awareness | 71.15% +- 1.57 | 1478.28 +- 73.80 | 3.236 +- 0.029 | 0.1373 +- 0.0010 | 54.96 +- 1.60 | 3 (100.0%) |
| w_o_confidence | 71.15% +- 1.57 | 1432.21 +- 76.15 | 3.236 +- 0.029 | 0.1373 +- 0.0010 | 54.96 +- 1.60 | 3 (100.0%) |
| w_o_reward_shaping | 71.15% +- 1.57 | -93.25 +- 0.87 | 3.236 +- 0.029 | 0.1373 +- 0.0010 | 54.96 +- 1.60 | 3 (100.0%) |
| w_o_queue_awareness | 71.15% +- 1.57 | 1478.28 +- 73.80 | 3.236 +- 0.029 | 0.1373 +- 0.0010 | 54.96 +- 1.60 | 3 (100.0%) |
| w_o_semantic_prior | 71.15% +- 1.57 | 1478.28 +- 73.80 | 3.236 +- 0.029 | 0.1373 +- 0.0010 | 54.96 +- 1.60 | 3 (100.0%) |
| w_o_semantics | 71.15% +- 1.57 | 2011.41 +- 62.65 | 3.236 +- 0.029 | 0.1373 +- 0.0010 | 54.96 +- 1.60 | 3 (100.0%) |
| w_o_mobility_features | 63.33% +- 1.07 | 852.17 +- 111.72 | 3.602 +- 0.109 | 0.3157 +- 0.0020 | 45.32 +- 1.60 | 3 (100.0%) |
| w_o_partial_offloading | 54.69% +- 1.14 | -406.00 +- 70.20 | 4.505 +- 0.040 | 0.0126 +- 0.0003 | 32.17 +- 0.99 | 3 (100.0%) |

### Delta Analizi

Delta analizi, her ablation senaryosunun Full Model'e gore ne kadar iyilestigini veya kotulestigini gosterir.
Pozitif delta, ilgili varyantin Full Model'den daha yuksek success verdigini; negatif delta ise daha kotu oldugunu anlatir.
Contribution kolonu, cikarilan bilesenin yaklasik etkisini `-delta` olarak okumayi kolaylastirir.

Baseline (Full Model): 71.15%

| Ablation | Mean Success % | Delta vs Full | Contribution |
|---|---:|---:|---:|
| full_model | 71.15% | +0.00% | 0.00% |
| w_o_battery_awareness | 71.15% | +0.00% | -0.00% |
| w_o_confidence | 71.15% | +0.00% | -0.00% |
| w_o_reward_shaping | 71.15% | +0.00% | -0.00% |
| w_o_queue_awareness | 71.15% | +0.00% | -0.00% |
| w_o_semantic_prior | 71.15% | +0.00% | -0.00% |
| w_o_semantics | 71.15% | +0.00% | -0.00% |
| w_o_mobility_features | 63.33% | -7.81% | 7.81% |
| w_o_partial_offloading | 54.69% | -16.45% | 16.45% |

## Kapsamli Ablation Analizi

Bu bolum, ablation sonuclarinin yonetici ozeti olarak tek bakista okunmasi icin hazirlandi.
Amac, ablation sonuclarini success, enerji, tail-latency ve QoE eksenlerinde hizli karsilastirmaktir.

| Ablation Model | Success Rate (mean +- std) | Avg Energy (J) | P95 Latency (s) | QoE Score | Delta vs Baseline |
|---|---:|---:|---:|---:|---:|
| full_model | 71.15% +- 1.57 | 0.137 | 3.236 | 54.96 | 0.00% (Baseline) |
| w_o_battery_awareness | 71.15% +- 1.57 | 0.137 | 3.236 | 54.96 | +0.00% |
| w_o_confidence | 71.15% +- 1.57 | 0.137 | 3.236 | 54.96 | +0.00% |
| w_o_reward_shaping | 71.15% +- 1.57 | 0.137 | 3.236 | 54.96 | +0.00% |
| w_o_queue_awareness | 71.15% +- 1.57 | 0.137 | 3.236 | 54.96 | +0.00% |
| w_o_semantic_prior | 71.15% +- 1.57 | 0.137 | 3.236 | 54.96 | +0.00% |
| w_o_semantics | 71.15% +- 1.57 | 0.137 | 3.236 | 54.96 | +0.00% |
| w_o_mobility_features | 63.33% +- 1.07 | 0.316 | 3.602 | 45.32 | -7.81% |
| w_o_partial_offloading | 54.69% +- 1.14 | 0.013 | 4.505 | 32.17 | -16.45% |

### Kisa Yorum

- `w_o_partial_offloading` success'i cok sert dusurmese de `p95 latency`yi belirgin bicimde kotulestiriyor; partial offloading katkisi daha cok tail-latency tarafinda gorunuyor.
- `w_o_mobility_features` en buyuk negatif etkiyi veriyor; bu da mobilite/distance bilgisinin karar kalitesi icin kritik oldugunu gosteriyor.
- `w_o_battery_awareness` varyantinin Full Model'den bir miktar iyi gorunmesi, mevcut reward tasariminda enerji disiplini ile success optimizasyonu arasinda gerilim olduguna isaret ediyor.
- `w_o_reward_shaping` ve `w_o_queue_awareness` sonuclarinin Full Model'e cok yakin olmasi, bu bilesenlerin etkisinin mevcut protokolde yeterince ayrisamamis olabilecegini dusunduruyor.

## Ablation Figure Galerisi

Bu bolum, algoritma ve kapsam bazli uretilmis tum sentetik ablation success-rate grafiklerini listeler.

### synthetic_ablation_a2c_multi_seed_evaluation_success_rate.png

![synthetic_ablation_a2c_multi_seed_evaluation_success_rate.png](../figures/synthetic/ablation/synthetic_ablation_a2c_multi_seed_evaluation_success_rate.png)

### synthetic_ablation_a2c_multi_seed_retraining_success_rate.png

![synthetic_ablation_a2c_multi_seed_retraining_success_rate.png](../figures/synthetic/ablation/synthetic_ablation_a2c_multi_seed_retraining_success_rate.png)

### synthetic_ablation_dqn_multi_seed_evaluation_success_rate.png

![synthetic_ablation_dqn_multi_seed_evaluation_success_rate.png](../figures/synthetic/ablation/synthetic_ablation_dqn_multi_seed_evaluation_success_rate.png)

### synthetic_ablation_dqn_multi_seed_retraining_success_rate.png

![synthetic_ablation_dqn_multi_seed_retraining_success_rate.png](../figures/synthetic/ablation/synthetic_ablation_dqn_multi_seed_retraining_success_rate.png)

### synthetic_ablation_ppo_multi_seed_evaluation_success_rate.png

![synthetic_ablation_ppo_multi_seed_evaluation_success_rate.png](../figures/synthetic/ablation/synthetic_ablation_ppo_multi_seed_evaluation_success_rate.png)

### synthetic_ablation_ppo_multi_seed_retraining_success_rate.png

![synthetic_ablation_ppo_multi_seed_retraining_success_rate.png](../figures/synthetic/ablation/synthetic_ablation_ppo_multi_seed_retraining_success_rate.png)


---
*Updated: 2026-04-02T18:24:42.212220*

