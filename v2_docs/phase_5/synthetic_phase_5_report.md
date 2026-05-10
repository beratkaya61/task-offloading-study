# Task Offloading Experiment Report - Synthetic Phase 5

Bu dosya Faz 5 icin tek kanonik okuma noktasi olarak uretilir. Ham veri workflow bazli CSV dosyalari halinde `results/phase_*/metrics/` altinda tutulur.

## Proje Akisi

- `models/`: egitilmis ajanlar
- `experiments/`: deneyleri kosan script'ler
- `results/phase_*/metrics/`: kaynaga en yakin deney loglari
- `v2_docs/phase_5/synthetic_phase_5_report.md`: insanlar icin tek ozet rapor
- `results/phase_*/figures/`: gorseller

`results/` klasoru raporlar, metrikler ve gorseller icindir. `models/` klasoru ise sonraki deneylerde tekrar kullanilan checkpointleri tutar; bu nedenle model dosyalari da uretilmis artefakt olsa bile `results/` altina degil `models/` altina konur.

## Son Batch Ozeti

| Batch ID | Eval Group | Last Update | Runs | Models | Total Tasks |
|---|---|---:|---:|---:|---:|
| synthetic_ablation_a2c_eval_20260510_195016 | synthetic_ablation_evaluation | 2026-05-10T19:52:29.582577 | 27 | 9 | 33750 |
| synthetic_ablation_a2c_retrain_20260510_185210 | synthetic_ablation_retraining | 2026-05-10T19:50:02.118049 | 27 | 9 | 13500 |
| synthetic_ablation_a2c_retrain_20260510_175130 | synthetic_ablation_retraining | 2026-05-10T18:50:47.054895 | 26 | 9 | 13000 |
| synthetic_ablation_dqn_eval_20260510_175016 | synthetic_ablation_evaluation | 2026-05-10T17:51:12.983561 | 27 | 9 | 33750 |
| synthetic_ablation_dqn_retrain_20260510_165228 | synthetic_ablation_retraining | 2026-05-10T17:48:40.844199 | 27 | 9 | 13500 |
| synthetic_ablation_ppo_eval_20260510_164603 | synthetic_ablation_evaluation | 2026-05-10T16:51:55.627781 | 27 | 9 | 33750 |
| synthetic_ablation_ppo_retrain_20260510_155751 | synthetic_ablation_retraining | 2026-05-10T16:45:27.964009 | 27 | 9 | 13500 |
| policy_eval_20260510_155654 | synthetic_policy_evaluation | 2026-05-10T15:57:20.628217 | 27 | 9 | 13500 |
| synthetic_retrain_20260510_153257 | synthetic_rl_retraining | 2026-05-10T15:53:53.770555 | 9 | 3 | 4500 |

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
- `configs/phase_5/synthetic_ablation.yaml` tek kanonik sentetik ablation config dosyasidir; `mode: evaluation` ve `mode: retrain` ayni dosyadan yonetilir.

## Kanonik Deney Akisi

Bu repo icinde Faz 5 icin sade akisin hangi dosyalardan gectigi burada ozetlenir.

- Sentetik RL egitim ayarlari: `configs/phase_5/synthetic_rl_training.yaml`
- Sentetik RL retraining orkestrasyonu: `configs/phase_5/synthetic_rl_retraining.yaml`
- Sentetik policy evaluation ayarlari: `configs/phase_5/synthetic_policy_evaluation.yaml`
- Sentetik ablation config ve mod secimi: `configs/phase_5/synthetic_ablation.yaml`
- Sentetik RL retraining scripti: `experiments/phase_5/run_synthetic_rl_retraining.py`
- Sentetik policy evaluation scripti: `experiments/phase_5/run_synthetic_policy_evaluation.py`
- Sentetik ablation scripti: `experiments/phase_5/run_synthetic_ablation_study.py`
- Sentetik-trace RL egitim configi: `configs/phase_6/synthetic_trace_rl_training.yaml`
- Trace RL egitim scripti: `experiments/phase_6/run_trace_training.py`
- Kanonik rapor: `v2_docs/phase_5/synthetic_phase_5_report.md`

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
| DQN | 69.80% +- 6.49 | 529.72 +- 554.25 | 3.650 +- 0.053 | 0.0933 +- 0.0397 | 51.55 +- 6.75 | 3 (51.4%) |
| PPO | 65.47% +- 3.63 | 1259.75 +- 175.89 | 3.652 +- 0.043 | 0.1388 +- 0.0055 | 47.20 +- 3.81 | 3 (100.0%) |
| A2C | 65.27% +- 2.60 | 1241.49 +- 71.04 | 3.683 +- 0.026 | 0.1380 +- 0.0022 | 46.85 +- 2.73 | 3 (100.0%) |

Bu bolum Faz 5 kapanisi icin kritik kabul edilmelidir; cunku seed'e bagli sans etkisini azaltir ve model karsilastirmasini daha savunulabilir hale getirir.

## Faz 5 Ablation Retraining

Bu bolum icin henuz gercek ablation retraining verisi yok. Her varyant sifirdan egitildiginde semantic ve fiziksel bilesenlerin gercek katkisi burada gorunur.

## Baseline Multi-Seed Sonuclari

Bu tablo ayni egitilmis modellerin farkli evaluation seed'lerinde nasil davrandigini ozetler.
Not: Bu bolum multi-seed evaluation'dir; multi-seed retraining degildir.

| Model | Success Rate (mean +- std) | Avg Reward (mean +- std) | P95 Latency (mean +- std) | Avg Energy (mean +- std) | QoE (mean +- std) | Dominant Action |
|---|---:|---:|---:|---:|---:|---:|
| GreedyLatency | 78.67% +- 0.81 | 159.90 +- 19.53 | 2.287 +- 0.031 | 0.0132 +- 0.0014 | 67.23 +- 0.73 | 5 (96.9%) |
| CloudOnly | 78.47% +- 0.99 | 52.79 +- 57.54 | 2.294 +- 0.030 | 0.0132 +- 0.0014 | 67.00 +- 0.90 | 5 (100.0%) |
| GeneticAlgorithm | 78.07% +- 0.76 | 263.80 +- 140.29 | 2.344 +- 0.033 | 0.0183 +- 0.0016 | 66.35 +- 0.88 | 5 (90.5%) |
| DQN | 72.40% +- 0.72 | 324.64 +- 38.10 | 3.575 +- 0.055 | 0.0736 +- 0.0021 | 54.53 +- 0.98 | 5 (69.6%) |
| A2C | 64.87% +- 0.31 | 1326.56 +- 47.72 | 3.619 +- 0.056 | 0.1353 +- 0.0023 | 46.77 +- 0.41 | 3 (100.0%) |
| PPO | 64.87% +- 0.31 | 1326.56 +- 47.72 | 3.619 +- 0.056 | 0.1353 +- 0.0023 | 46.77 +- 0.41 | 3 (100.0%) |
| Random | 53.87% +- 1.42 | -1092.22 +- 180.30 | 6.873 +- 0.195 | 0.2155 +- 0.0216 | 19.50 +- 2.29 | 2 (17.3%) |
| EdgeOnly | 51.40% +- 1.25 | -296.67 +- 45.37 | 4.811 +- 0.076 | 0.0132 +- 0.0014 | 27.35 +- 1.61 | 4 (100.0%) |
| LocalOnly | 27.47% +- 1.10 | -5292.34 +- 77.71 | 9.272 +- 0.155 | 0.5016 +- 0.0052 | -18.89 +- 1.61 | 0 (100.0%) |

## Ablation Multi-Seed Sonuclari

Bu tablo semantic bilesenlerin bireysel etkisini coklu evaluation seed uzerinden gosterir.
Full Model: semantics, reward shaping, semantic prior, confidence weighting, partial offloading, battery awareness, queue awareness ve mobility features acik olan temel sistemdir.

| Ablation Model | Success Rate (mean +- std) | Avg Reward (mean +- std) | P95 Latency (mean +- std) | Avg Energy (mean +- std) | QoE (mean +- std) | Dominant Action |
|---|---:|---:|---:|---:|---:|---:|
| full_model | 64.43% +- 0.82 | 1283.94 +- 58.27 | 3.652 +- 0.028 | 0.1361 +- 0.0019 | 46.17 +- 0.87 | 3 (100.0%) |
| w_o_battery_awareness | 64.43% +- 0.82 | 1283.94 +- 58.27 | 3.652 +- 0.028 | 0.1361 +- 0.0019 | 46.17 +- 0.87 | 3 (100.0%) |
| w_o_confidence | 64.43% +- 0.82 | 1282.30 +- 64.82 | 3.652 +- 0.028 | 0.1361 +- 0.0019 | 46.17 +- 0.87 | 3 (100.0%) |
| w_o_reward_shaping | 64.43% +- 0.82 | -1539.84 +- 15.12 | 3.652 +- 0.028 | 0.1361 +- 0.0019 | 46.17 +- 0.87 | 3 (100.0%) |
| w_o_queue_awareness | 64.43% +- 0.82 | 1283.94 +- 58.27 | 3.652 +- 0.028 | 0.1361 +- 0.0019 | 46.17 +- 0.87 | 3 (100.0%) |
| w_o_semantic_prior | 64.43% +- 0.82 | 1283.94 +- 58.27 | 3.652 +- 0.028 | 0.1361 +- 0.0019 | 46.17 +- 0.87 | 3 (100.0%) |
| w_o_semantics | 64.43% +- 0.82 | 1282.30 +- 64.82 | 3.652 +- 0.028 | 0.1361 +- 0.0019 | 46.17 +- 0.87 | 3 (100.0%) |
| w_o_mobility_features | 64.37% +- 0.83 | 1280.54 +- 60.65 | 3.653 +- 0.029 | 0.1363 +- 0.0019 | 46.11 +- 0.89 | 3 (100.0%) |
| w_o_partial_offloading | 49.92% +- 0.73 | -364.57 +- 52.15 | 4.854 +- 0.039 | 0.0132 +- 0.0008 | 25.65 +- 0.89 | 3 (100.0%) |

### Delta Analizi

Delta analizi, her ablation senaryosunun Full Model'e gore ne kadar iyilestigini veya kotulestigini gosterir.
Pozitif delta, ilgili varyantin Full Model'den daha yuksek success verdigini; negatif delta ise daha kotu oldugunu anlatir.
Contribution kolonu, cikarilan bilesenin yaklasik etkisini `-delta` olarak okumayi kolaylastirir.

Baseline (Full Model): 64.43%

| Ablation | Mean Success % | Delta vs Full | Contribution |
|---|---:|---:|---:|
| full_model | 64.43% | +0.00% | 0.00% |
| w_o_battery_awareness | 64.43% | +0.00% | -0.00% |
| w_o_confidence | 64.43% | +0.00% | -0.00% |
| w_o_reward_shaping | 64.43% | +0.00% | -0.00% |
| w_o_queue_awareness | 64.43% | +0.00% | -0.00% |
| w_o_semantic_prior | 64.43% | +0.00% | -0.00% |
| w_o_semantics | 64.43% | +0.00% | -0.00% |
| w_o_mobility_features | 64.37% | -0.05% | 0.05% |
| w_o_partial_offloading | 49.92% | -14.51% | 14.51% |

## Kapsamli Ablation Analizi

Bu bolum, ablation sonuclarinin yonetici ozeti olarak tek bakista okunmasi icin hazirlandi.
Amac, ablation sonuclarini success, enerji, tail-latency ve QoE eksenlerinde hizli karsilastirmaktir.

| Ablation Model | Success Rate (mean +- std) | Avg Energy (J) | P95 Latency (s) | QoE Score | Delta vs Baseline |
|---|---:|---:|---:|---:|---:|
| full_model | 64.43% +- 0.82 | 0.136 | 3.652 | 46.17 | 0.00% (Baseline) |
| w_o_battery_awareness | 64.43% +- 0.82 | 0.136 | 3.652 | 46.17 | +0.00% |
| w_o_confidence | 64.43% +- 0.82 | 0.136 | 3.652 | 46.17 | +0.00% |
| w_o_reward_shaping | 64.43% +- 0.82 | 0.136 | 3.652 | 46.17 | +0.00% |
| w_o_queue_awareness | 64.43% +- 0.82 | 0.136 | 3.652 | 46.17 | +0.00% |
| w_o_semantic_prior | 64.43% +- 0.82 | 0.136 | 3.652 | 46.17 | +0.00% |
| w_o_semantics | 64.43% +- 0.82 | 0.136 | 3.652 | 46.17 | +0.00% |
| w_o_mobility_features | 64.37% +- 0.83 | 0.136 | 3.653 | 46.11 | -0.05% |
| w_o_partial_offloading | 49.92% +- 0.73 | 0.013 | 4.854 | 25.65 | -14.51% |

### Kisa Yorum

- `w_o_partial_offloading` success'i cok sert dusurmese de `p95 latency`yi belirgin bicimde kotulestiriyor; partial offloading katkisi daha cok tail-latency tarafinda gorunuyor.
- `w_o_mobility_features` en buyuk negatif etkiyi veriyor; bu da mobilite/distance bilgisinin karar kalitesi icin kritik oldugunu gosteriyor.
- `w_o_battery_awareness` varyantinin Full Model'den bir miktar iyi gorunmesi, mevcut reward tasariminda enerji disiplini ile success optimizasyonu arasinda gerilim olduguna isaret ediyor.
- `w_o_reward_shaping` ve `w_o_queue_awareness` sonuclarinin Full Model'e cok yakin olmasi, bu bilesenlerin etkisinin mevcut protokolde yeterince ayrisamamis olabilecegini dusunduruyor.

## Ablation Figure Galerisi

Bu bolum, algoritma ve kapsam bazli uretilmis tum sentetik ablation success-rate grafiklerini listeler.

### synthetic_ablation_a2c_multi_seed_evaluation_success_rate.png

![synthetic_ablation_a2c_multi_seed_evaluation_success_rate.png](../phase_5/figures/synthetic/ablation/synthetic_ablation_a2c_multi_seed_evaluation_success_rate.png)

### synthetic_ablation_a2c_multi_seed_retraining_success_rate.png

![synthetic_ablation_a2c_multi_seed_retraining_success_rate.png](../phase_5/figures/synthetic/ablation/synthetic_ablation_a2c_multi_seed_retraining_success_rate.png)

### synthetic_ablation_dqn_multi_seed_evaluation_success_rate.png

![synthetic_ablation_dqn_multi_seed_evaluation_success_rate.png](../phase_5/figures/synthetic/ablation/synthetic_ablation_dqn_multi_seed_evaluation_success_rate.png)

### synthetic_ablation_dqn_multi_seed_retraining_success_rate.png

![synthetic_ablation_dqn_multi_seed_retraining_success_rate.png](../phase_5/figures/synthetic/ablation/synthetic_ablation_dqn_multi_seed_retraining_success_rate.png)

### synthetic_ablation_ppo_multi_seed_evaluation_success_rate.png

![synthetic_ablation_ppo_multi_seed_evaluation_success_rate.png](../phase_5/figures/synthetic/ablation/synthetic_ablation_ppo_multi_seed_evaluation_success_rate.png)

### synthetic_ablation_ppo_multi_seed_retraining_success_rate.png

![synthetic_ablation_ppo_multi_seed_retraining_success_rate.png](../phase_5/figures/synthetic/ablation/synthetic_ablation_ppo_multi_seed_retraining_success_rate.png)


---
*Updated: 2026-05-10T19:53:15.578652*
