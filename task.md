Bkz. ortak kavram sozlugu: v2_docs/project_concepts_glossary.md

Based on `TODO_ANTIGRAVITY_TASK_OFFLOADING_UPGRADE.md` and `AgentVNE`.

## Yol Haritasi Esleme Notu
- `TODO_ANTIGRAVITY_TASK_OFFLOADING_UPGRADE.md` ana master-roadmap olarak korunuyor.
- `task.md` ise fiili uygulama sirasina gore guncellenmis calisma listesidir.
- Bu nedenle TODO icindeki `Gelismis Metrik ve Istatistiksel Analiz` paketi eski numaralandirmada Faz 7 iken, guncel `task.md` akisi icinde Faz 9 olarak takip edilmektedir.
- Ayrintili esleme ve zorunlu metrik kapsami, `TODO_ANTIGRAVITY_TASK_OFFLOADING_UPGRADE.md` icindeki ilgili bolumde tutulmaktadir.

## Real-Data Duzeltme Notu (2026-05-09)
- Gercek veri omurgasi ve yeniden dogrulama kararlari `v2_docs/real_data_strategy.md` icinde merkezi olarak tutuluyor.
- Veri seti envanteri ve lokal indirme durumu `configs/phase_6/raw_real_data_manifest.yaml` icinde izlenecek.
- Mevcut Faz 5, Faz 7 ve Faz 8 sonuclari `synthetic/simulation-stage results` olarak etiketlenmelidir.
- Mevcut Faz 6 ciktisi raw real dataset dogrulamasi degil; `synthetic_didi / trace-inspired pipeline validation` olarak okunmalidir.
- Ana bilimsel iddia icin secilen real-data kaynaklari ve roller `v2_docs/real_data_strategy.md` icinde tutulacaktir.
- Real-data mode acildiginda sentetik fallback sessizce devreye girmeyecek; ham veri yoksa deney durdurulacaktir.

## Config ve Experiment Duzeni Notu (2026-05-10)
- `configs/` ve `experiments/` klasorleri artik veri-tipine gore degil, faz bazli okunacak sekilde duzenlenmistir.
- Ana klasorler: `phase_5`, `phase_6`, `phase_7`, `phase_8`.
- Veri rejimi farki dosya isimlerinde acikca yazilir: `synthetic_*`, `synthetic_trace_*`, `real_composite_trace_*`.
- `configs/README.md` ve `experiments/README.md` bu duzenin kisa haritasini verir.

## Results ve Test Duzeni Notu (2026-05-10)
- `results/` klasoru de ayni mantikla faz bazli okunur: `results/phase_5..8`.
- Ham deney CSV'leri `results/phase_*/metrics/`, gorseller `results/phase_*/figures/` altinda tutulur.
- `tests/` klasoru de faz bazli okunur; mevcut graph testleri `tests/phase_8/` altina tasinmistir.
- `logs/` klasoru tutulmaz; tekrar okunabilir tum deney artefaktlari `results/` altinda biriktirilir.

## Faz 1 - Reproducibility ve Kod Temizligi
- [x] 1.1 `src/baselines.py`, `src/evaluation.py`, `src/metrics.py`, `src/config.py`, `src/trace_loader.py`, `src/semantic_prior.py`, `src/pretrain_policy.py`, `src/utils/reproducibility.py`
- [x] 1.2 `configs/` and `results/` structure
- [x] 1.3 Seed helper for `numpy`, `random`, `torch`, `gymnasium`
- [x] 1.4 JSON/CSV logging infrastructure
- [x] 1.5 Phase 1 test and commit (user side)

## Faz 2 - Train Environment ve Gercek Simulasyon Hizalamasi
- [x] 2.1 Shared `src/state_builder.py`
- [x] 2.2 Simulator-backed reset in `rl_env.py`
- [x] 2.3 Reward components moved to `src/reward.py`
- [x] 2.4 Multi-step episode design
- [x] 2.5 Phase 2 test and commit (user side)

## Faz 3 - LLM Entegrasyonunu Gercek Katkiya Donustur
- [x] 3.1 6D action prior via `semantic_prior.py`
- [x] 3.2 Confidence integration
- [x] 3.3 Structured JSON output and logging
- [x] 3.4 Phase 3 test and commit (user side)

## Faz 4 - Baseline Ailesini Genislet
- [x] 4.1 LocalOnly, EdgeOnly, Random, GreedyLatency
- [x] 4.2 Genetic Algorithm baseline
- [x] 4.3 PPO v2, DQN, A2C scaffolding
- [x] 4.4 Common evaluation
- [x] 4.5 Phase 4 test/report

## Faz 5 - Sistematik Ablation Study
- [x] 5.1 Ablation config (`configs/ablation.yaml`)
- [x] 5.2 Ablation logs (90 run, Phase_5_Report.md)
- [x] 5.3 Anomali Giderimi ve Kalibrasyon
- [x] 5.4 Scientific Seal: Visuals (plots), Variance (StdDev) & State Pruning (noise reduction)
- [x] 5.5 Phase 5 final report update (Sealed with real metrics)
Not:
- Faz 5 sentetik ortamda kapatildi; gercek veri omurgasi uzerindeki zorunlu spot-check takibi `6R.5` maddesi altinda yapilacaktir.

## Faz 6 - Trace Pipeline / Synthetic-Didi Validation
- [x] 6.1 `trace_loader.py` implemented and wired into the trace pipeline
- [x] 6.2 Basic trace-to-task mapping via `src/core/trace_processor.py`
- [x] 6.3 Success Bonus (+100 sparse reward) Integration to reduce Semantic Dependency
- [x] 6.4 Adaptive/Dynamic Switching Overhead for Partial Offloading
- [x] 6.5 Domain shift analysis (Synthetic vs Trace results) [v2_docs/phase_6/trace_domain_shift_report.md ile tamamlandi]
- [x] 6.6 Phase 6 final test/report [v2_docs/phase_6/trace_holdout_test_report.md ve phase_reports/Phase_6_Report.md ile tamamlandi]

Not:
- 2026-05-09 kapsam duzeltmesi: Faz 6 teknik trace pipeline'i dogruladi, fakat mevcut lokal repo durumunda raw real dataset uzerinde nihai dogrulama yapilmis sayilmaz.
- `data/synthetic_trace/` altindaki mevcut splitler materyalize episode JSON dosyalaridir; trace isimleri `synthetic_didi` ailesinden geldigi icin bu sonuc real-data validated iddia olarak kullanilmayacaktir.
- `src/core/trace_loader.py` artik raw trace CSV ve kaydedilmis train/val/test episode split JSON dosyalarini yukleyebiliyor.
- Mevcut Faz 6 orchestrator'u (`experiments/phase_6/train_synthetic_trace_ppo.py`) trace hazirlama icin artik once `TraceLoader`, sonra `TraceProcessor` kullaniyor.
- `data/synthetic_trace/` altindaki episode JSON dosyalari yeniden kullanilabiliyor; raw trace yoksa processor tarafindaki fallback akisiyla yeni splitler uretilebiliyor.
- Trace config tarafinda `use_success_bonus: true` ve `success_bonus: 100.0` ile Faz 6 sparse success reward entegrasyonu acildi.
- `rl_env.py` icinde task boyutu, link kalitesi ve onceki aksiyon degisimine bagli dinamik `switching_overhead` eklendi.

## Ara Faz 6R - Real Data Recovery and Revalidation Gate
Not:
- Veri seti edinimi, manifest yonetimi, mapping duzeltmeleri ve real-data mode kurallari Faz 6 kapsaminda izlenecektir.
- Faz 8'in real-data tekrar kosusu bu fazda hazirlanan veri omurgasina dayanacaktir.
- Kullanilacak veri kaynaklari ve rolleri `v2_docs/real_data_strategy.md` ile `configs/phase_6/raw_real_data_manifest.yaml` icinde tutulur.
- Kod tarafinda ilk guard eklendi: real-data mode acildiginda manifest okunacak ve sentetik fallback yerine sert hata verilecektir. Ancak veri kaynaklari henuz lokal olarak mevcut olmadigi icin 6R maddeleri tamamlandi sayilmaz.
- Guncel inventory notu: `Glasgow MEC`, `UCI MEC execution-times` ve `Alibaba Cluster Trace` lokal olarak alindi ve temiz klasor yapisina indirildi. Opsiyonel tarafta `Google Cluster Trace` icin secondary-validation core subset, `Didi Gaia` icin ise sample-day mobility CSV'leri indirildi. Veri klasorlerinde yalnizca tutulacak veri dosyalari birakildi; zip ve gecici repo klasorleri temizlendi.
- Lokal veri profili artik `v2_docs/phase_6/real_data_inventory_report.md` icinde mevcut; bir sonraki teknik adim bu gozlenen kolonlardan real-data split builder cikarmaktir.
- Ilk real-data composite split builder artik calisiyor: `5000` task kaydi uretildi, `80/10/10` train/val/test splitleri `data/real_composite_trace/` altina yazildi ve `configs/phase_6/real_composite_trace_ppo_training.yaml` ile smoke-load dogrulandi.
- Kod mimarisi sadeleştirme notu: debug/synthetic kaynak yardimcilari ile real dataset okuyuculari `src/core/dataset_loader.py` icinde toplandi; `src/core/trace_loader.py` yalnizca materialized episode split / raw trace IO sorumlulugunu tasiyor. Real-data inventory ve readiness kontrolu de `experiments/phase_6/inspect_raw_real_datasets.py` altinda birlestirildi.

- [x] 6R.1 Real-data manifest ve lokal dataset envanteri olusturulsun
- [x] 6R.2 Real mode icin sessiz synthetic fallback kapatilsin
- [x] 6R.3 Secilen real-data kaynaklari ingest edilsin
- [x] 6R.4 Real-data train/val/test splitleri yeniden uretulsun ve trace mapping dokumani guncellensin
- [ ] 6R.5 Faz 5R real-data ablation spot-check yapilsin
  Faz 5 sentetik ortamda kapanmis olsa da, ana mekanizma bulgulari bilimsel gerceklik icin gercek veri omurgasi uzerinde en az spot-check seviyesinde yeniden sinanacaktir.
- [ ] 6R.6 Faz 7R real-trace staged-training yeniden kosulsun
- [ ] 6R.7 Faz 8 real-trace graph-vs-MLP karsilastirmasi yeniden kosulsun
- [ ] 6R.8 Phase raporlari guncellensin ve eski sentetik sonuclar ayri etiketle tutulsun

## Faz 7 - Two-Stage Training (AgentVNE Concept)
- [x] 7.1 Oracle / heuristic labels [results/phase_7/metrics/synthetic/pretraining/oracle_label_dataset.csv uretildi; kalibrasyon notu Phase_7_Report.md icine dusuldu]
- [x] 7.2 Imitation / supervised pretraining [teacher-policy sensitivity sonrasi kanonik teacher `teacher_contextual_reward_aligned` olarak sabitlendi; `models/ppo/teacher_policy_pretrained/contextual_reward_aligned/ppo_pretrained.zip` uretildi; best epoch `17`, val acc `82.67%`, test acc `83.11%`; ayrintilar `v2_docs/phase_7/teacher_policy_sensitivity_report.md` icinde tutuluyor]
- [x] 7.3 Fine-tune PPO vs scratch comparison [teacher-policy sensitivity tamamlandi; kanonik staged-training sonucu `teacher_contextual_reward_aligned` ile `v2_docs/phase_7/teacher_policy_sensitivity_report.md` dosyasina tasindi; `Pretrained + PPO = 75.20%` ve `Scratch PPO = 63.00%`; davranissal olarak final politika `Full Cloud` yerine `Edge %75` agirlikli kaldi. Tum teacher karsilastirmasi `v2_docs/phase_7/teacher_policy_sensitivity_report.md` icinde tutuluyor]
- [x] 7.4 Phase 7 test and commit [Faz 7 kanonik teacher secimi, teacher-policy sensitivity ozetinin tek raporda toplanmasi, legacy artefakt temizligi ve dokumantasyon hizalamasi tamamlandi; kapanis yorumu `phase_reports/Phase_7_Report.md` ve `v2_docs/phase_7/phase_7_Two_Stage_Training_plan.md` icine islendi]

## Faz 8 - Graph-Aware Policy Upgrade
Not:
- 2026-05-09 kapsam duzeltmesi: Faz 8 final karsilastirmasi graph mimarisini ve semantic prior fusion'i teknik olarak dogruladi; ancak bu karsilastirma henuz real-data backbone uzerinde yeniden kosulmamistir.
- Bu nedenle Faz 8 sonucu `synthetic/simulation-stage graph comparison` olarak raporlanacak; real-data tekrar kosusu Faz 6R kapsaminda hazirlanan veri omurgasi tamamlandiginda yapilacaktir.
- Faz 8'e gecis oncesi Faz 7'den kalan ana izleme notu, kanonik pretrained policy'nin `Edge %75` agirlikli kalmasidir. `Full Cloud` collapse kirilmis olsa da tam context-sensitive action diversity henuz nihai olarak cozulmus sayilmaz; Faz 8 boyunca bu davranissal sinir izlenecektir.
- Faz 8 baslangic plani `v2_docs/phase_8/phase_8_Graph_Aware_Policy_Upgrade_plan.md` icinde tutuluyor. Ilk uygulama sirasi: graph-state sozlesmesi, graph env/wrapper, GNN policy forward path, semantic prior fusion ve MLP-PPO vs graph-aware policy karsilastirmasidir.
- Faz 8'i hic bilmeyen biri icin "neden graph-aware policy, ne elde edecegiz, bize katkisi ne?" aciklamasi `v2_docs/phase_8/phase_8_explaination_of_studies.md` icinde tutuluyor. Faz 8 dokumanlarini okurken once bu aciklama, sonra teknik plan okunmalidir.
- Faz 8 zorunlu kapanis kapisi: `MLP-PPO`, `Pretrained MLP-PPO`, `GraphPolicy none`, `GraphPolicy late` ayni evaluator altinda karsilastirilmadan Faz 8 kapanmis sayilmayacak. Faz sonunda hikaye formatli kapanis ve fusion karsilastirma anlatimi ana aciklama dosyasinda tutulacak: `v2_docs/phase_8/phase_8_explaination_of_studies.md`.
- [x] 8.1 Graph state node and edge features [`src/env/graph_state_builder.py`, `tests/phase_8/test_graph_state_builder.py`; GraphState sozlesmesi `v2_docs/phase_8/phase_8_explaination_of_studies.md` icine tasindi; unit test: `python -m unittest tests.phase_8.test_graph_state_builder` OK]
- [x] 8.2 GNN policy implementation [PyTorch-only initial graph-aware policy forward path `src/agents/graph_policy.py` ile eklendi; `tests/phase_8/test_graph_policy.py`; combined unit test: `python -m unittest tests.phase_8.test_graph_state_builder tests.phase_8.test_graph_policy` OK]
- [x] 8.3 Semantic prior fusion [Graph policy fusion modes `none`, `input`, `late`, `input_late` olarak ayrildi; supervised graph warm-start hatti `src/training/pretrain_graph_policy.py` ile eklendi. Ilk `none` vs `late` sonucu sadece smoke/diagnostic bulgu olarak isaretlendi; final bilimsel iddia icin tek kanonik config `configs/phase_8/graph_supervised_pretraining.yaml` ve 5-seed mean/std/95% CI destekli `experiments/phase_8/run_graph_fusion_comparison.py` protokolu eklendi. Rapor kalabaligini onlemek icin fusion anlatimi ana aciklama dosyasinda tutulacak: `v2_docs/phase_8/phase_8_explaination_of_studies.md`]
- [x] 8.4 Phase 8 test and commit [Graph-vs-MLP evaluator koprusu `src/agents/graph_policy_evaluator.py` ile eklendi; `experiments/phase_8/run_graph_policy_comparison.py` ile `MLP-PPO`, `Pretrained MLP-PPO`, `GraphPolicy none`, `GraphPolicy late` ayni evaluator mantiginda karsilastirildi. Final env sonucu: `MLP-PPO=66.53%`, `Pretrained MLP-PPO=75.40%`, `GraphPolicy none=75.00%`, `GraphPolicy late=74.87%`. 5-seed supervised fusion protokolunde `late` fusion, `none`a gore teacher-imitation accuracy kazanci gosterdi (`82.76%` vs `74.04%`), ancak bu ustunluk final env kosusunda `GraphPolicy late > GraphPolicy none` seklinde korunmadi. Faz 8 kapanis raporu `phase_reports/Phase_8_Report.md` ve `v2_docs/phase_8/phase_8_explaination_of_studies.md` icine islendi; manual commit user tarafinda]

## Faz 9 - Advanced Metrics, Statistical Analysis and GUI
Not:
- Bu faz, eski master-roadmap icindeki "Gelismis Metrik ve Istatistiksel Analiz" paketinin guncel task.md karsiligidir.
- Faz 7.3 ve 7.4 tamamlandiktan sonra, Faz 7'den kalan zorunlu metrik genisletmeleri burada uygulanacaktir.
- Ozellikle staged-training karsilastirmasi (`PPO from scratch` vs `Pretrained + PPO`) bu fazdaki gelismis metrik ve istatistiksel analiz paketiyle yeniden raporlanacaktir.
- Real-data recovery tamamlanmadan Faz 9'daki nihai istatistiksel iddialar sentetik/simulation-stage olarak ayrilacak; real-data validated tablolar Faz 6R sonrasi uretilecektir.
- [ ] 9.1 Kanonik evaluator genisletmesi (`p99`, `deadline miss ratio`, `energy per success`, `battery depletion`, `queue waiting`, `decision overhead`)
- [ ] 9.2 Fairness/Jitter/QoE/Reward decomposition panelleri ve GUI entegrasyonu
- [ ] 9.3 5-seed protokolu, mean +- std, 95% CI ve uygun istatistiksel testler
- [ ] 9.4 Sonuc tablolarinin sentetik, trace ve staged-training karsilastirmalarina uygulanmasi
- [ ] 9.5 Teacher-policy sensitivity analizi (`teacher_latency_greedy`, `teacher_energy_greedy`, `teacher_balanced_semantic`, `teacher_contextual_reward_aligned`) ve staged-training sonuclarina etkisinin raporlanmasi
- [ ] 9.6 GUI experiment mode icinde trace replay / canliya yakin veri akisi secilirken teacher policy seciminin de acilabilir hale getirilmesi ve sonuc panellerinin buna gore guncellenmesi
- [ ] 9.7 Faz 7'den devralinan staged-training metrik eksiklerinin kapatilmasi
- [ ] 9.8 Phase 9 test and commit

## Faz 10 - LLM Self-Reflection & Experience Replay
- [ ] 10.1 Post-hoc analysis for bad decisions
- [ ] 10.2 Explanation bank as RAG/few-shot memory
- [ ] 10.3 Final technical documentation
- [ ] 10.4 Final test and commit










- Trace-to-task ceviri varsayimlari v2_docs/phase_6/trace_mapping_assumptions.md icinde merkezi olarak belgelendi.
- Domain-shift akisi `experiments/phase_6/evaluate_synthetic_trace_domain_shift.py` ve `configs/phase_6/synthetic_trace_domain_shift_evaluation.yaml` uzerinden calistirildi; guncel tablo `v2_docs/phase_6/trace_domain_shift_report.md` icinde tutuluyor.



