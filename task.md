Bkz. ortak kavram sozlugu: v2_docs/project_concepts_glossary.md

Based on `TODO_ANTIGRAVITY_TASK_OFFLOADING_UPGRADE.md` and `AgentVNE`.

## Yol Haritasi Esleme Notu
- `TODO_ANTIGRAVITY_TASK_OFFLOADING_UPGRADE.md` ana master-roadmap olarak korunuyor.
- `task.md` ise fiili uygulama sirasina gore guncellenmis calisma listesidir.
- Bu nedenle TODO icindeki `Gelismis Metrik ve Istatistiksel Analiz` paketi eski numaralandirmada Faz 7 iken, guncel `task.md` akisi icinde Faz 9 olarak takip edilmektedir.
- Ayrintili esleme ve zorunlu metrik kapsami, `TODO_ANTIGRAVITY_TASK_OFFLOADING_UPGRADE.md` icindeki ilgili bolumde tutulmaktadir.

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

## Faz 6 - Gercek Veri / Trace-Driven Deney Paketi
- [x] 6.1 `trace_loader.py` implemented and wired into the trace pipeline
- [x] 6.2 Basic trace-to-task mapping via `src/core/trace_processor.py`
- [x] 6.3 Success Bonus (+100 sparse reward) Integration to reduce Semantic Dependency
- [x] 6.4 Adaptive/Dynamic Switching Overhead for Partial Offloading
- [x] 6.5 Domain shift analysis (Synthetic vs Trace results) [results/tables/trace_domain_shift_report.md ile tamamlandi]
- [x] 6.6 Phase 6 final test/report [results/tables/trace_holdout_test_report.md ve phase_reports/Phase_6_Report.md ile tamamlandi]

Not:
- `src/core/trace_loader.py` artik raw trace CSV ve kaydedilmis train/val/test episode split JSON dosyalarini yukleyebiliyor.
- Mevcut Faz 6 orchestrator'u (`experiments/trace/train_ppo.py`) trace hazirlama icin artik once `TraceLoader`, sonra `TraceProcessor` kullaniyor.
- `data/traces/` altindaki episode JSON dosyalari yeniden kullanilabiliyor; raw trace yoksa processor tarafindaki fallback akisiyla yeni splitler uretilebiliyor.
- Trace config tarafinda `use_success_bonus: true` ve `success_bonus: 100.0` ile Faz 6 sparse success reward entegrasyonu acildi.
- `rl_env.py` icinde task boyutu, link kalitesi ve onceki aksiyon degisimine bagli dinamik `switching_overhead` eklendi.

## Faz 7 - Two-Stage Training (AgentVNE Concept)
- [x] 7.1 Oracle / heuristic labels [results/raw/synthetic/pretraining/oracle_label_dataset.csv uretildi; kalibrasyon notu Phase_7_Report.md icine dusuldu]
- [x] 7.2 Imitation / supervised pretraining [models/ppo/pretrained/ppo_weighted_oracle_pretrained.zip uretildi; best epoch 11, val acc 78.00%, test acc 80.22%, 30 epoch config early stopping ile 16 epochta durdu]
- [ ] 7.3 Fine-tune PPO vs scratch comparison
- [ ] 7.4 Phase 7 test and commit

## Faz 8 - Graph-Aware Policy Upgrade
- [ ] 8.1 Graph state node and edge features
- [ ] 8.2 GNN policy implementation
- [ ] 8.3 Semantic prior fusion
- [ ] 8.4 Phase 8 test and commit

## Faz 9 - Advanced Metrics, Statistical Analysis and GUI
Not:
- Bu faz, eski master-roadmap icindeki "Gelismis Metrik ve Istatistiksel Analiz" paketinin guncel task.md karsiligidir.
- Faz 7.3 ve 7.4 tamamlandiktan sonra, Faz 7'den kalan zorunlu metrik genisletmeleri burada uygulanacaktir.
- Ozellikle staged-training karsilastirmasi (`PPO from scratch` vs `Pretrained + PPO`) bu fazdaki gelismis metrik ve istatistiksel analiz paketiyle yeniden raporlanacaktir.
- [ ] 9.1 Kanonik evaluator genisletmesi (`p99`, `deadline miss ratio`, `energy per success`, `battery depletion`, `queue waiting`, `decision overhead`)
- [ ] 9.2 Fairness/Jitter/QoE/Reward decomposition panelleri ve GUI entegrasyonu
- [ ] 9.3 5-seed protokolu, mean +- std, 95% CI ve uygun istatistiksel testler
- [ ] 9.4 Sonuc tablolarinin sentetik, trace ve staged-training karsilastirmalarina uygulanmasi
- [ ] 9.5 Faz 7'den devralinan staged-training metrik eksiklerinin kapatilmasi
- [ ] 9.6 Phase 9 test and commit

## Faz 10 - LLM Self-Reflection & Experience Replay
- [ ] 10.1 Post-hoc analysis for bad decisions
- [ ] 10.2 Explanation bank as RAG/few-shot memory
- [ ] 10.3 Final technical documentation
- [ ] 10.4 Final test and commit










- Trace-to-task ceviri varsayimlari v2_docs/trace_mapping_assumptions.md icinde merkezi olarak belgelendi.
- Domain-shift akisi experiments/trace/evaluate_domain_shift.py ve configs/trace/domain_shift_evaluation.yaml uzerinden calistirildi; guncel tablo results/tables/trace_domain_shift_report.md icinde tutuluyor.




