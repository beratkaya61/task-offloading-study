Bkz. ortak kavram sozlugu: v2_docs/project_concepts_glossary.md

# Config Layout

Bu klasor faz bazli duzenlenmistir.

- `phase_5/`
  - sentetik RL egitimi
  - sentetik policy evaluation
  - sentetik ablation
  - sentetik RL retraining
  - gercek veri omurgasinda Faz 5 ablation yeniden kosusu

- `phase_6/`
  - synthetic-trace RL egitimi
  - synthetic-trace domain-shift / holdout evaluation
  - real-data manifest
  - real-composite-trace RL training config

- `phase_7/`
  - oracle labeling
  - supervised pretraining
  - staged training comparison
  - teacher policy varyantlari

- `phase_8/`
  - graph supervised pretraining

Isim kurali:
- `synthetic_*` = yalnizca sentetik ortam
- `synthetic_trace_*` = sentetikten turetilmis trace-inspired akis
- `real_data_*` = Faz 5 icinde sentetik deneylerin gercek veri karsiliklari
- `real_composite_trace_*` = ham gercek kaynaklardan birlestirilmis trace omurgasi
- `raw_real_data_manifest.yaml` = ham gercek veri kaynaklari envanteri
