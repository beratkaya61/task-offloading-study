Bkz. ortak kavram sozlugu: v2_docs/project_concepts_glossary.md

# Results Layout

Bu klasor deney artefaktlarini faz bazli tutar.

- `phase_5/`
  - `metrics/synthetic/`: sentetik RL retraining, policy evaluation ve ablation CSV'leri
  - `metrics/real_data/`: gercek veri RL retraining, policy evaluation ve ablation CSV'leri
  - `figures/synthetic/`: sentetik ablation gorselleri
  - `figures/real_data/`: gercek veri ablation gorselleri

- `phase_6/`
  - `metrics/synthetic_trace/`: synthetic-trace training, domain-shift ve holdout CSV'leri
  - `metrics/real_composite_trace/`: real-composite trace egitimi ve yeniden-kosular icin CSV'ler

- `phase_7/`
  - `metrics/synthetic/pretraining/`: oracle label datasetleri
  - `metrics/synthetic/teacher_policy_sensitivity/`: supervised pretraining ve staged-training CSV'leri

- `phase_8/`
  - `figures/`: graph-state gorselleri
  - `metrics/`: sadece acikca CSV yazdirilan graph warm-start / fusion olcumleri

Okuma kurali:
- once `phase_*` klasorune bak
- sonra `metrics` veya `figures` ayrimini oku
- veri rejimini `synthetic`, `synthetic_trace` veya `real_data` dizin adindan takip et
