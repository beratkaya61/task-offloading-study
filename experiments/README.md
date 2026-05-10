Bkz. ortak kavram sozlugu: v2_docs/project_concepts_glossary.md

# Experiment Layout

Bu klasor de config yapisiyla ayni mantikta faz bazli duzenlenmistir.

- `phase_5/`
  - sentetik RL retraining
  - sentetik policy evaluation
  - sentetik ablation kosulari ve plot uretimi

- `phase_6/`
  - ortak trace RL egitim orkestrasyonu
  - trace RL kosu giris noktasi
  - real-composite-trace ablation runner'i
  - synthetic-trace domain-shift / holdout evaluation
  - ham gercek veri envanter denetimi
  - real composite trace build

- `phase_7/`
  - oracle label uretimi
  - supervised pretraining
  - staged training comparison
  - teacher policy sensitivity
  - pretrained checkpoint evaluation

- `phase_8/`
  - graph warm-start
  - graph fusion comparison
  - graph-vs-MLP policy comparison
  - graph state gorsellestirme

Kisa okuma:
- once faz klasorune bak
- sonra script adindaki `synthetic`, `synthetic_trace` veya `real_data` etiketinden veri rejimini oku
