# Faz 5 - Sistematik Ablation Study Hazirlik Plani

## Faz 5 Hedefi
Semantik bilesenlerin (`LLM prior`, `reward shaping`, `confidence` vb.) task offloading sistemine katkisini sistematik olarak olcmek.

Ana soru:
`PPO_v2` basarisinin kaynagi nedir ve semantic bilesenlerin toplam katkisi ne kadardir?

---

## Yapilacaklar

### Adim 1: Ablation Konfigurasyonlarini Hazirlama
- [x] `configs/synthetic/ablation.yaml` olusturuldu
- [ ] Faz 4 sonuclariyla karsilastirma icin baseline kaydi korunacak

Planlanan ablation'lar:
| # | Ablation | Purpose | Expected Impact |
|---|---|---|---|
| 1 | Full Model | Baseline | ~62% success |
| 2 | w/o Semantics | LLM contribution | -5 to -15% |
| 3 | w/o Reward Shaping | Semantic reward impact | -3 to -8% |
| 4 | w/o Semantic Prior | Action bias from LLM | -2 to -5% |
| 5 | w/o Confidence | Confidence calibration | -1 to -3% |
| 6 | w/o Partial Offloading | Split offloading value | -10 to -20% |
| 7 | w/o Battery Awareness | Battery preservation | -5 to -10% |
| 8 | w/o Queue Awareness | Queue congestion avoidance | -3 to -8% |
| 9 | w/o Mobility Features | Distance/proximity impact | -2 to -5% |

### Adim 2: Ablation Runner Script Yazma
Hedef dosya:
- `experiments/synthetic/run_ablation_study.py`

### Adim 3: RL Environment Modifikasyonlari
Hedef dosya:
- `src/env/rl_env.py`

Gerekli degisiklikler:
- [ ] constructor icine ablation flags eklenmesi
- [ ] `_get_observation()` icinde semantic disable kontrolu
- [ ] reward akisinda `disable_reward_shaping` kontrolu
- [ ] confidence weighting'i acma/kapama
- [ ] state builder tarafinda feature masking

### Adim 4: Metrikleri Genisletme
Hedef dosya:
- `src/core/metrics.py`

Planlanan metrikler:
- [ ] `avg_latency`
- [ ] `p95_latency`
- [ ] `deadline_miss_ratio`
- [ ] `avg_energy`
- [ ] `fairness_score`
- [ ] `jitter`
- [ ] `qoe`

### Adim 5: Sonuclari Analiz Etme
Beklenen ciktillar:
- [ ] `v2_docs/phase_5/offloading_experiment_report.md`
- [ ] `results/figures/synthetic_ablation_<algorithm>_<scope>_success_rate.png`
- [ ] `results/raw/ablation_experiments.csv`

### Adim 6: Faz 5 Raporunu Yazma
Hedef dosya:
- `phase_reports/Phase_5_Report.md`

---

## Teknik Kararlar

### Karar 1: Her ablation icin modeli yeniden egitmek gerekli mi?
- Secenek A: Her ablation icin yeni modeli train et
- Secenek B: Tek egitilmis modeli kullanip bazi feature'lari devre disi birak

O donem icin tavsiye:
- once hizli tarama icin Secenek B
- sonra onemli varyantlari Secenek A ile validate et

### Karar 2: Istatistiksel testler yapilacak mi?
- [ ] Mean +- 95% CI
- [ ] Paired t-test vs Full Model
- [ ] p-value raporlamasi

### Karar 3: Diger baseline'larla ablation karsilastirmasi yapilacak mi?
Evet. Ama amac, RL politikasinin semantic bilesenlere ne kadar bagimli oldugunu gostermek olacak.

---

## Baglantilar
- Ablation config: `configs/synthetic/ablation.yaml`
- Onceki sonuc: `phase_reports/Phase_4_Report.md`
- Kanonik Faz 5 klasoru: `v2_docs/phase_5/`

---

## Faz 5 Tamamlama Kriterleri
- [ ] Ablation kosulari tamamlandi
- [ ] Sonuclar CSV'ye kaydedildi
- [ ] Mean +- 95% CI raporlandi
- [ ] Karsilastirma tablosu olusturuldu
- [ ] Impact chart'lari olusturuldu
- [ ] `Phase_5_Report.md` yazildi

---

## Not
Bu dosya Faz 5 baslamadan onceki plan notunu temsil eder. Faz 5'in gercek kapanis sonucu icin kanonik dokuman su dosyadir:
- `v2_docs/phase_5/offloading_experiment_report.md`
