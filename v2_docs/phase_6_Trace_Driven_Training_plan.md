Bkz. ortak kavram sozlugu: v2_docs/project_concepts_glossary.md

**Tarih:** 8 April 2026  
**Durum:** aktif gelistirme asamasi  
**Hedef:** Faz 5 sentetik bulgularini trace-driven ortamda yeniden sinamak ve gercek artefaktlarla dogrulamak

---

## Mevcut Durum

Tamamlanan temel Faz 6 adimlari:
- `src/core/trace_loader.py` implement edildi
- `experiments/trace/train_ppo.py` loader -> processor akisini kullaniyor
- `Success Bonus` trace reward akisina eklendi
- dinamik `switching_overhead` modeli eklendi

Hala acik kalan ana maddeler:
- domain-shift evaluation
- final trace training kosusu
- checkpoint + metrics CSV + rapor kapanisi

---

## Faz 5'ten Devreden Bulgular

- `mobility_features` sentetik tarafta en guclu bilesendi
- reward shaping etkisi tamamen kapanmadi, trace tarafinda yeniden test edilmeli
- partial offloading degerli ama etkisi ortama gore degisebilir
- edge enerji modeli trace yuklerinde yeniden sinanmali

---

## Siradaki Isler

### 1. Trace mapping varsayimlarini netlestirmek
- `size_bits`, `cpu_cycles`, `deadline`, `task_type` eslesmesini belgelemek
- Gerekirse kisa bir assumptions dokumani eklemek

### 2. Domain-shift analizi eklemek
- synthetic train -> trace test
- trace train -> synthetic test
- Faz 5 sentetik ozeti ile Faz 6 trace ozeti arasinda karsilastirmali tablo cikarmak

### 3. Trace PPO egitimini kosturmak
- `results/raw/trace_training/trace_training_metrics.csv`
- `models/ppo/trace_training/ppo_v3_trace_best.zip`

### 4. Faz 6 raporunu kapatmak
- Gercek artefakt yollarini yazmak
- Nicel sonuclari rapora tasimak

---

## Basari Kriteri

- Trace training hata vermeden tamamlanmali
- Artefaktlar repo icinde tekrar uretilebilir olmali
- Faz 5'ten gelen en az bir ana bulgu trace tarafinda dogrulanmali veya acikca curutulmeli
- Faz 6 raporu gercek sonuclarla kapanmali




## Yeni Not
- Trace mapping varsayimlari artik v2_docs/trace_mapping_assumptions.md icinde merkezi olarak tutuluyor.
- Domain-shift evaluation icin configs/trace/domain_shift_evaluation.yaml ve experiments/trace/evaluate_domain_shift.py eklendi.
- Bu adim, Faz 6 kapatilmadan once synthetic-train -> trace-test ve trace-train -> synthetic-test tablosunu uretmek icin kullanilacak.

