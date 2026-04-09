Bkz. ortak kavram sozlugu: v2_docs/project_concepts_glossary.md

**Tarih:** 9 April 2026  
**Durum:** 7.1, 7.2 ve 7.3 tamamlandi, 7.4 kapanis yorumu acik  
**Hedef:** oracle label -> supervised pretraining -> PPO fine-tuning zincirini calisir ve olculebilir hale getirmek

---

## Faz 7 Amaci

Faz 7'nin amaci, PPO ajanini her seferinde sifirdan RL ile baslatmak yerine once ogretmen niteliginde etiketlerle isitmak, sonra RL fine-tuning ile son policy'yi iyilestirmektir.

Bu asama, AgentVNE'deki staged training disiplininin bu projeye uyarlanmis halidir.

---

## Tamamlanan Asamalar

### 1. Oracle Label Uretimi

Tamamlananlar:
- `weighted_objective_oracle` teacher policy kalibre edildi
- `reward_aligned_oracle` eklendi
- `results/raw/synthetic/pretraining/oracle_label_dataset.csv` uretildi
- `v2_docs/phase_7/synthetic_oracle_label_summary.md` ile dagilim raporlandi

Kritik bulgu:
- Faz 7 icindeki anlamli teknik eksiklerden biri teacher objective ile RL reward hizasiydi
- bu nedenle reward-aligned teacher ayrica test edildi

### 2. Supervised Pretraining

Tamamlananlar:
- `src/training/pretrain_policy.py` icine supervised pretraining akisi eklendi
- `configs/synthetic/supervised_pretraining.yaml` ile tekrar uretilebilir config tanimlandi
- weighted ve reward-aligned teacher checkpointleri uretildi
- `results/raw/synthetic/pretraining/supervised_pretraining_metrics.csv` kaydedildi
- `v2_docs/phase_7/supervised_pretraining_report.md` yazildi

Kritik bulgu:
- weighted teacher daha yuksek supervised accuracy verdi
- reward-aligned teacher daha zor ogrenilen ama RL hedefiyle daha hizali bir ogretmen olarak davrandi

### 3. RL Fine-Tuning ve PPO Karsilastirmasi

Tamamlananlar:
- `results/raw/synthetic/staged_training/staged_training_comparison.csv` kaydedildi
- `results/raw/synthetic/staged_training/staged_training_progress.csv` ile ara adim performansi loglandi
- `v2_docs/phase_7/staged_training_comparison_report.md` ile final rapor yazildi
- action mapping ve action profili rapora eklendi
- staged-training'e ozgu ara metrikler eklendi:
  - deadline miss ratio
  - energy per success
  - step-to-75% success
  - best success during training
  - success curve AUC
  - success 95% CI
- reward-aligned teacher icin hafif policy anchoring denemesi yapildi

Kritik bulgular:
- weighted teacher + varsayilan fine-tuning: pretrained daha hizli ama finalde geride
- weighted teacher + dusuk LR: fark daraldi
- reward-aligned teacher + dusuk LR: final performans scratch ile esitlenebildi
- reward-aligned teacher + hafif anchoring: final parity korundu ve erken ogrenme avantaji netlestirildi
- reward-aligned teacher + `retention -> refinement` schedule: 3-seed protokolde final `success rate` ustunlugu gosterildi

Son yorum:
- staged training warm-start etkisi acisindan dogrulandi
- uygun schedule ile final `success rate` ustunlugu de gosterildi
- davranissal ayrisma tarafinda `full cloud` attractor etkisi hala surmektedir

Karsilastirma eksenleri:
- convergence hizi
- sample efficiency
- final success rate
- p95 latency
- avg energy
- deadline miss ratio
- energy per successful task
- success curve AUC

---

## Siradaki Asama

### 4. Faz 7 Kapanis Notu

Asil Faz 7 sorusu artik su sekilde cevaplanabilir:
- `PPO from scratch`
- `Pretrained + PPO`

Bu iki hat ayni sentetik protokolde karsilastirildi.
Kalan is, Faz 7 kapanis notunu yazmak ve bu bulguyu Faz 9'daki gelismis metrik paketiyle birlestirmektir.

---

## Done Kriteri

Faz 7 tamamlandi denebilmesi icin asgari olarak su kosullar aranacak:
- oracle label dataset'i tekrar uretilebilir olmali
- supervised pretraining calisir halde olmali
- pretrained checkpoint kaydedilmeli
- `scratch PPO` ve `Pretrained + PPO` ayni protokolde karsilastirilmali
- Faz 7 raporunda convergence veya sample efficiency farki acikca gosterilmeli

Durum:
- ilk dort kosul tamamlandi
- son madde de staged training raporuyla desteklendi
- resmi faz kapanisi icin sadece rapor son yorumu ve commit adimi kaldi

---

## Sonraki Dogru Adim

Bir sonraki dogru uygulama adimi, Faz 7.4 kapanis notunu yazmak ve staged-training bulgusunu Faz 9'da tekrar olculecek genis metrik paketiyle iliskilendirmektir.


## Yol Haritasi Notu

- `Gelismis Metrik ve Istatistiksel Analiz` paketi bu fazda unutulmus veya atlanmis degildir.
- Eski master-roadmap numaralandirmasinda Faz 7 olarak gecse de, guncel `task.md` akisi icinde Faz 9 olarak takip edilmektedir.
- Detayli kapsama notu `task.md` ve `TODO_ANTIGRAVITY_TASK_OFFLOADING_UPGRADE.md` icinde tutuluyor.


