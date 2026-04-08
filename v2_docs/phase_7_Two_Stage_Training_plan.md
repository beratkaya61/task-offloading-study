Bkz. ortak kavram sozlugu: v2_docs/project_concepts_glossary.md

**Tarih:** 8 April 2026  
**Durum:** 7.1 ve 7.2 tamamlandi  
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
- `results/raw/synthetic/pretraining/oracle_label_dataset.csv` uretildi
- `results/tables/synthetic_oracle_label_summary.md` ile dagilim raporlandi

Kritik bulgu:
- kalibrasyon sonrasinda `weighted_objective_oracle` cloud agirlikli bir ogretmenden cikti
- `edge_75` agirlikli daha dengeli bir teacher policy elde edildi

### 2. Supervised Pretraining

Tamamlananlar:
- `src/training/pretrain_policy.py` icine supervised pretraining akisi eklendi
- `configs/synthetic/supervised_pretraining.yaml` ile tekrar uretilebilir config tanimlandi
- `models/ppo/pretrained/ppo_weighted_oracle_pretrained.zip` uretildi
- `results/raw/synthetic/pretraining/supervised_pretraining_metrics.csv` kaydedildi
- `results/tables/supervised_pretraining_report.md` yazildi

Kritik bulgu:
- `30 epoch + early stopping` ile kosulmasina ragmen en iyi validation sonucu `epoch 11` civarinda kaldi
- best validation accuracy: `78.00%`
- final test accuracy: `80.22%`
- early stopping `16` epoch sonunda devreye girdi

Yorum:
- pretraining asamasi artik calisir ve tekrar uretilebilir durumda
- bir sonraki darbo gaz teacher dataset kalitesi degil, staged RL karsilastirmasi oldu

---

## Siradaki Asamalar

### 3. RL Fine-Tuning

Pretrained policy daha sonra RL ile fine-tune edilecek.
Karsilastirma icin ayni ortamda `scratch PPO` kosusu da tutulacak.

Karsilastirma eksenleri:
- convergence hizi
- sample efficiency
- final success rate
- p95 latency
- avg energy

### 4. Faz 7 Karsilastirma Tablosu

Asil Faz 7 sorusu su olacak:
- `PPO from scratch`
- `Pretrained + PPO`

Bu iki hat ayni protokolle kosulup ayni raporda karsilastirilacak.

---

## Done Kriteri

Faz 7 tamamlandi denebilmesi icin asgari olarak su kosullar aranacak:
- oracle label dataset'i tekrar uretilebilir olmali
- supervised pretraining calisir halde olmali
- pretrained checkpoint kaydedilmeli
- `scratch PPO` ve `Pretrained + PPO` ayni protokolde karsilastirilmali
- Faz 7 raporunda convergence veya sample efficiency farki acikca gosterilmeli

---

## Sonraki Dogru Adim

Bir sonraki dogru uygulama adimi, pretrained checkpoint'i RL fine-tuning akisina baglamak ve ayni sentetik protokolde `scratch PPO` ile yan yana kosmaktir.


## Yol Haritasi Notu

- `Gelismis Metrik ve Istatistiksel Analiz` paketi bu fazda unutulmus veya atlanmis degildir.
- Eski master-roadmap numaralandirmasinda Faz 7 olarak gecse de, guncel `task.md` akisi icinde Faz 9 olarak takip edilmektedir.
- Detayli kapsama notu `task.md` ve `TODO_ANTIGRAVITY_TASK_OFFLOADING_UPGRADE.md` icinde tutuluyor.
