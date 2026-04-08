Bkz. ortak kavram sozlugu: v2_docs/project_concepts_glossary.md

# Faz 7 Report: Two-Stage Training

**Tarih:** 8 April 2026  
**Durum:** 7.1 ve 7.2 tamamlandi / 7.3 icin hazir  
**Kapsam:** oracle label uretimi, supervised pretraining, PPO fine-tuning ve `scratch vs pretrained` karsilastirmasi

---

## Faz 7 Hedefi

Bu fazda hedef, PPO ajaninin once ogretmen etiketlerle isitilmasi ve daha sonra RL ile fine-tune edilmesidir.
Amac, egitimi daha hizli yakinlastirmak, sample efficiency'yi iyilestirmek ve gerekiyorsa final performansi yukari tasimaktir.

---

## Faz 6 Sonundan Devralinan Durum

Faz 7'ye su zeminle girildi:
- Faz 5 sentetik bulgulari donduruldu
- Faz 6 trace-driven training, domain-shift ve hold-out test tamamlandi
- sentetik ve trace akislari artik ayri ama tekrar uretilebilir sekilde belgelendi
- mevcut RL egitim giris noktasi `src/training/train_agent.py` icinde hazir
- `src/training/pretrain_policy.py` oracle label uretimi ve supervised pretraining icin dolduruldu

---

## 7.1 Oracle Label Uretimi

Uretilen artefaktlar:
- ham dataset: `results/raw/synthetic/pretraining/oracle_label_dataset.csv`
- ozet rapor: `results/tables/synthetic_oracle_label_summary.md`
- config: `configs/synthetic/oracle_labeling.yaml`
- experiment scripti: `experiments/synthetic/generate_oracle_labels.py`

Dataset ozeti:
- toplam satir: `9000`
- objective'ler: `latency_oracle`, `energy_oracle`, `weighted_objective_oracle`
- split dagilimi:
  - train: `6300`
  - val: `1350`
  - test: `1350`

Kalibrasyon sonrasi bulgular:
- `latency_oracle`: `cloud` agirlikli (`97.50%`)
- `energy_oracle`: `cloud` agirlikli (`97.70%`)
- `weighted_objective_oracle`:
  - `edge_75`: `59.57%`
  - `cloud`: `39.37%`
  - diger aksiyonlar: dusuk ama gorunur seviyede

Yorum:
- `latency_oracle` ve `energy_oracle` analiz amacli tutulabilir
- supervised pretraining icin asil teacher policy `weighted_objective_oracle` oldu
- cloud bias kalibrasyonla kirildi ve daha kullanisli bir ogretmen davranisi elde edildi

---

## 7.2 Supervised Pretraining

Uretilen artefaktlar:
- pretrained checkpoint: `models/ppo/pretrained/ppo_weighted_oracle_pretrained.zip`
- metrics CSV: `results/raw/synthetic/pretraining/supervised_pretraining_metrics.csv`
- ozet rapor: `results/tables/supervised_pretraining_report.md`
- config: `configs/synthetic/supervised_pretraining.yaml`
- experiment scripti: `experiments/synthetic/run_supervised_pretraining.py`

Guncel sonuclar:
- objective: `weighted_objective_oracle`
- configured epoch count: `30`
- executed epoch count: `16`
- early stopping patience: `5`
- early stopping triggered: `yes`
- best epoch: `11`
- best validation accuracy: `78.00%`
- final train accuracy: `80.33%`
- final validation accuracy: `78.00%`
- final test accuracy: `80.22%`

Yorum:
- `15 epoch` yerine `30 epoch + early stopping` denenmesine ragmen en iyi sonuc yine `epoch 11` civarinda kaldi
- bu, pretraining tarafinin fazla kisitli kalmadigini; aksine bu teacher dataset uzerinde erken doyuma ulastigini gosteriyor
- yani daha uzun schedule kaliteyi artirmadi ama mevcut sonucun tesadufi olmadigini dogruladi

---

## Planlanan Alt Adimlar

1. Oracle / heuristic label uretimi  
   durum: tamamlandi ve kalibre edildi
2. Supervised pretraining  
   durum: tamamlandi
3. PPO fine-tuning  
   durum: siradaki adim
4. `PPO from scratch` vs `Pretrained + PPO` karsilastirmasi  
   durum: henuz baslamadi

---

## Henuz Tamamlanmayanlar

- pretrained PPO icin RL fine-tuning hatti kurulmadi
- `scratch PPO` ile ayni protokolde karsilastirma henuz uretilmedi
- convergence / sample efficiency tablosu henuz yazilmadi

---

## Faz 7'de Beklenen Ana Kanit

Faz 7 sonunda su soruya cevap verilmis olmali:

`staged training, ayni PPO ajanini sifirdan baslatmaya gore daha hizli, daha stabil veya daha basarili hale getiriyor mu?`

---

## Sonraki Dogru Adim

Bir sonraki teknik adim, pretrained checkpoint'i RL fine-tuning akisina baglamak ve ayni protokolde `scratch PPO` ile karsilastirmayi baslatmaktir.

