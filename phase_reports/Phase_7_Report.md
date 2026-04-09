Bkz. ortak kavram sozlugu: v2_docs/project_concepts_glossary.md

# Faz 7 Report: Two-Stage Training

**Tarih:** 9 April 2026  
**Durum:** 7.1, 7.2 ve 7.3 tamamlandi / 7.4 acik  
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
- ozet rapor: `v2_docs/phase_7/synthetic_oracle_label_summary.md`
- config: `configs/synthetic/oracle_labeling.yaml`
- experiment scripti: `experiments/synthetic/generate_oracle_labels.py`

Dataset ozeti:
- toplam satir: `12000`
- objective'ler: `latency_oracle`, `energy_oracle`, `weighted_objective_oracle`, `reward_aligned_oracle`
- split dagilimi:
  - train: `8400`
  - val: `1800`
  - test: `1800`

Kalibrasyon sonrasi bulgular:
- `latency_oracle`: `cloud` agirlikli
- `energy_oracle`: `cloud` agirlikli
- `weighted_objective_oracle`: daha dengeli, `edge_75` agirlikli
- `reward_aligned_oracle`: RL reward ile daha dogrudan hizali ogretmen secenegi olarak eklendi

Yorum:
- Faz 7 icindeki ana teknik bosluk teacher objective ile RL reward arasindaki uyumdu
- bu nedenle reward-aligned teacher ayri olarak denenip reward mantigina daha yakin bir ogretmen uretildi

---

## 7.2 Supervised Pretraining

Uretilen artefaktlar:
- weighted teacher checkpoint: `models/ppo/pretrained/ppo_weighted_oracle_pretrained.zip`
- reward-aligned teacher checkpoint: `models/ppo/pretrained/ppo_reward_aligned_pretrained.zip`
- metrics CSV: `results/raw/synthetic/pretraining/supervised_pretraining_metrics.csv`
- ozet rapor: `v2_docs/phase_7/supervised_pretraining_report.md`
- config: `configs/synthetic/supervised_pretraining.yaml`
- experiment scripti: `experiments/synthetic/run_supervised_pretraining.py`

Guncel sonuclar:
- weighted teacher ile en iyi sonuc:
  - best validation accuracy: `78.00%`
  - final test accuracy: `80.22%`
- reward-aligned teacher ile:
  - min margin filter: `10.0`
  - best validation accuracy: `59.51%`
  - final test accuracy: `61.03%`

Yorum:
- reward-aligned teacher, reward mantigina daha yakin ama supervised taklit acisindan daha zor bir hedef cikardi
- bu da teacher objective ile reward uyumunun tek basina yeterli olmadigini; teacher davranisinin ogrenilebilir olmasinin da kritik oldugunu gosterdi

---

## 7.3 PPO Staged Training Karsilastirmasi

Uretilen artefaktlar:
- final sonuc CSV'si: `results/raw/synthetic/staged_training/staged_training_comparison.csv`
- ilerleme logu: `results/raw/synthetic/staged_training/staged_training_progress.csv`
- ozet rapor: `v2_docs/phase_7/staged_training_comparison_report.md`
- checkpointler:
  - `models/ppo/staged_training/scratch/`
  - `models/ppo/staged_training/pretrained/`

Calistirilan iterasyonlar:
1. weighted teacher + varsayilan fine-tuning
2. weighted teacher + dusuk LR fine-tuning (`1e-4`)
3. reward-aligned teacher + dusuk LR fine-tuning (`1e-4`)
4. reward-aligned teacher + policy anchoring (agresif schedule)
5. reward-aligned teacher + policy anchoring (hafif schedule)
6. reward-aligned teacher + `retention -> refinement` schedule

Ana bulgular:
- iterasyon 1: pretrained daha hizli isinmis ama finalde geride kalmistir
- iterasyon 2: fark daralmistir
- iterasyon 3: final metriklerde scratch ile esitlik yakalanmistir
- iterasyon 4: anchoring fazla agresif oldugunda bir seed `edge_75` tarafinda sapmis ve final kalite dusmustur
- iterasyon 5: hafif anchoring ile final parity korunurken erken ogrenme avantaji daha net hale gelmistir

Son iterasyonun net sonucu:
- Kullanilan teacher policy: `reward_aligned_oracle`
- Kullanilan pretrained checkpoint: `models/ppo/pretrained/ppo_reward_aligned_pretrained.zip`
- Kullanilan staged fine-tuning schedule:
  - retention: `10000` step, `learning_rate = 5e-05`, hafif `policy anchoring`
  - refinement: `20000` step, `learning_rate = 1.5e-04`, anchoring kapali
- `PPO_from_scratch`
  - success: `83.67% +- 0.76`
  - p95 latency: `2.006 +- 0.029`
  - avg energy: `0.0140 +- 0.0016`
  - QoE: `73.64 +- 0.90`
- `PPO_pretrained_finetuned`
  - success: `84.60% +- 0.80`
  - p95 latency: `2.010 +- 0.018`
  - avg energy: `0.0146 +- 0.0022`
  - QoE: `74.55 +- 0.87`

Faz 7.3'e ozgu ara metrikler:
- deadline miss ratio
- energy per success
- step-to-75% success
- best success during training
- success curve AUC
- success 95% CI

Yorum:
- `Pretrained + PPO`, `%75` success esigine ortalama `11667` stepte ulasiyor
- `Scratch PPO`, ayni esige ortalama `18333` stepte ulasiyor
- `success curve AUC` karsilastirmasi bu batchte scratch lehine hafif daha yuksek kaldi: `0.6443` vs `0.6363`
- buna ragmen final `success rate` farki pretrained lehine dondu: `+0.93` puan
- `QoE` de pretrained lehine `+0.91` artis gosterdi
- iki tarafin da finalde `full cloud` agirlikli politikaya yakinlamasi, env/reward yapisinin hala ayni cekim noktasina ittigini gosteriyor

Cikarim:
- Faz 7 amacinin `sample efficiency` kismi desteklendi
- Faz 7 amacinin `final performansi yukari tasima` kismi icin de ilk net pozitif sinyal alindi
- sonraki bilimsel soru artik yalnizca "staged training faydali mi?" degil, "bu ustunlugu daha buyuk ve daha davranissal olarak farkli hale nasil getiririz?" sorusudur

---

## Siradaki Asamalar

### 4. Faz 7 Kapanis Testi ve Son Yorum

Kalan is:
- Faz 7'nin kapanis notu yazilacak
- staged training sonucunun final yorumu netlestirilecek
- Faz 9'da tekrar ele alinacak gelismis metrik ve istatistik paketi icin devralma notu korunacak

---

## Done Kriteri

Faz 7 tamamlandi denebilmesi icin asgari olarak su kosullar aranacak:
- oracle label dataset'i tekrar uretilebilir olmali
- supervised pretraining calisir halde olmali
- pretrained checkpoint kaydedilmeli
- `scratch PPO` ve `Pretrained + PPO` ayni protokolde karsilastirilmali
- Faz 7 raporunda convergence veya sample efficiency farki acikca gosterilmeli

Durum:
- butun teknik kosullar saglandi
- son kapanis yorumu ve faz tamamlama notu icin `7.4` acik tutuluyor

---

## Sonraki Dogru Adim

Bir sonraki dogru uygulama adimi, Faz 7.4 kapanis notunu yazmak ve staged-training bulgusunu Faz 9'da genisletilecek zorunlu metrik paketiyle iliskilendirmektir.


