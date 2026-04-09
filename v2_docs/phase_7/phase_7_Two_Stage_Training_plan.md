Bkz. ortak kavram sozlugu: v2_docs/project_concepts_glossary.md

**Tarih:** 9 April 2026  
**Durum:** Faz 7 tamamlandi (`7.1 -> 7.4`)  
**Kanonik Teacher:** `teacher_contextual_reward_aligned`

---

## Faz 7 Amaci

Faz 7'nin amaci, PPO ajanini sifirdan RL ile baslatmak yerine once teacher-labeled dataset ile isitmak, sonra PPO fine-tuning ile daha hizli ve daha guclu bir policy elde etmektir.

---

## Kanonik Artefaktlar

- Oracle dataset: `results/raw/synthetic/pretraining/oracle_label_dataset.csv`
- Oracle summary: `v2_docs/phase_7/synthetic_oracle_label_summary.md`
- Kanonik supervised pretraining config: `configs/synthetic/supervised_pretraining.yaml`
- Kanonik supervised pretraining report: `v2_docs/phase_7/teacher_policy_sensitivity_report.md`
- Kanonik staged comparison config: `configs/synthetic/staged_training_comparison.yaml`
- Kanonik staged comparison report: `v2_docs/phase_7/teacher_policy_sensitivity_report.md`
- Teacher sensitivity report: `v2_docs/phase_7/teacher_policy_sensitivity_report.md`

---

## 7.1 Oracle Label Uretimi

- `teacher_latency_greedy`, `teacher_energy_greedy`, `teacher_balanced_semantic` ve `teacher_contextual_reward_aligned` ayni dataset icinde uretildi.
- Coverage-aware selection ve train rebalance ile tum teacher'larda `local / edge_25 / edge_50 / edge_75 / edge_100 / cloud` coverage'i saglandi.

---

## 7.2 Supervised Pretraining

Kanonik teacher secimi:
- `teacher_contextual_reward_aligned`

Kanonik sonuc:
- best epoch: `17`
- best validation accuracy: `82.67%`
- final test accuracy: `83.11%`
- checkpoint: `models/ppo/teacher_policy_pretrained/contextual_reward_aligned/ppo_pretrained.zip`

---

## 7.3 PPO Fine-Tuning vs Scratch PPO

Kanonik staged-training sonucu:
- `Scratch PPO`: `63.00%`
- `Pretrained + PPO`: `75.20%`
- success delta: `+12.20` puan
- pretrained kolu `%75` success esigine ulasir; scratch kolu bu batchte ulasamaz
- final pretrained policy `Edge %75` agirlikli kalir (`56.5%`) ve `Full Cloud`a cokmez

Teacher sensitivity yorumu:
- `teacher_latency_greedy` daha yuksek success delta verir, ancak final policy'yi `Full Cloud`a iter
- `teacher_contextual_reward_aligned` biraz daha dusuk success delta ile daha savunulabilir decision structure korur
- bu nedenle Faz 7 icin kanonik teacher olarak `teacher_contextual_reward_aligned` secildi

---

## Pretrained Checkpoint Kontrolu

Faz 7 sonrasinda ek bir dogrulama daha yapildi: supervised pretraining checkpoint'i fine-tuning oncesi dogrudan environment uzerinde test edildi.
Amac, `supervised accuracy` ile `fine-tuned RL success` arasindaki farkin retention problemi mi yoksa farkli metriklerden gelen dogal bir ayrim mi oldugunu ayirmakti.

Kanonik teacher sonucu:
- pretrained-only success: `74.47%`
- fine-tuned success: `75.20%`
- fark: `+0.73` puan

Sonuc:
- kanonik teacher kolunda fine-tuning teacher bilgisini dusurmemis, kucuk bir RL katkisi uretmistir
- Faz 8 oncesi asil teknik odak, success kaybi degil davranissal cesitlilik olmaya devam etmektedir

## 7.4 Kapanis Karari

- Faz 7 kanonik teacher olarak `teacher_contextual_reward_aligned` ile kapatildi
- teacher sensitivity ozetleri `v2_docs/phase_7/teacher_policy_sensitivity_report.md` icinde tek raporda toplandi
- generic / legacy Faz 7 artefaktlari temizlendi
- Faz 8 oncesi kabul edilen son durum: `Pretrained + PPO = 75.20%` ve `Scratch PPO = 63.00%`
- behavior tarafinda final pretrained policy `Edge %75` agirlikli kalarak `Full Cloud` collapse riskini kanonik kolda asmisti

## Faz 8'e Gecis Notu

Faz 8'e gecis, Faz 7'nin asil amacinin karsilandigi kabul edilerek yapilir. Two-stage training, graph-aware policy oncesinde hem warm-start hem de daha savunulabilir decision structure acisindan yeterli temel saglamistir.

## Faz 8 Oncesi Izlenecek Risk

- kanonik staged-training kolu `Full Cloud` yerine `Edge %75` agirlikli kalmistir
- bu, Faz 7 icin kabul edilebilir ve savunulabilir bir ilerlemedir
- ancak tam context-sensitive action diversity henuz tamamlanmis sayilmaz
- Faz 8 degerlendirmelerinde graph-aware policy'nin bu davranissal siniri kirip kiramadigi acikca izlenecektir

## Faz 9'a Devredilenler

Gelismis metrik, istatistiksel analiz ve GUI odakli sensitivity genisletmeleri Faz 9 altinda takip edilecektir.
