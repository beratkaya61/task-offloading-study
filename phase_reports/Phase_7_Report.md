Bkz. ortak kavram sozlugu: v2_docs/project_concepts_glossary.md

# Faz 7 Report: Two-Stage Training

**Tarih:** 9 April 2026  
**Durum:** Faz 7 tamamlandi (`7.1 -> 7.4`)  
**Kanonik Teacher:** `teacher_contextual_reward_aligned`

---

## Faz 7 Hedefi

Bu fazda hedef, PPO ajaninin once teacher-labeled oracle dataset ile isitilmasi ve sonra PPO fine-tuning ile `scratch PPO`ya gore hem daha hizli hem de daha guclu hale getirilmesidir.

---

## 7.1 Oracle Label Uretimi

Kanonik artefaktlar:
- dataset: `results/raw/synthetic/pretraining/oracle_label_dataset.csv`
- summary: `v2_docs/phase_7/synthetic_oracle_label_summary.md`
- config: `configs/synthetic/oracle_labeling.yaml`

Son durum:
- coverage-aware selection ve train rebalance sonrasinda tum teacher'larda `local / edge_25 / edge_50 / edge_75 / edge_100 / cloud` coverage'i saglandi
- coverage problemi Faz 7 icin blocker olmaktan cikti

---

## 7.2 Supervised Pretraining

Kanonik artefaktlar:
- config: `configs/synthetic/supervised_pretraining.yaml`
- report: `v2_docs/phase_7/teacher_policy_sensitivity_report.md`
- checkpoint: `models/ppo/teacher_policy_pretrained/contextual_reward_aligned/ppo_pretrained.zip`

Kanonik sonuc:
- teacher policy: `teacher_contextual_reward_aligned`
- best epoch: `17`
- best validation accuracy: `82.67%`
- final test accuracy: `83.11%`

---

## 7.3 PPO Fine-Tuning ve Teacher Sensitivity

Kanonik staged-training artefaktlari:
- final CSV: `results/raw/synthetic/teacher_policy_sensitivity/contextual_reward_aligned/staged_training_comparison.csv`
- progress CSV: `results/raw/synthetic/teacher_policy_sensitivity/contextual_reward_aligned/staged_training_progress.csv`
- report: `v2_docs/phase_7/teacher_policy_sensitivity_report.md`
- teacher sensitivity summary: `results/raw/synthetic/teacher_policy_sensitivity/teacher_policy_sensitivity.csv`
- teacher sensitivity report: `v2_docs/phase_7/teacher_policy_sensitivity_report.md`

Kanonik sonuc (`teacher_contextual_reward_aligned`):
- `PPO_from_scratch`: `63.00% +- 2.46`
- `PPO_pretrained_finetuned`: `75.20% +- 2.50`
- success delta: `+12.20` puan
- p95 latency delta: `-0.696 s`
- QoE delta: `+15.68`
- energy per success delta: `-0.1197`
- `Pretrained + PPO`, `%75` success esigine ulasir; scratch kolu bu batchte ulasamaz

Teacher secim yorumu:
- `teacher_latency_greedy` en yuksek success delta'yi verir (`+14.33`), ancak final policy'yi `Full Cloud`a iter
- `teacher_contextual_reward_aligned` biraz daha dusuk success delta ile final policy'yi `Edge %75` agirlikli tutar (`56.5%`) ve Faz 8 oncesi daha savunulabilir decision structure korur
- bu nedenle Faz 7 icin kanonik teacher olarak `teacher_contextual_reward_aligned` secildi

---

## Pretrained Checkpoint Kontrolu

Faz 7 sonunda, supervised pretraining ile olculen yuksek accuracy degerlerinin RL fine-tuning sonrasi gercekten kaybolup kaybolmadigini anlamak icin ek bir kontrol yapildi.
Bu kontrolde her teacher policy icin `pretrained-only success` olculdu ve `fine-tuned success` ile yan yana kondu.

Kanonik teacher (`teacher_contextual_reward_aligned`) sonucu:
- supervised test accuracy: `83.11%`
- pretrained-only env success: `74.47%`
- fine-tuned env success: `75.20%`
- fark: `+0.73` puan

Yorum:
- burada karsilastirilan iki sayi ayni metrik degildir; `supervised accuracy` teacher etiket taklidini, `env success` ise gercek task basarisini olcer
- kanonik teacher kolunda fine-tuning sonrasi bir dusus degil, sinirli ama pozitif bir RL katkisi goruldu
- bu nedenle Faz 7 sonundaki ana risk, performance dususu degil; action diversity ve decision structure siniridir

Ayrintili tablo: `v2_docs/phase_7/teacher_policy_sensitivity_report.md` ve `v2_docs/phase_7/pretrained_checkpoint_evaluation_report.md`.

## Faz 7 Cikarimi

- staged training bu problemde gercek bir warm-start avantaji veriyor
- teacher secimi final success'i ciddi bicimde etkiliyor
- coverage problemi cozulduktan sonra ana ayrim, teacher'in final policy'yi hangi attractor'a ittigi oldu
- Faz 8 oncesi sadece daha yuksek success degil, daha savunulabilir decision structure da gerektigi icin Faz 7 kanonik kolu `teacher_contextual_reward_aligned` uzerinde sabitlendi

---

## 7.4 Kapanis Notu

Faz 7, teacher-policy sensitivity calismasi tamamlanip kanonik teacher secimi sabitlenerek kapatildi.

Kapanis karari:
- Faz 7 icin kanonik teacher `teacher_contextual_reward_aligned` olarak secildi
- teacher bazli butun ozetler `v2_docs/phase_7/teacher_policy_sensitivity_report.md` icinde tek raporda toplandi
- legacy Faz 7 artefaktlari ve eski generic checkpoint referanslari kaldirildi
- Faz 8 oncesi behavior coverage problemi kapatildi ve final karar yapisinin `Full Cloud` yerine `Edge %75` agirlikli kalmasi saglandi

Faz 7'nin nihai cikarimi:
- `Pretrained + PPO`, kanonik teacher ile `Scratch PPO`yu anlamli sekilde gecti
- staged training bu problemde gercek bir warm-start ve performance kazanci sagladi
- teacher seciminin final policy attractor'unu belirgin bicimde etkiledigi gosterildi
- Faz 8'e graph-aware policy asamasina gecis icin daha saglam bir davranissal temel hazirlandi

## Faz 8 Oncesi Risk Notu

Faz 7 kapatilmis olsa da Faz 8 oncesi bilincli olarak izlenecek bir teknik sinir vardir:
- kanonik teacher `teacher_contextual_reward_aligned` ile `Full Cloud` collapse kirilmistir
- ancak final pretrained policy hala tam anlamiyla context-sensitive bir action dagilimina yayilmis degildir
- karar yapisi agirlikli olarak `Edge %75` ekseninde toplanmaktadir

Bu nedenle Faz 8 boyunca sadece final success degil, action diversity ve decision structure da tekrar izlenecektir. Graph-aware policy asamasi, bu davranissal siniri daha da iyilestirip iyilestirmedigi acisindan degerlendirilecektir.

## Faz 9'a Devredilen Paket

Faz 7 kapatilirken bilincli olarak Faz 9'a devredilen kisim, gelismis metrik ve istatistiksel analiz paketidir. Bu paket Faz 7 sonucunu degistiren degil, Faz 7 cikarimini daha guclu ve daha savunulabilir hale getirecek tamamlayici asamadir.
