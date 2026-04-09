Bkz. ortak kavram sozlugu: v2_docs/project_concepts_glossary.md

# Faz 7 Aciklamasi: Two-Stage Training Calismasinin Uctan Uca Anlatimi

## Bu Dokuman Neden Var

Faz 7 teknik olarak cok verimli ilerledi, ancak ayni anda birden fazla deney varyanti calistigi icin buyuk resim zaman zaman kopuk gorundu.
Bu dokumanin amaci, Faz 7'yi hic bilmeyen birinin bile bastan sona anlayabilecegi aciklikta aciklamaktir.

Bu dokumanda su sorulari cevapliyoruz:

- Faz 7 neden var?
- `two-stage training` tam olarak ne demek?
- `oracle label dataset` nasil uretildi?
- `supervised pretraining` nasil yapildi?
- `RL fine-tuning` neden ikinci asama olarak gerekiyor?
- Faz 7'de neden iki farkli kol var?
- Bu iki kolu neyle kiyasliyoruz?
- Son durumda ne elde ettik?
- Hangi aciklar kapandi, hangileri hala dikkat istiyor?


---

## 1. Faz 7'nin Ana Amaci Nedir

Faz 7'nin ana sorusu su sekilde kurulmustur:

> PPO agent'i her seferinde sifirdan RL ile baslatmak yerine, once bir `teacher policy` ile isitip sonra RL ile gelistirirsek daha iyi bir sonuca ulasabilir miyiz?

Buradaki hedef iki parcaliydi:

1. `sample efficiency`yi arttirmak
2. mumkunse final `success rate` ve genel performansi da yukari tasimak

Yani Faz 7, yalnizca pretraining yapalim demiyor.
Asil soru su:

- `PPO from scratch` mi daha iyi?
- yoksa `Pretrained + PPO` mu daha iyi?

Bu nedenle Faz 7'nin sonunda aslinda iki farkli ogrenme stratejisini kiyasliyoruz.

---

## 2. Faz 7'ye Neden Ihtiyac Duyduk

Faz 5 ve Faz 6 sonunda su tablo ortaya cikti:

- PPO calisiyor
- sentetik ortamda anlamli sonuclar uretiyor
- trace-driven ortamda da genelleme gosterebiliyor
- fakat egitim hala pahali
- policy bazen cok basit bir davranisa, ozellikle tek bir baskin aksiyona, once `full cloud` daha sonra `edge_75` secimine, fazla kolay yakinlayabiliyor

Bu nedenle Faz 7'de su fikir test edildi:

Eger policy'ye RL oncesinde anlamli karar ornekleri gosterilirse, agent her seyi sadece reward deneme-yanilmasiyla ogrenmek zorunda kalmayabilir.

Bu, AgentVNE makalesindeki `staged training` fikrinin bu projeye uyarlanmis halidir.

---

## 3. Faz 7'deki Iki Ana Kol

Faz 7 gercekte iki kola ayrilir.
Bu cok onemlidir, cunku tum karsilastirma bunun uzerinden kurulur.

### Kol 1: `Scratch PPO Branch`

Bu kolun amaci su soruya cevap vermektir:

> PPO hicbir `teacher knowledge` almadan, yalnizca RL ile ne kadar iyi ogreniyor?

Bu kolun akisi:

- random initialization
- normal PPO training
- evaluation

Bu kol bizim referans cizgimizdir.
Yani `baseline learning strategy` budur.

### Kol 2: `Pretrained + PPO Branch`

Bu kolun amaci su soruya cevap vermektir:

> PPO once `teacher labels` ile isitilip sonra RL ile gelistirilirse ne oluyor?

Bu kolun akisi:

- `oracle label generation`
- `supervised pretraining`
- `retention stage`
- `refinement stage`
- evaluation

Bu kol bizim `two-stage training strategy`mizdir.

---

## 3.1 Faz 7 Visual Flow

Asagidaki diyagram Faz 7'nin tum akisini tek bakista gosterir.

```mermaid
flowchart TD
    A["Synthetic Environment States"] --> B["Oracle Label Generation"]
    B --> C["oracle_label_dataset.csv"]
    C --> D["Supervised Pretraining"]
    D --> E["pretrained PPO checkpoint"]

    A --> S["Scratch PPO Training"]
    E --> F["Retention Stage"]
    F --> G["Refinement Stage"]

    S --> H["Common Evaluation Pipeline"]
    G --> H

    H --> I["staged_training_comparison.csv"]
    H --> J["staged_training_progress.csv"]
    H --> K["teacher_policy_sensitivity_report.md"]
```

### Diyagramin Okunusu

Bu diyagram su siranin calistigini anlatir:

1. Sentetik environment state'leri uretilir.
2. Bu state'ler uzerinden bir `teacher policy` ile `oracle labels` uretilir.
3. Bu etiketler `oracle label dataset` haline getirilir.
4. PPO policy network bu dataset ile `supervised pretraining` yapar.
5. Buradan bir `pretrained PPO checkpoint` elde edilir.
6. Daha sonra `pretrained` kolu RL ile `fine-tuning` yapar.
7. Bu `fine-tuning`, iki alt RL asamasina ayrilir:
   - `retention stage`: teacher bilgisinin bir anda silinmemesi icin daha dusuk `learning rate` ve hafif `policy anchoring` ile yapilan ilk RL evresidir.
   - `refinement stage`: policy artik RL reward'a daha rahat uyum saglasin diye anchoring'in gevsetildigi veya kapatildigi ikinci RL evresidir.
8. Ayni anda baska bir kolda `scratch PPO` normal sekilde egitilir.
9. En sonda iki kol ayni `common evaluation pipeline` ile olculur.
10. Karsilastirma CSV'leri ve markdown raporu uretilir.

Kisa not: Diyagramda ayri bir `fine-tuning` kutusu yok gibi gorunebilir. Bunun sebebi, `Retention Stage -> Refinement Stage` zincirinin birlikte zaten `fine-tuning` adimini temsil etmesidir. Yani `fine-tuning`, bu iki RL asamasinin ust kavramidir.
---

## 4. `Oracle Label Dataset` Nedir

Faz 7'de pretraining icin kullandigimiz veri seti harici bir dataset degildir.
Bu dataset proje icinde sentetik environment ustunde uretilmistir.

Guncel artifact:

- `results/raw/synthetic/pretraining/oracle_label_dataset.csv`

Bu dataset su script ile uretilir:

- `experiments/synthetic/generate_oracle_labels.py`

Asil implementation burada bulunur:

- `src/training/pretrain_policy.py`

Kullandigi config:

- `configs/synthetic/oracle_labeling.yaml`

Yani bu dataset, bizim offloading problemimize gore olusturulmus, problem-ozel bir `teacher dataset`tir.

---

## 5. Bu Dataset Nasil Uretildi

Pipeline su sekilde calisir:

1. Sentetik environment kuruluyor:

   - `max_steps = 50`
   - `num_edge_servers = 3`
   - `num_devices = 5`
2. Her episode ve her step icin mevcut observation okunuyor.
3. Tum valid actions sirayla simule ediliyor.
4. Her action icin su degerler tahmin ediliyor:

   - `predicted_delay`
   - `predicted_energy`
   - `predicted_reward`
   - `deadline_met`
   - `semantic_target`
   - `semantic_match`
   - `edge_energy_cost`
   - `edge_energy_ratio`
   - `switching_overhead`
5. Sonra secilen `teacher policy` bu adaylari skorliyor.
6. En iyi action `selected_action_id` olarak seciliyor.
7. En iyi action ile ikinci en iyi action arasindaki fark `teacher_margin` olarak kaydediliyor.

Bu margin degeri kritik onemdedir.
Cunku teacher'in kararinin ne kadar net oldugunu anlatir.

---

## 6. Dataset'in Icindeki Alanlar Nelerdir

Dataset artik yalnizca isimize yarayan ve adindan ne oldugu anlasilan kolonlari tutar.

Teacher policy kolonlari:

- `teacher_policy`
- `selected_action_id`
- `selected_action_name`
- `teacher_score`
- `teacher_margin`

Prediction / semantic kolonlari:

- `predicted_delay`
- `predicted_energy`
- `predicted_reward`
- `deadline_met`
- `semantic_target`
- `semantic_match`
- `priority_score`
- `semantic_confidence`

Split ve ornek kimligi:

- `split`
- `episode_id`
- `step_id`

State kolonlari:

- `state_snr_norm`
- `state_task_size_norm`
- `state_cpu_cycles_norm`
- `state_battery_norm`
- `state_edge_load_norm`
- `state_edge_energy_norm`
- `prior_local`
- `prior_edge_25`
- `prior_edge_50`
- `prior_edge_75`
- `prior_edge_full`
- `prior_cloud`

Bu isimlendirme su problemi cozer:
- `obs_0`, `obs_1` gibi kolonlarda neyin ne oldugu anlasilmiyordu
- `objective` ve `oracle_action` gibi isimler de teacher policy mantigini yeterince acik gostermiyordu

Guncel teacher policy adlari:

- `teacher_latency_greedy`
- `teacher_energy_greedy`
- `teacher_balanced_semantic`
- `teacher_contextual_reward_aligned`

Guncel sayilar:

- total sample: `12000`
- `train = 8400`
- `val = 1800`
- `test = 1800`

---

## 6.1 `Teacher Policy` Nedir

`Teacher policy`, belirli bir state goruldugunde "bu durumda en mantikli action hangisi olmali?" sorusuna otomatik cevap veren ogretmen karar mekanizmasidir.

Bu projede `teacher policy` disaridan alinmis hazir bir model degildir. Bizim sentetik environment icinde, her state icin tum gecerli action'lari tek tek skorlayan ve en iyi adayi sececek sekilde olusturulmus bir kararlayicidir.

Yani pratikte su isi yapar:
- mevcut observation'i alir
- butun valid action'lari dener veya skorlar
- her action icin delay, energy, reward, deadline ve semantic etkileri tahmin eder
- secilen teacher policy scoring mantigina gore en iyi action'i `selected_action_id` olarak etiketler

Bu nedenle `teacher policy`, PPO'ya "dogru cevap" vermekten cok, "iyi bir baslangic davranisi" gosteren ogretmen rolundedir.
## 7. Teacher Variants Neleri Temsil Ediyor

### `teacher_latency_greedy`

Bu teacher daha dusuk delay'e odaklanir.
Teacher-policy sensitivity sonucunda final success artisi en yuksek teacher bu oldu.
Ancak staged PPO sonunda final politika agirlikli olarak `Full Cloud` tarafina kaydi.
Bu nedenle saf success acisindan guclu, davranissal olarak ise riskli bir teacher olarak okundu.

### `teacher_energy_greedy`

Bu teacher daha dusuk energy cost'a odaklanir.
Pretraining ve fine-tuning sonrasi `Scratch PPO` uzerine ciksa da final karar yapisi yine `Full Cloud` eksenine yaklasti.
Bu nedenle enerji odakli yarar gosterse de Faz 7'nin context-sensitive behavior hedefi icin yeterli gorulmedi.

### `teacher_balanced_semantic`

Bu teacher birden fazla unsuru birlikte degerlendirir:

- normalized delay
- normalized energy
- deadline pressure
- cloud penalty
- battery risk
- edge risk
- semantic alignment
- partial offloading bonuses

Bu teacher dengeli bir ara varyant sagladi.
Ancak final staged PPO sonucu yine `Full Cloud` agirligini tam kiramadi.
Bu nedenle kanonik secim olarak alinmadi.

### `teacher_contextual_reward_aligned`

Bu teacher, secim mantigini downstream RL reward'una daha yakin kurar; buna ek olarak dataset coverage'i action-aware sekilde dengelenmis ve train splitte tum action'lar gorunur hale getirilmis varyanttir.
Faz 7'nin kanonik teacher secimi budur.

Bu teacher'in secilme nedeni su iki kosulu birlikte saglamasidir:
- `Pretrained + PPO > Scratch PPO`
- final politika `Full Cloud`a cokmeden `Edge %75` agirlikli ama daha savunulabilir bir karar yapisinda kalir

Faz 7'nin ana bulgularindan biri burada netlesmistir:

> En iyi teacher her zaman en yuksek supervised accuracy veren ya da en yuksek final success artisini veren teacher degildir. Teacher secimi, final decision structure ile birlikte okunmalidir.

---

## 8. `oracle_margin` Neden Bu Kadar Onemli Oldu

Zaman icinde su anlasildi:

Her teacher sample ayni kalitede degil.

Bazi state'lerde teacher'in secimi cok nettir.
Bazilarinda ise birden fazla action birbirine yakindir.

Bu nedenle `min_margin` filtresi eklendi.
Bu filtre ile daha kararsiz label'lar ayiklanabilir.

Ama Faz 7'de burada ince bir bulgu daha elde edildi:

- tarihsel olarak reward-aligned teacher varyantlarinda margin buyudukce dataset daha dar ve daha tek-aksiyonlu hale gelebiliyordu; bu bulgu daha sonra contextual reward-aligned teacher tasarimina geciste dikkate alindi

Bu da bize sunu ogretti:

> `high-margin` her objective icin otomatik olarak daha iyi teacher anlamina gelmez.

Yani margin filtresi objective'e gore yorumlanmalidir.

---

## 8.1 `PPO Policy` Tam Olarak Nedir

Bu kavram cok kritik oldugu icin acik yazmak gerekir.

`policy`, agent'in mevcut state'i gorup hangi action'i sececegine karar veren neural network yapisidir.
Bizim projede bu network, observation vector'u alir ve su action'lardan biri icin skor / olasilik uretir:

- `local`
- `edge_25`
- `edge_50`
- `edge_75`
- `edge_100`
- `cloud`

Dolayisiyla `supervised pretraining ile isitilmis PPO policy` dedigimiz sey, PPO ajaninin karar veren network agirliklarinin teacher labels ile onceden egitilmis halidir.

Bu asama sonunda elimizde final policy yoktur. Elimizde sadece RL oncesinde daha iyi bir baslangic davranisi olan PPO karar agi vardir.

## 9. `Supervised Pretraining` Bu Projede Nasil Calisiyor

Implementation:

- `src/training/pretrain_policy.py`

Entrypoint:

- `experiments/synthetic/run_supervised_pretraining.py`

Config:

- `configs/synthetic/supervised_pretraining.yaml`

Bu asamada PPO policy network bir siniflandirici gibi egitilir.

Girdi:

- observation vector

Hedef:

- `selected_action_id`

Loss:

- cross-entropy over action logits

Kontrol edilen hyperparameter'lar:

- `epochs`
- `batch_size`
- `learning_rate`
- `early_stopping_patience`
- `early_stopping_min_delta`
- `objective`
- `min_margin`

Bu asama sonunda normal PPO biciminde bir checkpoint kaydedilir.
Bu checkpoint, teacher labels ile isitilmis PPO policy agirliklarini tasir ve bir sonraki RL `fine-tuning` asamasinin girisi olur.

---

## 10. `Supervised Pretraining` Asamasinda Neler Gorduk

Faz 7 boyunca pretraining tarafinda birden fazla teacher varyanti denendi.
Bu nedenle burada tek bir accuracy sayisi degil, iki farkli seviyeyi ayirmak gerekir:

1. `supervised accuracy`
   - teacher etiketlerinin ne kadar iyi taklit edildigi
2. `downstream RL utility`
   - bu teacher ile isitilan checkpointin daha sonra PPO fine-tuning sirasinda ne kadar iyi bir baslangic verdigi

Teacher-policy sensitivity sonucunda guncel teacher'lar icin supervised test accuracy tablosu su sekilde oldu:

- `teacher_latency_greedy`: `77.56%`
- `teacher_contextual_reward_aligned`: `83.11%`
- `teacher_balanced_semantic`: `83.56%`
- `teacher_energy_greedy`: `79.33%`

Buradan cikan ana ders su oldu:

- en yuksek supervised accuracy tek basina en iyi kanonik teacher secimi anlamina gelmedi
- pretraining asamasinda iyi taklit edilen bir teacher, fine-tuning sonunda davranissal olarak istenmeyen bir attractor'a da goturebilir
- bu nedenle Faz 7'de `imitability` ile `objective alignment` birlikte degerlendirildi

Kanonik teacher olan `teacher_contextual_reward_aligned` bu noktada kritik bir denge kurdu:
- supervised pretraining accuracy yuksek kaldi (`83.11%` test)
- fine-tuning sonrasi `Scratch PPO`yu gecti
- final karar yapisi `Full Cloud` yerine `Edge %75` agirlikli kaldi

Yani Faz 7 icin dogru soru sunun gibi okunmadi:
- hangi teacher en yuksek accuracy verdi?

Bunun yerine su soru soruldu:
- hangi teacher, pretraining + fine-tuning zincirinin sonunda hem performans hem de decision structure acisindan daha savunulabilir bir sonuc verdi?

---

## 11. Pretraining Sonrasinda `Fine-Tuning` Neden Gerekli

Pretraining bittiginde elimizde hazir policy yoktur.
Elimizde sadece daha iyi bir baslangic noktasi vardir.

Daha sonra RL `fine-tuning` baslar:

- ayni environment ailesi
- PPO optimization
- gercek reward
- scratch PPO ile ayni evaluation mantigi

Buradaki esas soru su olur:

> Bu pretrained initialization, random initialization'a gore daha iyi bir learning trajectory uretiyor mu?

---

## 12. Faz 7'nin Kiyaslamasi Tam Olarak Ne Ile Ne Arasinda Kuruldu

Son karsilastirma yalnizca iki model dosyasinin karsilastirilmasi degildir.
Aslinda iki farkli ogrenme stratejisi kiyaslanir:

- `learning from scratch`
- `learning with teacher warm-start`

Model isimleriyle soyle gorunur:

- `PPO_from_scratch`
- `PPO_pretrained_finetuned`

Kullanilan ana artifacts:

- `results/raw/synthetic/teacher_policy_sensitivity/contextual_reward_aligned/staged_training_comparison.csv`
- `results/raw/synthetic/teacher_policy_sensitivity/contextual_reward_aligned/staged_training_progress.csv`
- `v2_docs/phase_7/teacher_policy_sensitivity_report.md`

Olculen baslica metrikler:

- `success rate`
- `avg reward`
- `p95 latency`
- `avg energy`
- `QoE`
- `deadline miss ratio`
- `energy per successful task`
- `step-to-75% success`
- `success curve AUC`
- `success 95% CI`
- action profile

---

## 13. "Pretrained PPO Daha Hizli Isiniyor" Ifadesi Tam Olarak Ne Demek

Bu ifadeyle kastimiz sudur:

> Pretrained policy, scratch PPO'ya gore daha az RL step ile kullanisli bir success bolgesine ulasir.

Bu projede bunu su metrik ile olctuk:

- `step-to-75% success`

Yani bu ifade su anlama gelir:

- pretrained policy ise yarar seviyeye daha erken geliyor
- daha hizli `warm start` aliyor
- ayni budget icinde daha erken verimli hale gelebiliyor

Bu ifade otomatik olarak su demek degildir:

- finalde kesinlikle daha iyi olacak
- latency mutlaka dusuk olacak
- energy mutlaka daha iyi olacak

Yani `faster heating up` = `better early learning`, ama `guaranteed better final result` degildir.

---

## 14. Faz 7'nin Son Durumu: Kanonik Sonuc Nedir

Evet. Faz 7 sonunda teacher-policy sensitivity tamamlandi ve kanonik kol su sekilde sabitlendi:

- kanonik teacher: `teacher_contextual_reward_aligned`
- kanonik pretrained checkpoint: `models/ppo/teacher_policy_pretrained/contextual_reward_aligned/ppo_pretrained.zip`
- kanonik staged comparison: `results/raw/synthetic/teacher_policy_sensitivity/contextual_reward_aligned/staged_training_comparison.csv`

Guncel 3-seed sonuc:

- `PPO_from_scratch`
  - success: `63.00% +- 2.46`
  - p95 latency: `3.664 +- 0.057`
  - avg energy: `0.1380 +- 0.0077`
  - QoE: `44.68 +- 2.67`
- `PPO_pretrained_finetuned`
  - success: `75.20% +- 2.50`
  - p95 latency: `2.968 +- 0.025`
  - avg energy: `0.0749 +- 0.0043`
  - QoE: `60.36 +- 1.72`

Bunun anlami su:

- staged training artik sadece parity saglamiyor
- final `success rate` tarafinda da anlamli bir fark uretiyor
- kanonik kolda final davranis `Full Cloud`a degil, `Edge %75` agirlikli bir profile gidiyor

---

## 14.1 Bu Sonucu Ureten Kanonik Branch Hangisiydi

Bu noktadaki kanonik branch su oldu:

- selected teacher policy: `teacher_contextual_reward_aligned`
- pretrained checkpoint: `models/ppo/teacher_policy_pretrained/contextual_reward_aligned/ppo_pretrained.zip`
- retention stage: `10000` step, `learning_rate = 5e-05`, hafif `policy anchoring`
- refinement stage: `20000` step, `learning_rate = 1.5e-04`, dusuk agirlikli anchoring devam ettirilerek

Yani final kanonik sonuc, Faz 7'nin son teacher-policy sensitivity ve staged fine-tuning schedule iterasyonundan geldi.

---

## 14.2 Bu Sonuc Gereksinimi Tam Olarak Karsiliyor mu

Buyuk olcude evet.

Faz 7'nin cekirdek gereksinimi suydu:
- `Pretrained + PPO`, `Scratch PPO`'dan daha hizli ogrenebilmeli
- ve final `success rate`te de onu gecebilmeliydi

Su an bu iki hedef de karsilanmis durumda:
- daha hizli ogrenme sinyali goruldu
- final `success rate`te `75.20% > 63.00%` farki gosterildi

Ama hala tam cozulmemis bir davranissal sinir var:
- final politika `Full Cloud`a cokmuyor, bu iyi
- fakat karar yapisi hala agirlikli olarak `Edge %75` ekseninde toplaniyor
- yani davranissal cesitlilik artmis olsa da tam `context-sensitive action diversity` seviyesine ulasilmis degil

Bu ne anlama gelir:
- Faz 7'nin `staged training works` kismini guclu sekilde destekliyoruz
- ama `staged training bizi tamamen zengin ve baglama duyarli offloading davranisina goturdu` demek icin henuz erken

Dolayisiyla Faz 7, performans hedefini karsiliyor; asik kalan ana teknik konu ise davranissal cesitlilik olarak Faz 8'e devrediliyor.

---

## 15. Bu Son Sonuc Nasil Yorumlanmali

Dogru yorum sunun kombinasyonudur:

1. `Pretrained + PPO`, erken ogrenmede daha hizli ilerliyor.
2. Kanonik teacher ile final `success rate`te de `scratch PPO`yu geciyor.
3. Fine-tuning teacher bilgisini bozup basariyi dusurmuyor; tersine kucuk ama pozitif bir RL katkisi veriyor.
4. `full cloud` attractor kirildi, ancak politika hala agirlikli olarak `edge_75` etrafinda toplaniyor.

Yani Faz 7 artik su noktaya geldi:

- `sample efficiency advantage` gosterildi
- final success ustunlugu gosterildi
- fine-tuningin teacher bilgisini bozmadigi gosterildi
- fakat policy behavior tarafinda hala tam yapisal ayrisma yok

Bu, Faz 7'nin artik deneysel olarak savunulabilir ve Faz 8'e tasinabilir bir asamaya geldigini gosterir.

---

## 16. Sonucu Iyilestirmek Icin Faz 7 Icinde Neler Denendi

Faz 7 tek deney olmadi; iteratif bir `research loop` haline geldi.

Denenen ana rafinmanlar:

1. eski `weighted / reward-aligned` teacher varyantlari ile ilk pretraining denemeleri
2. daha dusuk fine-tuning learning rate denemeleri
3. hafif ve agirlikli `policy anchoring` denemeleri
4. action coverage-aware dataset rebalance adimi
5. tum teacher policyler icin `teacher-policy sensitivity` karsilastirmasi
6. kanonik kol icin `retention -> refinement` two-stage fine-tuning schedule
7. `pretrained-only evaluation` ile fine-tuning katkisini ayiran ek kontrol

En son guclu adimlar su oldu:

- coverage-aware teacher dataset kurulmasi
- `teacher_contextual_reward_aligned` secimi
- `retention -> refinement` schedule ile staged PPO fine-tuning
- `pretrained-only` ve `fine-tuned` environment success tablosunun birlikte okunmasi

Bu zincir sayesinde Faz 7'nin son kanonik sonucu, hem performance hem de yorumlanabilirlik acisindan daha savunulabilir hale geldi.

---

## 17. Faz 7 Bize Simdiden Neyi Kanitladi

Faz 7 artik su bulgulari guvenle soyleyebilecek noktada:

1. Tam bir `oracle -> pretraining -> fine-tuning -> comparison` pipeline kuruldu.
2. `teacher policy` secimi kritik.
3. `objective alignment` ve `imitability` farkli kavramlar.
4. Two-stage training olculebilir bir `warm-start` etkisi sagliyor.
5. Uygun schedule ile final `success rate` avantajina da ulasilabiliyor.
6. Hala dikkat edilmesi gereken ana sinir `action collapse` ve `edge_75` attractor etkisi.

Bu nedenle Faz 7, artik yalnizca "guzel fikir" seviyesinde degil; deneysel olarak savunulabilir bir asamaya gelmis durumda.

---

## 18. Faz 7'yi En Sade Dille Ozetlersek

Projeyi ilk kez goren biri icin Faz 7'nin ozet cumlesi sunlar olur:

- Once problem icin `teacher labels` urettik.
- Sonra PPO'yu bu etiketlerle isitacak sekilde egittik.
- Daha sonra ayni PPO'yu RL ile gelistirdik.
- Sonunda bunu `PPO from scratch` ile kiyasladik.
- Ilk denemelerde sadece erken-learning avantaji gorduk.
- Son `retention -> refinement` schedule ile final `success rate`te de ustunluk sinyali aldik.

Yani Faz 7'nin en guncel hikayesi su:

> `Two-stage training`, bu projede hem daha hizli ogrenme hem de uygun schedule ile daha yuksek final success icin anlamli bir yol olarak gorunmeye basladi.

---

## 19. Bu Faz Hangi Sirayla Okunmali

Faz 7'yi en temiz sirada okumak icin su akisi kullan:

1. `v2_docs/phase_7/phase_7_explaination_of_studies.md`
2. `v2_docs/phase_7/phase_7_Two_Stage_Training_plan.md`
3. `phase_reports/Phase_7_Report.md`
4. `v2_docs/phase_7/synthetic_oracle_label_summary.md`
5. `v2_docs/phase_7/teacher_policy_sensitivity_report.md`
6. `v2_docs/phase_7/pretrained_checkpoint_evaluation_report.md`

Bu siralama seni su cizgide tasir:

- kavramsal cerceve
- plan
- implementation sonucu
- destekleyici artifacts






---

## 20. Neden Pretrained-Only Evaluation Yapildi

Faz 7 sonunda su soru dogal olarak ortaya cikti:
- supervised pretraining test accuracy neden yuksekken, fine-tuning sonrasi gordugumuz success rate daha dusuk bir yuzde gibi okunuyor?

Bu sorunun cevabi, ayni sayilari karsilastirmiyor olmamizdir.
- `supervised accuracy` = teacher etiket taklidi
- `pretrained-only success` = supervised-pretrained checkpointin environment basarisi
- `fine-tuned success` = PPO ile RL guncellemesi sonrasi environment basarisi

Bu nedenle ek olarak `pretrained-only evaluation` yapildi ve teacher policy bazinda tablo cikarildi.
Ayrintili rapor:
- `v2_docs/phase_7/pretrained_checkpoint_evaluation_report.md`
- `results/raw/synthetic/teacher_policy_sensitivity/pretrained_checkpoint_evaluation.csv`

## 21. Faz 7'nin Nihai Sonucu

Faz 7, teacher-policy sensitivity tamamlanip kanonik teacher secimi sabitlenerek kapatilmistir.

Nihai Faz 7 sonucu:
- kanonik teacher: `teacher_contextual_reward_aligned`
- kanonik pretrained checkpoint: `models/ppo/teacher_policy_pretrained/contextual_reward_aligned/ppo_pretrained.zip`
- `Scratch PPO`: `63.00%`
- `Pretrained + PPO`: `75.20%`
- final karar yapisi: `Edge %75` agirlikli, `Full Cloud` degil

Bu nedenle Faz 7 icin ana iddia su sekilde sabitlenmistir:

> Two-stage training bu projede yalnizca ogrenme hizini degil, dogru teacher secildiginde final performansi ve decision structure kalitesini de iyilestirebilir.

Teacher bazli ayrintili karsilastirma tek kaynak olarak `v2_docs/phase_7/teacher_policy_sensitivity_report.md` icinde tutulur.
