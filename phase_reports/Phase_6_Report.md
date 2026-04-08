Bkz. ortak kavram sozlugu: v2_docs/project_concepts_glossary.md

# Faz 6 Report: Trace-Driven Training

**Tarih:** 08 April 2026  
**Durum:** tamamlandi / Faz 7 icin hazir  
**Kapsam:** trace-driven PPO egitimi, domain-shift degerlendirmesi, hold-out test ve Faz 5 -> Faz 6 bulgu baginin kurulmasi

---

## Ozet

Faz 6'da sentetik ortamdan cikilip trace tabanli episode splitleri ile PPO modeli yeniden egitildi.
Bu asamanin amaci, Faz 5'te sentetik tarafta gorulen bulgularin trace verisi uzerinde ne kadar tasinabildigini gormekti.

Bu faz sonunda su zincir tamamlandi:
- trace training kosusu basariyla tamamlandi,
- yeni checkpoint `models/ppo/trace_training/ppo_v3_trace_best.zip` olarak uretildi,
- training metrics `results/raw/trace/training/trace_training_metrics.csv` altina yazildi,
- synthetic -> trace ve trace -> synthetic domain-shift tablosu uretildi,
- `test_episodes.json` uzerinde ayri final hold-out evaluation kosturuldu,
- Faz 6 raporu gercek artefaktlarla temiz ve okunur hale getirildi.

Bu nedenle Faz 6 artik acik bir is listesi degil, kapanmis bir trace-driven deney paketi olarak okunabilir.

---

## Kullanilan Veri ve Splitler

Trace pipeline su dosyalari kullanir:
- egitim: `data/traces/train_episodes.json`
- validation: `data/traces/val_episodes.json`
- hold-out test: `data/traces/test_episodes.json`

Bu rapordaki trace training sonucu, `train_episodes.json` ile egitim ve `val_episodes.json` ile ara dogrulama mantigina dayanir.
Final kapanis kontrolu ise ayri olarak `test_episodes.json` uzerinde yapilmistir.

---

## Trace Training Sonucu

Kaynak artefaktlar:
- checkpoint: `models/ppo/trace_training/ppo_v3_trace_best.zip`
- training metrics: `results/raw/trace/training/trace_training_metrics.csv`

Trace training CSV ozetine gore:
- episode sayisi: `532`
- egitim boyunca ortalama success rate: `98.59%`
- egitim boyunca ortalama delay: `0.3702 s`
- egitim boyunca ortalama enerji: `0.0318`
- episode bazli success araligi: `92.00% - 100.00%`

Validation tarafinda kaydedilen ana sonuc:
- average success rate (validation): `99.20%`

Bu sayilar, trace ortaminda PPO'nun hizli sekilde istikrarli bir policy ogrendigini gosteriyor.

---

## Final Split Karsilastirmasi ve Hold-Out Test

Kaynak artefaktlar:
- CSV: `results/raw/trace/holdout/trace_holdout_evaluation.csv`
- rapor: `results/tables/trace_holdout_test_report.md`

Ayni trace checkpoint icin `train`, `val` ve `test` splitleri birlikte olculdu:

| Split | Episode Count | Mean Success | Std | Min | Max | Mean Delay | Mean Energy |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 80 | 99.38% | 1.08 | 96.00% | 100.00% | 0.3245 s | 0.0128 |
| val | 10 | 99.00% | 1.00 | 98.00% | 100.00% | 0.3453 s | 0.0207 |
| test | 10 | 99.60% | 0.80 | 98.00% | 100.00% | 0.3317 s | 0.0128 |

Gap analizi:
- Train - Val success gap: `+0.38` puan
- Val - Test success gap: `-0.60` puan
- Train - Test success gap: `-0.22` puan

Bu sonuc kritik oneme sahip, cunku validation'da iyi gorunen modelin ayrilmis test split uzerinde de ayni seviyede kaldigini gosteriyor.
Validation ve test sonuclari birbirine cok yakin kaldigi, train-test farki da sinirli oldugu icin belirgin bir overfitting izi gorulmuyor.

---

## Trace Uzerinde Faz 5 Spot Check

Trace validation sirasinda Faz 5'ten gelen iki temel bilesen hizli bir kontrol ile tekrar sinandi:

| Variant | Validation Success |
|---|---:|
| `full_model` | 98.40% |
| `no_reward_shaping` | 98.20% |
| `no_partial_offloading` | 96.60% |

Ilk okuma:
- `partial_offloading` trace tarafinda da faydali gorunuyor,
- `reward_shaping` farki ise trace validation'da daha sinirli,
- yani Faz 5'te sentetik tarafta kritik gorunen bilesenlerin trace tarafinda siddeti degisebiliyor.

---

## Domain-Shift Sonuclari

Kaynak artefaktlar:
- CSV: `results/raw/trace/domain_shift/trace_domain_shift_evaluation.csv`
- rapor: `results/tables/trace_domain_shift_report.md`

| Train Domain | Test Domain | Model | Success Rate | P95 Latency | Avg Energy | Dominant Action |
|---|---|---|---:|---:|---:|---:|
| synthetic | trace | PPO | 99.60% | 0.5618 | 0.0255 | 3 |
| trace | synthetic | PPO | 53.20% | 4.6883 | 0.0101 | 4 |

Yorum:
- `synthetic -> trace` yonu sasirtici derecede guclu cikti.
- `trace -> synthetic` yonu ise belirgin sekilde zayif.
- Bu, Faz 6'da artik yalnizca domain shift degil, ayni zamanda `cross-domain asymmetry` gozledigimizi gosteriyor.

Baska bir ifadeyle, sentetik tarafta ogrenilen policy trace ortamina tasinabiliyor; fakat trace tarafinda ogrenilen davranis sentetik env'e geri dondugunde ayni sekilde calismiyor.
Bu fark, iki ortam arasinda sadece veri kaynagi degil, karar kalibi ve action bias acisindan da asimetri oldugunu dusunduruyor.

---

## Faz 5 ile Kiyas

Faz 6'ya gecerken Faz 5 ile karsilastirmali tablo yapmak gerekiyordu; bu adim artik tamamlandi.
Ama bu tabloyu "hangisi daha buyuk" gibi naif okumamak gerekir, cunku ortamlar artik ayni degildir.
Dogru kullanim, Faz 5'in sentetik tabanini Faz 6'nin trace ve cross-domain sonucuyla birlikte okumaktir.

| Asama | Train Domain | Eval Domain | Model | Success Rate | Latency Metric | Avg Energy | Dominant Action |
|---|---|---|---|---:|---:|---:|---:|
| Faz 5 full model retraining | synthetic | synthetic | PPO | 76.17% | P95 = 2.799 | 0.0768 | 3 |
| Faz 6 trace validation | trace | trace (val) | PPO | 99.20% | training avg delay = 0.3702 | 0.0318 | n/a |
| Faz 6 trace hold-out | trace | trace (test) | PPO | 99.60% | avg delay = 0.3349 | 0.0144 | n/a |
| Faz 6 domain shift | synthetic | trace | PPO | 99.60% | P95 = 0.5618 | 0.0255 | 3 |
| Faz 6 domain shift | trace | synthetic | PPO | 53.20% | P95 = 4.6883 | 0.0101 | 4 |

Bu tablo ne soyluyor:
- Faz 5'te sentetik tarafta kurdugumuz PPO tabani anlamsiz degil; trace tarafina tasininca tamamen cokmuyor.
- `synthetic -> trace` sonucu cok guclu oldugu icin, Faz 5'teki mimari secimlerin bir kismi trace tarafinda da tasinmis gorunuyor.
- `trace -> synthetic` yonundeki belirgin dusus, iki ortam arasinda yonlu bir genelleme farki oldugunu gosteriyor.
- validation ve hold-out test birbirine yakin kaldigi icin, trace tarafindaki yuksek performans yalnizca validation sansi olarak aciklanmiyor.

---

## Faz 6'nin Katkisi

Bu faz sonunda artik su iddialari daha guvenli kurabiliyoruz:

1. Proje yalnizca sentetik RL prototipi degil.
Trace splitleri ile egitim, degerlendirme ve checkpoint uretimi calisan bir pipeline haline geldi.

2. Faz 5'te kurulan sentetik PPO tabani tamamen yapay bir basari degildi.
Sentetikten trace'e geciste policy guclu kalabildi.

3. Domain shift olgusu projede olculebilir hale geldi.
Yani "bir ortamda iyi olan model diger ortamda ne yapiyor?" sorusu artik sayisal olarak cevaplanabiliyor.

4. Domain shift simetrik degil.
Bu da Faz 6'yi sadece bir egitim fazi olmaktan cikarip, genelleme ve dagilim farki uzerine bilimsel bulgu ureten bir asamaya donusturuyor.

---

## Faz 6 Sonunda Acik Notlar

Faz 6 tamamlanmis kabul edilse de, ileride guclendirilebilecek noktalar vardir:

1. Faz 5 bulgularinin trace tarafindaki dogrulamasi henuz sinirlidir.
Su an sadece hizli ablation spot-check vardir. Gerekirse trace tarafinda daha sistematik semantic/partial/mobility karsilastirmasi eklenebilir.

2. `experiments/trace/train_ppo.py` icindeki otomatik rapor yazimi bir onceki kosuda encoding bozulmasi uretmisti.
Bu rapor temizlenmis son surumdur; ileride script raporu tekrar overwrite edecekse encoding akisinin sabitlenmesi gerekir.

---

## Artefaktlar

- [trace_training_metrics.csv](D:/task-offloading-study/results/raw/trace/training/trace_training_metrics.csv)
- [trace_holdout_evaluation.csv](D:/task-offloading-study/results/raw/trace/holdout/trace_holdout_evaluation.csv)
- [trace_domain_shift_evaluation.csv](D:/task-offloading-study/results/raw/trace/domain_shift/trace_domain_shift_evaluation.csv)
- [trace_holdout_test_report.md](D:/task-offloading-study/results/tables/trace_holdout_test_report.md)
- [trace_domain_shift_report.md](D:/task-offloading-study/results/tables/trace_domain_shift_report.md)
- [ppo_v3_trace_best.zip](D:/task-offloading-study/models/ppo/trace_training/ppo_v3_trace_best.zip)
- [Phase_5_Report.md](D:/task-offloading-study/phase_reports/Phase_5_Report.md)

---

## Faz 6 Karari

Faz 6 tamamlandi.

Trace training, domain-shift evaluation ve final hold-out test birlikte okundugunda, proje artik trace-driven dogrulama asamasini gecmis durumdadir.
Bir sonraki dogru adim Faz 7'de two-stage training / oracle-label hattina gecmektir.

