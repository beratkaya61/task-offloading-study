Bkz. ortak kavram sozlugu: v2_docs/project_concepts_glossary.md

# Faz 6 Report: Trace-Driven Training

**Tarih:** 08 April 2026  
**Durum:** trace pipeline tamamlandi; raw real-data revalidation gerekli  
**Kapsam:** trace-inspired PPO egitimi, domain-shift degerlendirmesi, hold-out test ve Faz 5 -> Faz 6 bulgu baginin kurulmasi

---

## 2026-05-09 Kapsam Duzeltmesi

Bu rapor artik raw real dataset uzerinde tamamlanmis nihai dogrulama raporu olarak okunmayacaktir.
Repo durum kontrolunde `data/synthetic_trace/` altindaki mevcut episode splitlerinin `synthetic_didi` ailesinden gelen trace-inspired episode'lar oldugu ve lokal raw Didi/Glasgow/UCI/Alibaba dosyalarinin henuz dogrulanmadigi netlesmistir.

Bu nedenle Faz 6'nin dogru etiketi sudur:

```text
synthetic_didi / trace-inspired pipeline validation
```

Bu fazdaki sonuclar cope atilmaz; trace loader, processor, split, hold-out test ve domain-shift deney omurgasinin calistigini gosterir.
Ancak tez/makale icin `real-data validated` iddia kurulmadan once `v2_docs/real_data_strategy.md` icindeki veri kaynaklariyla Faz 6R yeniden kosulmalidir.

Kullanilacak real-data omurgasi:

- Glasgow MEC real-data dataset: mobility + server utilisation backbone
- UCI MEC image-recognition execution-time dataset: edge execution/turnaround latency calibration
- Alibaba Cluster Trace: workload/resource-demand ve server-load kaynagi
- Google Cluster Trace: opsiyonel workload cross-check
- Didi Gaia: opsiyonel ikinci mobility/domain validation

## 2026-05-10 Feasibility Audit ve Kalibrasyon Guncellemesi

Faz 6R altinda olusturulan ilk `real_composite_trace` benchmark'i once fiziksellik acisindan denetlenmis, sonra builder yeniden kalibre edilmis ve audit tekrar kosturulmustur:

- ilk audit ve sorun tespiti: `v2_docs/phase_6/real_composite_feasibility_audit.md`
- duzeltme karari: `v2_docs/phase_6/real_composite_calibration_plan.md`
- guncel build ozetleri: `v2_docs/phase_6/real_composite_build_report.md`

Guncel benchmark ana bulgulari:
- medyan `cpu_cycles`: yaklasik `1.60B`
- medyan deadline penceresi: yaklasik `0.69 s`
- medyan best-case delay: yaklasik `0.60 s`
- lower-bound feasibility: `81.74%`
- `cpu_norm` saturasyon orani: `0.00%`
- `size_norm` saturasyon orani: `4.20%`

Bu tablo su anlama geliyor:
- onceki `veri fizigi bozuk` sorunu belirgin sekilde toparlandi
- benchmark artik MEC task offloading icin anlamli bir rejime cekildi
- sonraki dogru adim, PPO'yu bu yeni benchmark uzerinde yeniden kosup performansi tekrar okumaktir

## 2026-05-10 Benchmark Sanity-Check ve Env-Contract Guncellemesi

Kalibrasyon sonrasi benchmark'in environment icinde de anlamli kalip kalmadigini ayirmak icin ek bir sanity-check protokolu kosturuldu:

- config: `configs/phase_6/real_data_benchmark_sanity_check.yaml`
- script: `experiments/phase_6/run_real_data_benchmark_sanity_check.py`
- artefaktlar:
  - `results/phase_6/metrics/real_data/benchmark_sanity/real_data_benchmark_sanity.csv`
  - `v2_docs/phase_6/real_data_benchmark_sanity_check.md`

Bu turda iki kritik contract duzeltmesi yapildi:
- trace location'lar artik env koordinat duzlemine projekte ediliyor
- edge server yerlesimi random degil, trace konum dagilimindan turetilen deterministic centroid'lerle kuruluyor

Heuristic ortalama sonuclari:

| Policy | Mean Success | Mean P95 Latency | Dominant Action |
|---|---:|---:|---:|
| `GeneticAlgorithm` | 48.73% | 1.4505 s | 5 |
| `GreedyLatency` | 42.40% | 1.4497 s | 5 |
| `CloudOnly` | 38.80% | 1.4713 s | 5 |
| `Random` | 24.93% | 3.3610 s | degisken |
| `EdgeOnly` | 12.20% | 2.6865 s | 4 |
| `LocalOnly` | 0.40% | 5.1387 s | 0 |

Mevcut PPO retraining referansi:
- seed `42`: `42.00%`, dominant action `3`
- seed `43`: `42.00%`, dominant action `3`
- seed `44`: `42.00%`, dominant action `3`
- PPO ortalamasi: `42.00%`

Bu tablo Faz 6R icin cok kritik bir ayrim sagladi:
- benchmark tamamen fiziksel olarak bozuk degil; cunku heuristic aile tutarli ve anlamli ayrisiyor
- ama PPO tarafinda yeni problem `seed-varyansi` degil, `action=3` etrafinda sabitlenen zayif local optimum`
- dolayisiyla bundan sonraki asil mudahale noktasi `benchmark'i tekrar bozmak` degil, `reward geometry`, `semantic prior bias` ve `PPO aksiyon kilitlenmesi` tarafidir

---

## Ozet

Faz 6'da sentetik ortamdan cikmaya hazirlanan trace pipeline kurulmus ve trace-inspired episode splitleri ile PPO modeli yeniden egitilmistir.
Bu asamanin amaci, Faz 5'te sentetik tarafta gorulen bulgularin trace benzeri veri akisi uzerinde ne kadar tasinabildigini gormekti.

Bu faz sonunda su zincir tamamlandi:
- trace training kosusu basariyla tamamlandi,
- yeni checkpoint `models/ppo/trace_training/ppo_v3_trace_best.zip` olarak uretildi,
- training metrics `results/phase_6/metrics/synthetic_trace/training/trace_training_metrics.csv` altina yazildi,
- synthetic -> trace ve trace -> synthetic domain-shift tablosu uretildi,
- `test_episodes.json` uzerinde ayri final hold-out evaluation kosturuldu,
- Faz 6 raporu mevcut artefaktlarla temiz ve okunur hale getirildi.

Bu nedenle Faz 6 artik teknik trace pipeline acisindan kapanmis sayilabilir.
Fakat raw real-data bilimsel dogrulama acisindan Faz 6R adimlari aciktir.

Ek mimari not:
- raw veri okuyuculari ve debug/synthetic kaynak yardimcilari `src/core/dataset_loader.py` icinde toplandi
- materialized episode split ve raw trace IO ise `src/core/trace_loader.py` icinde tutuldu
- inventory + readiness denetimi `experiments/phase_6/inspect_raw_real_datasets.py` altinda birlestirildi

---

## Kullanilan Veri ve Splitler

Trace pipeline su dosyalari kullanir:
- egitim: `data/synthetic_trace/train_episodes.json`
- validation: `data/synthetic_trace/val_episodes.json`
- hold-out test: `data/synthetic_trace/test_episodes.json`

Kapsam notu:
Bu dosyalar mevcut repo durumunda materyalize episode splitleridir.
Raw real dataset dosyalarindan yeniden uretildikleri dogrulanmadigi icin bu bolum `real-data validated` olarak degil, `synthetic_didi / trace-inspired` olarak okunmalidir.

Bu rapordaki trace training sonucu, `train_episodes.json` ile egitim ve `val_episodes.json` ile ara dogrulama mantigina dayanir.
Final kapanis kontrolu ise ayri olarak `test_episodes.json` uzerinde yapilmistir.

---

## Trace Training Sonucu

Kaynak artefaktlar:
- checkpoint: `models/ppo/trace_training/ppo_v3_trace_best.zip`
- training metrics: `results/phase_6/metrics/synthetic_trace/training/trace_training_metrics.csv`

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
- CSV: `results/phase_6/metrics/synthetic_trace/holdout/trace_holdout_evaluation.csv`
- rapor: `v2_docs/phase_6/trace_holdout_test_report.md`

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
- CSV: `results/phase_6/metrics/synthetic_trace/domain_shift/trace_domain_shift_evaluation.csv`
- rapor: `v2_docs/phase_6/trace_domain_shift_report.md`

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

1. Proje yalnizca tek dosyalik sentetik RL prototipi degil.
Trace-style splitler ile egitim, degerlendirme ve checkpoint uretimi calisan bir pipeline haline geldi.

2. Faz 5'te kurulan sentetik PPO tabani tamamen yapay bir basari degildi.
Sentetikten trace-inspired akisa geciste policy guclu kalabildi.

3. Domain shift olgusu projede olculebilir hale geldi.
Yani "bir ortamda iyi olan model diger ortamda ne yapiyor?" sorusu artik sayisal olarak cevaplanabiliyor.

4. Domain shift simetrik degil.
Bu da Faz 6'yi sadece bir egitim fazi olmaktan cikarip, genelleme ve dagilim farki uzerine bilimsel bulgu ureten bir asamaya donusturuyor.

---

## Faz 6 Sonunda Acik Notlar

Faz 6 teknik pipeline olarak tamamlanmis kabul edilse de, bilimsel real-data dogrulama icin acik noktalar vardir:

1. Raw real datasetler henuz lokal olarak dogrulanmamistir.
Glasgow MEC, UCI MEC execution-time, Alibaba Cluster Trace ve opsiyonel Didi/Google kaynaklari indirilip manifest ile kayda alinmalidir.

2. Real-data mode icin sessiz synthetic fallback kapatilmalidir.
Ham veri yoksa deney durmali, synthetic episode uretimi otomatik olarak devreye girmemelidir.

3. Faz 5 bulgularinin real-data tarafindaki dogrulamasi henuz yapilmamistir.
Gercek veri omurgasinda Faz 5'in tam ablation yeniden kosusu henuz tamamlanmamistir. Semantic/partial/mobility aileleri gercek veri tarafinda coklu algoritma ve coklu seed ile yeniden kosulmalidir.

4. `experiments/phase_6/train_trace_rl.py` icindeki otomatik rapor yazimi bir onceki kosuda encoding bozulmasi uretmisti.
Bu rapor temizlenmis son surumdur; ileride script raporu tekrar overwrite edecekse encoding akisinin sabitlenmesi gerekir.

---

## Artefaktlar

- [trace_training_metrics.csv](D:/task-offloading-study/results/phase_6/metrics/synthetic_trace/training/trace_training_metrics.csv)
- [trace_holdout_evaluation.csv](D:/task-offloading-study/results/phase_6/metrics/synthetic_trace/holdout/trace_holdout_evaluation.csv)
- [trace_domain_shift_evaluation.csv](D:/task-offloading-study/results/phase_6/metrics/synthetic_trace/domain_shift/trace_domain_shift_evaluation.csv)
- [trace_holdout_test_report.md](D:/task-offloading-study/v2_docs/phase_6/trace_holdout_test_report.md)
- [trace_domain_shift_report.md](D:/task-offloading-study/v2_docs/phase_6/trace_domain_shift_report.md)
- [ppo_v3_trace_best.zip](D:/task-offloading-study/models/ppo/trace_training/ppo_v3_trace_best.zip)
- [Phase_5_Report.md](D:/task-offloading-study/phase_reports/Phase_5_Report.md)

---

## Faz 6 Karari

Faz 6 teknik trace pipeline olarak tamamlandi.

Trace training, domain-shift evaluation ve final hold-out test birlikte okundugunda, proje artik trace-style deney omurgasina sahiptir.
Ancak raw real-data validated asama henuz gecilmis sayilmaz.

Bir sonraki dogru adim, mevcut mimariyi koruyarak real-data recovery hattini acmak ve Faz 6R kapsaminda gercek veri kaynaklariyla yeniden dogrulamaktir.

