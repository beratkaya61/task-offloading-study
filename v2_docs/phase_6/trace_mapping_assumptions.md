Bkz. ortak kavram sozlugu: v2_docs/project_concepts_glossary.md

# Trace Mapping Assumptions

Amac: Faz 6 trace-driven egitim ve domain-shift analizinde kullanilan trace verisini, proje icindeki task alanlarina nasil cevirdigimizi acik ve tekrar kullanilabilir bicimde belgelemek.

## 2026-05-09 Kapsam Duzeltmesi

Bu dokumanin ilk surumu, Faz 6'daki mevcut episode splitlerini trace-driven akis olarak acikliyordu.
Son repo kontrolunde bu splitlerin raw real dataset dosyalarindan dogrulanmadigi ve `synthetic_didi / trace-inspired` pipeline ciktisi olarak ele alinmasi gerektigi netlesti.

Bu nedenle bu dosya iki katmanli okunmalidir:

1. Mevcut Faz 6 mapping'i: `synthetic_didi / trace-inspired pipeline validation`
2. Yeni real-data mapping hedefi: `v2_docs/real_data_strategy.md` icindeki Glasgow MEC, UCI MEC execution-time, Alibaba Cluster Trace ve opsiyonel Google/Didi kaynaklari

Real-data mode acikken raw dataset bulunamazsa sistem sentetik fallback'e sessizce donmemelidir.
Deney durmali ve eksik veri manifestte acikca gorunmelidir.

Lokal inventory ve gozlenen kolon yapisi:

- `v2_docs/phase_6/real_data_inventory_report.md`

Bu rapor, 2026-05-10 itibariyla workspace icine indirilen gercek veri dosyalarinin satir/kolon ozetini verir ve Faz 6R.4 birlestirme adiminin giris noktasi olarak kullanilir.

Ilk composite split build ciktilari:

- `v2_docs/phase_6/real_composite_build_report.md`
- `configs/phase_6/real_composite_trace_ppo_training.yaml`

Ilk fiziksellik denetimi:

- `v2_docs/phase_6/real_composite_feasibility_audit.md`

Bu audit, ilk kompozit benchmark'in teknik olarak uretildigini ama MEC task offloading problemi icin henuz dogru kalibre edilmedigini gostermisti.
Ozellikle:
- medyan `cpu_cycles` yaklasik `100B`
- medyan deadline penceresi yaklasik `2.1 s`
- medyan en iyi cloud alt siniri yaklasik `20.1 s`
- en iyimser alt sinirda bile feasibility orani yalnizca `3.28%`

Bu nedenle ilk kompozit build, nihai real-data benchmark olarak degil, `ilk kalibrasyonsuz kompozit deneme` olarak okunmustur.

Sonraki duzeltme:

- `v2_docs/phase_6/real_composite_calibration_plan.md`

Bu plan uygulanmis ve builder yeniden kalibre edilmistir.
Guncel benchmark durumu artik yeni audit raporunda tutulur:

- `v2_docs/phase_6/real_composite_feasibility_audit.md`

Bu iki artefakt, ilk gercek veri kompozitinin hangi kaynaklardan beslendigini, hangi alanlarin proxy oldugunu ve train/val/test splitlerinin hangi dizine yazildigini sabitler.

Bu dokuman iki isi birden yapar:
- tez ve rapor tarafinda hangi alanin neye donusturuldugunu aciklar
- kod tarafinda TraceLoader ve TraceProcessor icindeki varsayimlari merkezi bir yerde toplar

## 1. Neden Mapping Assumptions Gerekiyor?

Trace veri setleri, offloading ortamimizin ihtiyac duydugu tum alanlari birebir vermez. Bu nedenle bazi alanlari dogrudan okuyoruz, bazilarini ise mantikli varsayimlarla turetiyoruz.

Ornek:
- bir trace dosyasinda cpu_cycles vardir ama task_type yoktur
- deadline mutlak zaman gibi gelebilir; bizim env ise goreli deadline kullanir
- mobilite konumu vardir ama semantic oncelik yoktur

Bu ceviriler kayit altina alinmazsa Faz 6 sonuclari metodolojik olarak zayif gorunur.

## 1.1 Literatur Acisindan Dogruluk Kriteri

Task offloading ve mobile edge computing literaturunde eksiksiz tek bir gercek veri setiyle calisan calisma sayisi sinirlidir.
Bu nedenle mobility, server utilisation, workload ve latency gibi boyutlari farkli gercek kaynaklardan beslemek metodolojik olarak kabul edilebilir.

Ama burada cizgi nettir:

- Kabul edilebilir olan: rol ayrimi yapilmis coklu gercek kaynak birlestirmesi
- Kabul edilemez olan: eksik alanlari gizlice uydurup tumunu "gercek veri" diye sunmak

Bu yuzden bizim kuralimiz su olacak:

1. Dogrudan gozlenen alanlar `real`
2. Turetilen alanlar `derived/proxy`
3. Semantik veya task-type gibi dogrudan olmayan alanlar ana real-data deneyde kapatilacak ya da ayri ablation olarak raporlanacak

Bu metodoloji notunun ust seviyedeki ozet karari `v2_docs/real_data_strategy.md` icinde tutulur.

## 2. Kodda Bu Mapping Nerede Yapiliyor?

- src/core/dataset_loader.py: debug/synthetic kaynak yardimcilari ve indirilen real dataset okuyucularini tutar
- src/core/trace_loader.py: diskten ham trace CSV veya kaydedilmis episode JSON dosyalarini yukler
- src/core/trace_processor.py: trace kayitlarini episode ve task yapisina cevirir
- src/env/rl_env.py: trace task objesini env icindeki gorev formatina donusturur

Kisaca:
- Dataset loader = kaynak veriyi oku
- Trace loader = kaydedilmis split veya ham trace dosyasini oku
- Processor = hazirla ve bol
- Env = egitimde kullanilacak son task nesnesine cevir

## 3. Alan Eslesme Tablosu

Asagidaki tablo mevcut Faz 6 trace-inspired episode formatinin mapping'ini anlatir.
Raw real-data mapping icin kaynak rolleri ayrica `v2_docs/real_data_strategy.md` icinde sabitlenmistir.

| Trace Alani | Hedef Alan | Donusum veya Varsayim |
|---|---|---|
| task_id | task_id | Dogrudan korunur |
| device_id | device_id | Dogrudan korunur |
| arrival_time | arrival_time | Dogrudan korunur |
| deadline | task.deadline | max(0.1, deadline - arrival_time) ile goreli deadline'a cevrilir |
| data_size (KB) | task.size_bits | data_size * 8 * 1024 |
| cpu_cycles | task.cpu_cycles | Dogrudan korunur |
| priority (0-3) | priority_score ve semantic oncelik | Mevcut trace-inspired akista dusuk -> local, orta -> cloud, yuksek -> edge egilimi olarak yorumlanir; raw real-data deneyde bu alan yoksa proxy/ablation olarak etiketlenmelidir |
| location_x, location_y | location | (x, y) tuple olarak saklanir |
| task_type yoksa | task.task_type | Faz 6 mevcut surumde enum icinden rastgele secilir; real-data iddiasinda bu alan dogrudan gercek alan gibi sunulmayacaktir |
| semantic etiket yoksa | semantic_analysis | priority tabanli basit heuristic ile turetilir; real-data ana deneyde kapatilabilir veya proxy ablation olarak raporlanir |

## 3.1 Real-Data Kaynak Rolleri

| Proje alani | Birincil kaynak | Not |
|---|---|---|
| mobile location / movement | Glasgow MEC veya opsiyonel Didi Gaia | Glasgow/Rome Taxi ana mobility backbone olarak secildi |
| edge/server load | Glasgow MEC + Alibaba Cluster Trace | Glasgow seti Alibaba utilisation bilgisiyle birlikte kullanilacak; gerekirse raw Alibaba ile genisletilecek |
| task arrival / workload pressure | Alibaba Cluster Trace veya Google Cluster Trace | Production workload karakteri icin kullanilacak |
| edge execution latency | UCI MEC execution-time | Offloaded image-recognition turnaround time ile latency kalibrasyonu yapilacak |
| cpu/resource demand | Alibaba veya Google Cluster Trace | CPU cycles birebir yoksa resource-demand proxy olarak belgelenir |
| deadline | Dataset yoksa design parameter veya trace-derived proxy | Dogrudan real field gibi yazilmaz |
| task_type / semantic priority | Datasetlerde dogrudan yok | Real-data ana deneyde kapatilir veya proxy ablation olarak ayrilir |

## 4. Bugun Kullandigimiz Temel Varsayimlar

Not:
Asagidaki varsayimlar ilk builder ile guncel kalibre builder arasindaki evrimi aciklar.
Ozellikle `cpu_cycles`, `deadline` ve state normalization alanlari yeni builder surumunde yeniden kurulmustur.

### 4.1 Data size -> size_bits
Trace tarafinda data_size KB cinsinden geliyor. Env tarafi ise transmission ve enerji hesabini bit cinsinden yaptigi icin su donusum kullaniliyor:
size_bits = data_size * 8 * 1024

### 4.2 Absolute deadline -> relative deadline
Trace kaydindaki deadline degeri dogrudan gorev bitis zamani gibi yorumlaniyor. Env ise bir gorevin o andan itibaren ne kadar suresi kaldigiyla calisiyor. Bu nedenle su ceviri kullaniliyor:
relative_deadline = max(0.1, deadline - arrival_time)

Buradaki 0.1 tabani, sifir veya negatif deadline degerlerinin env'i bozmasini engelleyen guvenlik katmanidir.

Ilk real-composite builder'da `deadline` alani dogrudan kaynak veriden gelmedigi icin UCI execution-time uzerinden turetilmis kisa proxy pencere kullanilmisti.
Yeni builder surumunde deadline, task-specific best-case lower bound uzerine kurulan compute-proportional pencere olarak yeniden tasarlandi.

### 4.3 Priority -> semantic recommendation
Trace veri seti semantic analyzer cikisi vermedigi icin Faz 6'da gecici bir heuristic kullaniyoruz:
- priority >= 3 -> edge
- priority == 2 -> cloud
- digerleri -> local

Bu, nihai semantic model degil; trace-driven egitim sirasinda semantic kanalin tamamen bos kalmamasini saglayan pratik bir koprudur.

### 4.4 Missing task_type
Trace tarafinda gorev tipi birebir yoksa TaskType enum icinden rastgele seciliyor. Bu bugun icin kabul edilebilir bir placeholder, ama Faz 6 kapanisinda sinir olarak not edilmelidir.

### 4.5 CPU cycles ve observation saturation siniri
Ilk real-composite builder'da `cpu_cycles = plan_cpu * 1e9` varsayimi kullanildi.
Bu secim state tarafinda `cpu_norm` saturasyonunu yaklasik `97.32%` seviyesine tasimisti.
Yeni builder surumunde:
- `cpu_cycles`, Alibaba difficulty ranking + UCI execution-time olcegi ile MEC kapasitesine kalibre edildi
- `cpu_norm` ve `size_norm` icin log/quantile tabanli hint'ler eklendi
- guncel audit sonucunda `cpu_norm` saturasyon orani `0.00%`, `size_norm` saturasyon orani `4.20%` seviyesine indi

## 5. Bu Varsayimlarin Sonuclara Etkisi Nedir?

- Deadline mapping, success rate ve reward'u dogrudan etkiler.
- Priority -> semantics cevirisi, gercek LLM cikisi degil; trace icinden turetilmis proxy sinyaldir.
- Task type rastgele geldigi icin task turune bagli ince semantic ayrismalar Faz 6'da zayif gorunebilir.

## 6. Faz 6 Icin Simdilik Kabul Ettigimiz Sinirlar

- mevcut lokal Faz 6 episode splitleri raw real dataset olarak dogrulanmis degildir
- trace veri setleri, env icin gereken tum semantic alanlari dogrudan saglamaz
- semantic recommendation su an proxy ve heuristic nitelikte olabilir
- task_type alani gercek trace semantiginden gelmiyorsa real-data iddiasinda kullanilmayacaktir
- bu nedenle mevcut Faz 6, `synthetic_didi / trace-inspired pipeline validation` olarak okunmalidir

## 7. Faz 6 Kapanisinda Ne Guncellenmeli?

Real-data recovery tamamlandiginda bu dosyaya sunlar eklenmeli:
- hangi raw dataset dosyalari lokal olarak kullanildi
- dataset manifest path'leri ve checksum/status bilgileri
- nihai mapping tablosu degisti mi
- hangi alanlar dogrudan geldi, hangi alanlar proxy/derived olarak kullanildi
- semantic reconstruction kullanildiysa bunun real field degil proxy oldugu
- ek normalizasyon veya filtering adimlari

## 8. Kisa Ozet

Faz 6'da trace-style deney yaparken veriyi dogrudan env'e vermiyoruz.
1. TraceLoader diskteki raw real trace veya episode split dosyasini okur
2. TraceProcessor bunlari episode ve task yapisina hazirlar
3. rl_env.py her trace task'i env icindeki gorev formatina cevirir
4. Bu ceviriler sirasinda data_size, deadline ve priority gibi alanlar belirli varsayimlarla map edilir
5. Real-data mode icin raw veri yoksa sentetik fallback calismamalidir

