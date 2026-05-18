Bkz. ortak kavram sozlugu: v2_docs/project_concepts_glossary.md

# Real Data Strategy and Recovery Plan

## Neden Bu Dokuman Var

Bu proje artik sadece sentetik simÃ¼lasyon sonuclariyla ilerlemeyecek.
Sentetik veri debug, smoke test ve hizli gelistirme icin tutulacak; ana bilimsel iddia ise gercek veri kaynaklariyla tekrar dogrulanacak.

Bu dokumanin amaci, hangi gercek veri setinin hangi alani besleyecegini ve hangi fazlarin gercek veriyle yeniden kosulmasi gerektigini tek yerde sabitlemektir.

Operasyonel manifest:

- `configs/phase_6/raw_real_data_manifest.yaml`
- Birlesik inventory/readiness denetimi: `experiments/phase_6/inspect_raw_real_datasets.py`

Bu manifest, secilen veri setlerinin lokal durumunu, hangi rol icin kullanildigini ve real-data mode icin sentetik fallback'in kapali oldugunu kayit altina alir.

## Temel Karar

Ana deney hatti icin tek bir veri setine yaslanmayacagiz.
Task offloading problemi su alanlari birlikte ister:

- mobile node konumu ve hareketliligi
- edge/server utilisation
- task/workload arrival ve resource demand
- execution/turnaround latency
- deadline ve semantic/task type bilgisi

Bu alanlar tek bir acik veri setinde eksiksiz gelmedigi icin, veri kaynaklari rollere ayrilacaktir.
Eksik alanlar gizli sentetik degerlerle doldurulmayacak.
Eger bir alan veri setinde yoksa ya deneyden cikarilacak ya da acikca `derived/proxy` olarak etiketlenecektir.

## Literaturde Bu Ne Kadar Dogru?

Kisa cevap:

- Evet, MEC/task-offloading literaturunde tamamen tek bir eksiksiz gercek veri seti kullanimi nadirdir.
- Cok sayida calisma hala sentetik simulasyon veya sentetik task generator kullanir.
- Gercek veri kullanan daha ciddi calismalarda ise mobility, server utilisation ve latency gibi farkli alt boyutlar farkli gercek kaynaklardan alinip birlestirilebilir.

Ancak burada cok kritik bir metodoloji kurali vardir:

> Birden fazla gercek veri kaynagindan alan birlestirmek mesrudur; ama bu, "tek parca ham gercek offloading dataset'i kullandik" diye sunulamaz.

Dogru sunum sekli:

- `composite real-data benchmark`
- `trace-inspired real-data fusion`
- `mobility from dataset A, server utilisation from dataset B, latency calibration from dataset C`

Yanlis sunum sekli:

- proxy alanlari dogrudan olculmus gercek alan gibi yazmak
- semantik/task-type/deadline gibi eksik alanlari gizlice uretip sonra "gercek veri" demek
- birlestirme mantigini rapordan saklamak

## Literaturden Bizim Icin En Onemli Dersler

1. Glasgow MEC dataset ve ilgili makale zaten veri birlestirme mantigi kullaniyor.
`Delay-Tolerant Sequential Decision Making for Task Offloading in Mobile Edge Computing Environments` calismasi, Rome taxi mobility ile Alibaba Cluster server utilisation verisini birlestirerek MEC offloading degerlendirmesi yapiyor.
Bu, bizim coklu gercek kaynak kullanma kararimizin literatur disi olmadigini gosterir.

2. UCI MEC execution-time dataset gibi olculmus latency datasetleri alt-bilesen kalibrasyonu icin degerlidir.
Bu tip veri setleri butun offloading dunyasini tek basina vermez; ama execution/turnaround time gibi kritik alt bilesenleri gercek olcumle sabitler.

3. Benchmark ve standart eksigi literaturde zaten acik bir sorundur.
Cloud/fog/edge performans degerlendirmesi uzerine yapilan benchmark incelemeleri, standart benchmark eksikliginin arastirma toplulugu icin sorun oldugunu acikca soyluyor.
Bu nedenle bizim veri stratejimizde en onemli sey "tek dataset bulmak" degil, veri kaynaklari ve proxy alanlar konusunda tam seffaf olmaktir.

## Bizim Veri Stratejimiz Ne Zaman Bilimsel Olarak Dogru Sayilir?

Asagidaki kosullar saglanirsa evet, savunulabilir:

- her veri setinin rolu acikca ayrilmissa
- hangi alanin dogrudan gercek, hangisinin proxy oldugu yazilmissa
- ana sonuclar `single raw dataset result` degil, `composite real-data benchmark result` diye etiketlenirse
- ablation veya sensitivity ile proxy alanlarin etkiye ne kadar hassas oldugu gosterilirse

Asagidaki durumda ise zayif olur:

- semantik oncelik, task type veya deadline gibi eksik alanlar keyfi doldurulursa
- bu alanlarin etkisi ayrica test edilmezse
- sonuclar tek bir "gercek veri" etiketiyle fazla iddiali sunulursa

Ek zorunlu kosul:

- birlestirilen alanlarin MEC task offloading fizigiyle tutarli oldugu feasibility audit ile gosterilmelidir

Bu kosul artik teorik degil, pratiktir.
Ilk audit, ilk kompozit benchmark'in bu kapidan gecemedigini gostermisti.
Sonrasinda builder kalibre edildi ve audit tekrar kosturuldu.
Guncel audit, benchmark'in MEC task offloading fizigine anlamli bicimde yaklastigini gostermektedir.

## Secilen Veri Setleri

| Veri seti | Kaynak | Projedeki rol | Sagladigi alanlar | Sinirlar |
|---|---|---|---|---|
| Simulating Mobile Edge Computing Environment with Real Data Sets | University of Glasgow, DOI `10.5525/gla.researchdata.896`, https://researchdata.gla.ac.uk/896/ | Ana real-data backbone | Rome Taxi mobility + Alibaba server utilisation birlesimi; mobile node hareketi, GPS/location, server id, cpu utilisation, memory utilisation | Task semantic, image/task payload ve LLM semantic priority dogrudan yok |
| UCI Image Recognition Task Execution Times in Mobile Edge Computing | UCI, DOI `10.24432/C5N617`, https://uci-ics-mlr-prod.aws.uci.edu/dataset/859/image%2Brecognition%2Btask%2Bexecution%2Btimes%2Bin%2Bmobile%2Bedge%2Bcomputing | Edge execution/latency kalibrasyonu | 4 edge server icin real image-recognition turnaround execution time, 4000 instance | Mobility, CPU demand ve semantic alanlari yok |
| Alibaba Cluster Trace v2018 | Alibaba Cluster Trace Program, https://github.com/alibaba/clusterdata | Workload ve server-load gercek kaynak | Production cluster CPU/memory utilisation, machine metadata, batch workload/resource usage | Mobile mobility ve semantic task type yok |
| Google Cluster Trace | Google Borg traces, https://github.com/google/cluster-data | Opsiyonel workload cross-check | Production workload traces, job/task scheduling ve resource demand karakteristikleri | Mobility ve MEC execution time yok |
| Didi Gaia | DiDi GAIA Open Data Initiative | Opsiyonel ikinci mobility/domain validation | Trajectory/mobility/location karakteristigi | Task workload, CPU cycles, deadline ve semantic priority tek basina yok |

## Veri Alanlari Nasil Doldurulacak

| Proje alani | Birincil gercek kaynak | Yorum |
|---|---|---|
| mobile location / movement | Glasgow MEC dataset veya Didi Gaia | Ana hat icin Glasgow/Rome Taxi; Didi Gaia ikinci mobility validation olabilir |
| edge/server load | Glasgow MEC dataset ve Alibaba Cluster Trace | Glasgow dataset zaten Alibaba utilisation ile birlestirilmis; gerekirse Alibaba raw trace ile genisletilir |
| task arrival / workload pressure | Alibaba Cluster Trace veya Google Cluster Trace | Gercek production workload kaynaklari kullanilir |
| edge execution latency | UCI MEC execution-time dataset | Offloaded image-recognition turnaround time ile latency modeli kalibre edilir |
| cpu demand / resource usage | Alibaba veya Google Cluster Trace | CPU cycles birebir yoksa resource-demand proxy olarak acikca belgelenir |
| deadline | Veri setinde yoksa deney tasarim parametresi veya trace-derived proxy | Gercek kaynakta yoksa asla "dogrudan real field" diye yazilmaz |
| task_type / semantic priority | Gercek veri setlerinde dogrudan yok | Real-data ana deneyde devre disi birakilabilir veya ayri semantic/proxy ablation olarak etiketlenir |

## Sentetik Verinin Yeni Rolu

Sentetik veri artik ana bilimsel sonuc degildir.
Kullanim alani:

- unit test
- smoke test
- debug
- hizli pipeline dogrulama
- modelin kirilmadigini anlamak

Real-data mode acikken sentetik fallback sessizce devreye girmemelidir.
Gercek ham veri bulunamazsa deney durmalidir.

## Guncel Lokal Inventory Durumu

2026-05-09 itibariyla zorunlu birlesik benchmark cekirdegi lokal workspace icine alinmistir:

- `Glasgow MEC`
- `UCI MEC execution-times`
- `Alibaba Cluster Trace` icin secilen minimal alt kume: `machine_meta.csv`, `machine_usage.csv`, `batch_task.csv`

Opsiyonel kaynaklar:

- `Google Cluster Trace`
- `Didi Gaia`

secondary-validation amacli hafifletilmis kapsamla lokal workspace icine alinmistir:

- `Google Cluster Trace`: core subset (`machine_events`, `machine_attributes`, `job_events` ilk shard, `task_events` ilk shard, `task_usage` ilk shard)
- `Didi Gaia`: sample-day mobility CSV'leri

Bu bilincli bir karardir; amac, ilk birlesik benchmark icin gerekli cekirdegi tutup gereksiz veri kalabaligi olusturmamaktir.

## Guncel Kalibrasyon Durumu

Ilk composite build teknik olarak basariyla uretilmis, sonra feasibility audit ile denetlenmis ve ardindan yeniden kalibre edilmistir.

Guncel audit tablosu:

- `v2_docs/phase_6/real_composite_feasibility_audit.md`

Guncel ana bulgular:
- medyan `cpu_cycles`: yaklasik `1.60B`
- medyan deadline penceresi: yaklasik `0.69 s`
- medyan best-case delay: yaklasik `0.60 s`
- lower-bound feasibility: `81.74%`
- `cpu_norm` saturasyon orani: `0.00%`
- `size_norm` saturasyon orani: `4.20%`

Bu nedenle guncel yorum su olacak:

> Real-composite benchmark artik MEC task offloading icin fiziksel olarak anlamli bir rejime cekilmistir; bir sonraki dogru adim PPO'yu bu benchmark uzerinde yeniden kosup performansi yeniden yorumlamaktir.

## Tekrar Edilecek Fazlar

### Faz 6R - Real Data Recovery

Zorunlu.
Gercek veri kaynaklari indirilecek, `configs/phase_6/raw_real_data_manifest.yaml` ile kayit altina alinacak, fallback kapatilacak ve trace/real-data episode splitleri yeniden uretilecek.
Bu recovery yalnizca veri indirme degil, benchmark kalibrasyon duzeltmesini de kapsiyordu.
Bu kalibrasyon adimi tamamlandi; simdi siradaki kapÄ± PPO'nun guncel benchmark uzerinde yeniden kosulmasidir.

### Faz 5R - Real-Data Ablation Spot Check

Kismi tekrar.
En az `full_model`, `w/o semantic prior`, `w/o reward shaping`, `w/o partial offloading` gercek veri uzerinde tekrar olculmelidir.

Bu adim atlanmayacaktir.
Cunku Faz 5'in sentetik ortamda verdigi ana mekanizma sinyalleri, final tez/makale dilinde ancak gercek veri omurgasi uzerinde en az spot-check seviyesinde tekrar gorulurse guclu bicimde savunulabilir.

### Faz 7R - Real-Trace Staged Training

Zorunlu.
Teacher/oracle labels gercek veri kaynaklarindan uretilen task stream uzerinde yeniden kurulmalidir.

### Faz 8 - Real-Trace Graph Policy Comparison Rerun

Zorunlu.
`MLP-PPO`, `Pretrained MLP-PPO`, `GraphPolicy none`, `GraphPolicy late` gercek veri kaynakli task stream uzerinde tekrar karsilastirilmalidir.

### Faz 9 - Advanced Metrics

Faz 9 artik real-data sonuclarini da kapsayacak sekilde ilerlemelidir.
Sentetik sonuc tablolarÄ± sadece debug/baseline olarak ayrilmalidir.

## Guncel Sonuclarin Yeni Etiketi

Faz 5, Faz 7 ve Faz 8'deki mevcut ana sonuclar:

```text
synthetic/simulation-stage results
```

olarak okunmalidir.

Faz 6'daki mevcut trace sonuclari ise repo durumuna gore:

```text
synthetic_didi / trace-inspired pipeline validation
```

olarak okunmalidir.

Bu sonuclar cope atilmaz; pipeline ve model mimarisi icin faydalidir.
Ancak nihai literatur katkisi icin real-data validated sonuc olarak sunulmayacaktir.

## Kapanis Karari

Real-data validated iddia icin yeni kapÄ± sudur:

> Gercek veri kaynaklari lokal olarak indirilmeden, real-data manifest doldurulmadan ve real mode sentetik fallback olmadan calismadan hicbir yeni sonuc "gercek veri uzerinde dogrulandi" diye raporlanmayacak.


