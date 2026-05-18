Bkz. ortak kavram sozlugu: v2_docs/project_concepts_glossary.md

# Real Composite Calibration Plan

Bu dokuman, mevcut `real_composite_trace` benchmark'inin neden MEC task offloading icin yeniden kalibre edilmesi gerektigini ve yeni builder surumunde hangi kurallarin uygulanacagini sabitler.

Bu bir sonuc raporu degil, kontrollu duzeltme planidir.
Ama amac "rasgele duzeltme yapmak" degildir; raw kaynaklarin rolleri korunurken fiziksel olarak anlamli bir MEC karar problemi uretmektir.

## 1. Sorun Nedir?

`v2_docs/phase_6/real_composite_feasibility_audit.md` su tabloyu gosterdi:

- medyan `cpu_cycles`: yaklasik `100B`
- medyan deadline penceresi: yaklasik `2.1 s`
- medyan en iyi cloud alt siniri: yaklasik `20.1 s`
- en iyimser alt sinirda dahi feasibility orani: `3.28%`
- `cpu_norm` saturasyon orani: `97.32%`

Yani su anki benchmark, yalnizca zor degil; buyuk bolumu fiziksel olarak imkansiz task dagilimi uretmektedir.
Bu nedenle mevcut dusuk PPO sonucu, su asamada algoritma basarisizligi diye yorumlanmayacaktir.

## 2. Kaynaklarin Dogru Rolu Ne Olmali?

### 2.1 Glasgow MEC

Bu kaynak, Rome taxi mobility ile Alibaba server utilisation verisini birlestirerek olusturulmustur.
Bizim icin ana rol:

- mobil cihaz hareketi ve konum
- mobility-temporal context

Sinir:
- tek basina zengin coklu edge topolojisi vermez
- task workload veya semantic alan vermez

### 2.2 Alibaba Cluster Trace

Bu kaynak production datacenter workload ve machine utilisation verisidir.
Bizim icin ana rol:

- task arrival yapisi
- gorev zorluk/rank sinyali
- server-side load/capacity context

Sinir:
- `plan_cpu` ve `plan_mem` dogrudan MEC `cpu_cycles` veya payload boyutu gibi kullanilmayacak
- bunlar goreli zorluk/proxy sinyali olarak ele alinacak

### 2.3 UCI MEC Execution Times

Bu kaynak, edge server'larda gercek offloaded image-recognition turnaround time olcumleri verir.
Bizim icin ana rol:

- MEC zaman olcegi anchori
- execution/turnaround kalibrasyonu

Sinir:
- tek basina workload arrival veya mobility vermez
- task semantic ya da payload alanlarini dogrudan vermez

## 3. Yeni Benchmark Ilkesi

Yeni benchmark su sekilde tanimlanacak:

> Glasgow mobiliteyi, Alibaba workload ve server context'i, UCI ise MEC turnaround olcegini verecek; fakat hicbir alan kaynak anlaminin disina cikarilip dogrudan fiziksel buyukluk gibi kullanilmayacak.

Kisaca:

- Glasgow = mobility backbone
- Alibaba batch_task = arrival + difficulty ranking
- Alibaba machine_usage/meta = edge/server context
- UCI = MEC service-time scale

## 4. Alan Bazli Yeni Mapping Kurallari

## 4.1 arrival_time

- Kaynak: Alibaba `batch_task.start_time`
- Kullanim: gorev gelis sirasi ve temporal workload yogunlugu
- Not: mutlak zaman degil, goreli trace zamani olarak korunur

## 4.2 device_id ve location

- Kaynak: Glasgow mobility
- Kullanim: cihaz kimligi, lat/long, hareketlilik
- Not: mobility cihazlari ile workload kayitlari birebir ayni sistemin kaydi degildir; bu nedenle bu alan `composite pairing` olarak raporlanir

## 4.3 server context

- Kaynak: Alibaba `machine_usage` + `machine_meta`
- Kullanim:
  - edge server id
  - cpu/memory load context
  - edge capacity proxy

- Degisiklik:
  - mevcut builder'daki Glasgow `serverId` ana server context kaynagi olmaktan cikacak
  - Glasgow mobiliteyi, Alibaba ise server kapasite/yuk baglamini tasiyacak

Gerekce:
Glasgow consecutive subset'te fiilen tek `serverId` gorulmektedir.
Bu tek basina coklu edge karar problemi icin yeterli degildir.

## 4.4 workload difficulty

- Kaynak: Alibaba `plan_cpu`, `plan_mem`, gerekirse `duration`
- Kullanim: gorevin goreli zorluk sinyali
- Kural:
  - bu alanlar dogrudan fiziksel `cpu_cycles` ve `data_size` gibi kullanilmayacak
  - once percentile/quantile uzayina tasinacak
  - sonra MEC olcegine map edilecek

## 4.5 cpu_cycles

Yeni ilke:

- `cpu_cycles = plan_cpu * 1e9` kullanilmayacak
- `plan_cpu` sadece goreli zorluk siralamasi verecek
- fiziksel buyukluk UCI MEC execution-time olcegi ile kalibre edilecek

Onerilen kalibrasyon mantigi:

1. Alibaba `plan_cpu` icin goreli zorluk skoru hesaplanacak
   - log/quantile tabanli difficulty score
2. UCI execution-time havuzundan bir MEC service-time anchor secilecek
3. `cpu_cycles`, env'deki edge compute hizi (`2e9 cycles/s`) ile tutarli olacak sekilde turetilecek
4. Son deger makul MEC bandina clip edilecek

Pratik hedef:

- medyan edge compute suresi saniyeler degil, alt-saniye ile birkac saniye arasi olmalidir
- `cpu_norm` saturasyon orani dusuk tutulmalidir

Ilk hedef aralik:

- `cpu_norm` saturasyon orani < `20%`
- medyan en iyi-case compute/gecikme alt siniri, medyan deadline penceresine cok uzak olmamalidir

## 4.6 data_size

Yeni ilke:

- `data_size = plan_mem * 1024 KB` dogrudan kullanilmayacak
- `plan_mem` payload boyutunun goreli agirlik sinyali olarak ele alinacak
- image/offloading senaryosuna uygun MEC payload bandina map edilecek

Bu alan acikca `derived/proxy` olarak etiketlenecek.

## 4.7 deadline

Yeni ilke:

- deadline, yalnizca `execution_time * 4 + 0.5` gibi sabit kisa pencereyle turetilmeyecek
- deadline, gorevin fiziksel en iyi-case servis alt siniri ile tutarli olacak

Onerilen mantik:

1. task icin best-case lower-bound delay hesapla
   - local
   - edge
   - cloud
   - partial-offloading alt siniri
2. deadline window'u bu alt sinirin uzerine kontrollu slack ile kur
3. slack da tamamen keyfi degil, UCI turnaround olcegi ve difficulty sinyaliyle birlikte secilsin

Pratik hedef:

- lower-bound feasibility ne cok dusuk ne de neredeyse `100%` olsun
- benchmark zor ama cozulur kalsin

Ilk hedef aralik:

- lower-bound feasibility: yaklasik `55% - 80%`

Bu band iki seyi ayni anda saglar:

- benchmark imkansiz olmaz
- benchmark asiri kolay da olmaz

## 4.8 priority / semantic alanlar

Bu alanlar dogrudan gercek kaynaklardan gelmedigi icin:

- ana real-data benchmark'ta `semantic prior` etkisi daha dikkatli yorumlanacak
- priority semantic gercek alan gibi sunulmayacak
- gerekirse ayrik proxy ablation olarak ele alinacak

## 5. State Normalization Duzeltmesi

Mecut sabit normalize:

- `size_norm = min(1.0, size_bits / 1e7)`
- `cpu_norm = min(1.0, cpu_cycles / 1e10)`

Bu yapi real-data benchmark'ta saturasyon yaratti.

Yeni ilke:

- real-data benchmark icin log/quantile tabanli normalize metadata'si uretilecek
- normalization, benchmark ile birlikte kaydedilen calibration metadata'dan beslenecek

Onerilen form:

- `norm(x) = clip((log1p(x) - log1p(p05)) / (log1p(p95) - log1p(p05)), 0, 1)`

Bu sayede:

- aykiri buyuk degerler state'i ele gecirmez
- ajan orta band farklarini ayirt edebilir

## 6. Basari Kriterleri

Yeni builder, asagidaki kapilardan gecmeden kabul edilmeyecek:

1. feasibility audit tekrar kosulacak
2. `lower_bound_feasible_rate` makul banda gelmeli
3. `cpu_norm_saturation_rate` belirgin sekilde dusmeli
4. hicbir priority grubu `0% feasible` gibi sacma durumda kalmamali
5. yeni benchmark, `phase_reports/Phase_6_Report.md` ve mapping dokumanlarinda acikca anlatilacak

## 7. Bu Duzeltmeden Sonra Ne Olacak?

Sira su olacak:

1. builder kalibrasyonu
2. benchmark'in yeniden uretimi
3. feasibility audit tekrar kosusu
4. PPO'nun duzeltilmis benchmark uzerinde yeniden kosulmasi
5. ancak ondan sonra `DQN`, `A2C`, Faz 7 ve Faz 8

Bu sira ozellikle korunacak.
Yani once benchmark dogrulanacak, sonra ajanlar tekrar degerlendirilecek.
