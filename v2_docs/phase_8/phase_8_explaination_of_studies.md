Bkz. ortak kavram sozlugu: v2_docs/project_concepts_glossary.md

# Faz 8 Aciklamasi: Graph-Aware Policy Upgrade Calismasini Sifirdan Anlamak

## Bu Dokuman Neden Var

Bu projede artik basit bir "kod calissin" hedefinden daha ileri bir noktadayiz.
Faz 1-7 boyunca sistemi daha tekrarlanabilir, daha adil karsilastirilabilir, trace-driven deneylere daha yakin ve staged-training destekli hale getirdik.

Fakat bu noktada proje gittikce daha teknik kavramlar icermeye basladi:

- RL agent
- PPO policy
- semantic prior
- trace-driven task
- staged training
- graph-aware policy
- GNN

Bu kavramlar ilk bakista soyut gelebilir. Bu dokumanin amaci, Faz 8'de yapacagimiz isleri hic bilmeyen birinin bile okuyup su sorulara cevap verebilmesini saglamaktir:

- Biz tam olarak neyi degistiriyoruz?
- Neden bu degisiklige ihtiyac var?
- Bunun sonucunda sistem neyi daha iyi yapabilecek?
- Bize tez, makale ve deneysel calisma acisindan ne katacak?
- Hangi dosyalari neden yazacagiz?
- Faz sonunda basarili olup olmadigimizi nasil anlayacagiz?

Bu dokuman teknik uygulama plani degil; uygulama planinin arkasindaki mantigi anlatan calisma rehberidir.

---

## 1. Faz 8'e Gelmeden Once Sistem Ne Yapiyordu

Su ana kadar agent'in karar vermesi icin kullandigimiz ana bilgi formu bir `vektor state` idi.

Basitce soyle dusun:

Agent'e her karar aninda bir sayilar listesi veriyoruz.
Bu liste su tarz bilgileri iceriyor:

- sinyal kalitesi iyi mi?
- gorev buyuk mu?
- gorev cok CPU istiyor mu?
- cihaz bataryasi dolu mu?
- edge server yogun mu?
- LLM hangi aksiyonu daha mantikli goruyor?

Kod seviyesinde bu temsil `src/env/state_builder.py` icinde uretiliyor.
Guncel state 12 boyutlu:

- 6 fiziksel ozellik
- 6 semantic prior ozelligi

Yani agent dunyayi su sekilde goruyor:

```text
[snr, task_size, cpu_cycles, battery, edge_load, edge_energy,
 prior_local, prior_edge_25, prior_edge_50, prior_edge_75, prior_edge_full, prior_cloud]
```

Bu temsil kotu degil. Hatta Faz 5-7 boyunca isimize yaradi.
Ama artik daha buyuk bir soruya geldik:

> Mobil edge computing sistemi dogal olarak bir graf yapisina sahipken, agent'e bunu sadece duz bir sayilar listesi olarak gostermek yeterli mi?

Faz 8 bu soruya cevap arar.

---

## 1.1 Faz 8 Oncesi Hangi Policy Mekanizmalari Vardi

Burada `policy` kelimesini daha acik kullanmak gerekir.
Policy, agent'in karar verme mekanizmasidir.
Yani mevcut durum bilgisine bakip "local mi, edge mi, cloud mu, partial mi?" sorusuna cevap veren yapidir.

Faz 8 oncesinde projede uc ana karar mekanizmasi ailesi vardi.

### 1. Heuristic / rule-based policy ailesi

Bu policy'ler ogrenmez.
Yani neural network egitimi yapmazlar.
Onceden yazilmis kurallara gore karar verirler.

Ornekler:

- `LocalOnly`: her task'i local calistirir
- `EdgeOnly`: her task'i edge'e gonderir
- `CloudOnly`: her task'i cloud'a gonderir
- `RandomPolicy`: rastgele aksiyon secer
- `GreedyLatency`: gecikmesi en dusuk gorunen aksiyonu secer
- `GreedyEnergy`: enerjisi en dusuk gorunen aksiyonu secer
- `GeneticAlgorithm`: meta-heuristic arama ile iyi karar bulmaya calisir

Bu policy'ler bizim icin karsilastirma tabanidir.
Yani "bizim ogrenebilen agent, basit kurallardan daha iyi mi?" sorusunu cevaplamak icin kullanilirlar.

Graph-aware policy bu policy'lerin dogrudan aynisi degildir.
Graph-aware policy daha cok ogrenebilen neural policy ailesini guclendirmek icin tasarlanir.
Ama sonunda bu heuristic policy'lerle de ayni metriklerde karsilastirilabilir.

### 2. MLP tabanli RL policy ailesi

Bu aile Faz 8 oncesindeki ana neural decision mekanizmasidir.

`MLP`, Multi-Layer Perceptron demektir.
Basitce duz vektor alan klasik neural network olarak dusunulebilir.

Bu policy'ler mevcut 12 boyutlu vektor state'i okuyup 6 aksiyon icin karar verir:

```text
12D vector state -> MLP policy -> 6 action score -> selected action
```

Bu ailede kullandigimiz ana algoritmalar:

- `PPO`
- `DQN`
- `A2C`

Bu algoritmalarin hepsi ayni sekilde davranmaz, ama ortak noktalari sudur:

- input olarak duz vektor state alirlar
- karar veren policy/value network yapisi MLP tabanlidir
- graph node/edge iliskilerini dogrudan modellemezler

Faz 5-6 tarafinda bu aile bize sentetik ve trace-driven deneylerde ana RL karsilastirma hattini verdi.

### 3. Semantic prior destekli PPO ve staged-training PPO ailesi

Faz 3'ten itibaren LLM/semantic analyzer artik sadece aciklama veren bir modul olmaktan cikti.
Agent'e 6 boyutlu action prior saglamaya basladi.

Bu durumda MLP policy su bilgileri birlikte kullanmaya basladi:

- fiziksel state ozellikleri
- semantic prior
- confidence / reward shaping etkileri

Faz 7'de ise PPO policy once teacher-labeled dataset ile isitildi, sonra PPO fine-tuning yapildi.
Bu aile icindeki en onemli iki karsilastirma:

- `PPO_from_scratch`
- `PPO_pretrained_finetuned`

Kanonik Faz 7 sonucu:

- `Scratch PPO = 63.00%`
- `Pretrained + PPO = 75.20%`

Yani Faz 8 oncesindeki en guclu neural baseline'imiz, duz vektor state ile calisan semantic prior destekli ve staged-training ile isitilmis `MLP-PPO` cizgisidir.

### Graph-aware policy neye alternatif olacak

Graph-aware policy, en basta su hatta alternatif ve guclendirici bir mimari olarak dusunulmelidir:

```text
12D vector state -> MLP-PPO / DQN / A2C
```

yerine:

```text
graph state -> GNN / graph-aware policy
```

Yani asil hedefimiz sunu test etmek:

> Duz vektor state ile calisan MLP tabanli RL policy'ler yerine, cihaz-edge-cloud-task iliskilerini graph olarak goren bir policy kullanirsak karar kalitesi ve action diversity iyilesiyor mu?

Bu nedenle Faz 8 sonunda ozellikle su karsilastirma onemli olacak:

- `MLP-PPO`
- `Pretrained MLP-PPO`
- `Graph-aware policy`
- `Graph-aware policy + semantic prior`

Bu acik ayrim not defteri icin cok onemli:

> Graph-aware policy, basit heuristic'lerin yerine yazilan bir kural seti degil; duz vektor state ile calisan MLP tabanli neural RL policy'lere daha yapisal ve iliskisel bir alternatif olarak kurgulaniyor.

---

## 2. Vektor State Neden Sinirli Kaliyor

Vektor state'i bir tablo satiri gibi dusunebiliriz.
Her satirda bazi sayilar var ve agent bu sayilardan karar cikarmaya calisiyor.

Bu yaklasimin avantaji:

- basit
- hizli
- PPO gibi klasik RL algoritmalariyla kolay calisir
- onceki fazlardaki deneyleri kurmak icin uygundur

Ama dezavantaji su:

Sistemdeki iliskileri dogrudan temsil etmez.

Mesela gercek sistemde sunlar onemlidir:

- Cihaz hangi edge server'a daha yakin?
- Edge-1 yogun ama Edge-2 bos mu?
- Cihaz hareket ettikce hangi baglanti zayifliyor?
- Bir edge server'in kuyrugu artinca diger kararlar nasil etkileniyor?
- Cloud uzak ama kapasitesi yuksek oldugu icin bazi gorevlerde hala mantikli mi?
- Gorev buyukse ve deadline dar ise hangi yol daha riskli?

Duz vektor bunlari tamamen kaybetmez, ama iliski olarak da acikca gostermez.

Ornek:

```text
edge_load = 0.7
distance = 0.2
cloud_latency = 0.5
```

Bu sayilar agent'e verilir; fakat "bu cihaz su edge'e bagli", "bu edge su an dolu", "bu gorev edge'e giderse kuyruk bekler" gibi baglanti yapisi acik degildir.

Faz 8'de bu eksigi kapatmak istiyoruz.

---

## 3. Graph Ne Demek

Graph kelimesi burada matematiksel anlamiyla kullaniliyor.

Bir graph iki seyden olusur:

1. Node
2. Edge

`Node`, sistemdeki varliktir.
Bizim problemimizde node'lar sunlar olabilir:

- mobil cihaz
- edge server
- cloud
- mevcut task

`Edge`, bu varliklar arasindaki iliskidir.
Bizim problemimizde edge'ler sunlar olabilir:

- cihaz ile edge server arasindaki kablosuz baglanti
- cihaz ile cloud arasindaki baglanti
- task ile cihaz arasindaki "yerelde islenebilirlik" iliskisi
- task ile edge/cloud arasindaki "offload edilebilirlik" iliskisi

Yani Faz 8'de agent'e dunyayi soyle gostermek istiyoruz:

```text
Task
  | \
  |  \ 
Device -- Edge-1
   |      Edge-2
   |      Edge-3
   |
 Cloud
```

Bu sadece sekil olsun diye degil.
Bu sayede agent, karar verirken varliklarin tek tek ozelliklerini ve aralarindaki baglantilari birlikte gorebilir.

---

## 4. Graph-Aware Policy Ne Demek

`Policy`, agent'in karar verme fonksiyonudur.

Basitce:

```text
state -> policy -> action
```

Yani agent mevcut durumu gorur ve su aksiyonlardan birini secer:

- local
- edge_25
- edge_50
- edge_75
- edge_100
- cloud

Su ana kadar ana ogrenebilen policy'lerimiz, yani `PPO`, `DQN`, `A2C` ve Faz 7'deki `Pretrained + PPO` kolu, duz vektor state uzerinden karar veriyordu.
Bu modellerin karar aglari MLP tabanliydi.
Faz 8'de bu MLP tabanli policy ailesine graph temsili verebilen yeni bir karar mekanizmasi eklemek istiyoruz.

Bu durumda akis su hale gelir:

```text
graph state -> graph policy -> action
```

Bu yeni policy, GNN yani Graph Neural Network kullanabilir.

GNN'in temel fikri sudur:

> Her node, komsu node'lardan bilgi toplar; sonra kendi karar temsiline bu bilgiyi katar.

Bizim problemimize cevirirsek:

- device node, yakin oldugu edge server'lardan bilgi alir
- task node, deadline ve CPU ihtiyacini graph icine tasir
- edge node'lari kendi queue/load bilgisini yayar
- cloud node, uzak ama guclu islem kapasitesi bilgisini tasir

Sonra model butun bu graph bilgisinden bir karar cikarir.

---

## 5. Bunu Neden Yapacagiz

Faz 8'in en temel nedeni sudur:

> Offloading problemi aslinda iliskisel bir problemdir; bu yuzden iliskileri gorebilen bir model mimarisi daha dogal ve daha guclu olabilir.

Mevcut MLP tabanli RL policy ailesi, ozellikle Faz 7 sonunda en guclu baseline olan `Pretrained MLP-PPO`, su soruya cevap veriyor:

> Bu 12 sayiya bakinca hangi aksiyon iyi?

Graph-aware policy ise daha zengin bir soru sorabilir:

> Bu cihaz, bu task, bu edge server'lar ve cloud arasindaki mevcut iliskiye gore hangi aksiyon iyi?

Bu fark cok onemli.

Ozellikle mobil edge computing gibi dinamik sistemlerde karar sadece task'in buyuklugune veya bataryaya bagli degildir.
Baglanti kalitesi, edge yogunlugu, kuyruk, uzaklik, cloud gecikmesi ve semantic oncelik birlikte dusunulmelidir.

Faz 8 bu birlikte dusunme kapasitesini arttirmayi hedefler.

---

## 6. Faz 7'den Faz 8'e Devreden Problem Nedir

Faz 7'de staged training sonucunda guzel bir kazanim elde ettik:

- `Pretrained + PPO = 75.20%`
- `Scratch PPO = 63.00%`

Bu iyi bir sonuc.
Ama bir davranissal sinir hala duruyor:

Kanonik pretrained policy `Full Cloud`a cokmuyor, bu iyi.
Fakat kararlar hala agirlikli olarak `Edge %75` tarafinda toplanabiliyor.

Buna basitce `action diversity` problemi diyebiliriz.

Yani agent basarili olsa bile, farkli durumlarda yeterince farkli kararlar veriyor mu?

Ornek:

- batarya cok dusukken local veya partial agirlik artiyor mu?
- edge kuyrugu doluyken cloud veya local secimi artiyor mu?
- deadline cok darsa daha hizli yollar seciliyor mu?
- task kucukken gereksiz cloud kullanimi azaliyor mu?

Faz 8'de graph-aware policy'nin bu davranissal siniri iyilestirip iyilestirmedigini inceleyecegiz.

---

## 7. Sonunda Ne Elde Etmek Istiyoruz

Faz 8 sonunda elimizde sadece yeni bir model dosyasi olsun istemiyoruz.
Asil hedef daha bilimsel bir katki paketi olusturmak.

Elde etmek istedigimiz ana ciktilar:

1. Graph state builder
   - Simulator durumunu graph formatina cevirebilen kod.
   - Bu, projenin mimari seviyesini yukari tasir.

2. Graph-aware policy
   - Duz vektor yerine node/edge iliskilerini kullanarak karar veren model.

3. Semantic prior fusion
   - LLM'den gelen prior bilgisinin graph policy ile nasil birlestirilecegini gosteren yapi.

4. Karsilastirma
   - `MLP-PPO` ile `GNN/graph-aware policy` ayni action space ve benzer evaluator mantigi ile karsilastirilacak.

5. Davranissal yorum
   - Graph-aware policy sadece success rate'i arttirdi mi?
   - Yoksa karar cesitliligini ve context-sensitive davranisi da iyilestirdi mi?

Bu son madde cok degerli.
Cunku tez/makale dilinde sadece "daha iyi skor aldik" demek yerine sunu diyebiliriz:

> Graph-aware policy, offloading kararini sistem topolojisine daha duyarli hale getirdi.

Bu iddia, projenin bilimsel katkisini ciddi sekilde guclendirir.

---

## 8. Hangi Dosyayi Neden Yazacagiz

### `src/env/graph_state_builder.py`

Bu dosya simulator durumunu graph formatina cevirecek.

Neden gerekli?

Mevcut state builder duz vektor uretiyor.
GNN ise node ve edge bilgisi ister.
Bu dosya iki dunya arasindaki kopru olacak.

Ne uretecek?

- node feature matrix
- edge index
- edge feature matrix
- semantic prior
- metadata

Katkisi ne?

Bu dosya sayesinde sistem artik sadece "12 sayilik durum" degil, "cihaz-edge-cloud-task iliski agi" olarak temsil edilebilecek.

### `src/env/graph_rl_env.py`

Bu dosya mevcut `OffloadingEnv` davranisini bozmadan graph observation donduren bir katman olacak.

Neden gerekli?

Mevcut MLP-PPO deneylerini korumak zorundayiz.
Onlari bozarsak Faz 5-7 karsilastirmalari zarar gorur.
Bu yuzden graph deneylerini ayri bir yol olarak kuracagiz.

Katkisi ne?

Eski sistem ve yeni graph sistem yan yana calisabilir hale gelecek.

### `src/agents/graph_policy.py`

Bu dosya graph state'i alip 6 aksiyon icin skor uretecek model olacak.

Neden gerekli?

Graph state tek basina yeterli degil.
Bu state'i okuyup karar verecek bir policy mimarisi gerekiyor.

Katkisi ne?

Bu dosya Faz 8'in model tarafindaki ana yeniligi olacak.

### `experiments/synthetic/...`

Burada graph policy icin deney scriptleri olacak.

Neden gerekli?

Kod yazmak tek basina bilimsel calisma degildir.
Ayni kosullarda deney yapip sonuc uretmemiz gerekir.

Katkisi ne?

MLP-PPO ile graph-aware policy adil sekilde karsilastirilabilecek.

### `v2_docs/phase_8/`

Bu klasor Faz 8 boyunca ne yaptigimizi anlatan dokumanlari tutacak.

Neden gerekli?

Proje buyudukce hafiza dagilir.
Bu klasor, "neden boyle yaptik?" sorusunun cevabini kaybetmememizi saglar.

### `phase_reports/Phase_8_Report.md`

Faz sonunda resmi kapanis raporu olacak.

Neden gerekli?

Her fazin sonunda ne yapildi, hangi testler calisti, hangi sonuc alindi ve ne devredildi net olmali.

---

## 9. Faz 8'i Nasil Parcalara Bolerek Yapacagiz

Faz 8'i tek seferde buyuk bir GNN sistemi yazmak olarak gormemeliyiz.
Daha saglam olan yol, kucuk ve test edilebilir parcalarla ilerlemek.

### Adim 1: Graph state'i tanimla

Ilk soru:

> Bir karar anini graph olarak nasil temsil ederiz?

Bu asamada henuz model egitmiyoruz.
Sadece dogru veri formatini kuruyoruz.

Basari kriteri:

- node sayisi dogru
- edge sayisi dogru
- feature boyutlari dogru
- degerlerde NaN/Inf yok
- semantic prior graph ile birlikte tasiniyor

Guncel durum:

- Bu adim icin `src/env/graph_state_builder.py` eklendi.
- Cikti sozlesmesi `GraphState` dataclass'i ile sabitlendi.
- Detayli sozlesme `v2_docs/phase_8/graph_state_builder_contract.md` icinde aciklandi.
- Ilk unit testler `tests/test_graph_state_builder.py` ile calistirildi.

### Adim 2: Graph policy forward pass

Ikinci soru:

> Model bu graph'i okuyup 6 aksiyon icin skor uretebiliyor mu?

Bu asamada model henuz iyi karar vermek zorunda degil.
Sadece teknik olarak dogru calismali.

Basari kriteri:

- input graph aliyor
- output shape `(6,)` veya batch icin `(batch, 6)`
- NaN/Inf yok
- deterministic seed ile tekrar edilebilir

### Adim 3: Supervised warm-start

Ucuncu soru:

> Faz 7'deki teacher labels kullanilarak graph policy'ye baslangic davranisi ogretilebilir mi?

Bu asama cok mantikli, cunku Faz 7'de zaten `oracle_label_dataset.csv` urettik.
Bu dataset'i graph policy icin de kullanabiliriz.

Basari kriteri:

- train/val loss duser
- test accuracy anlamli seviyeye gelir
- model tek aksiyona cokmez

### Adim 4: RL veya environment evaluation

Dorduncu soru:

> Graph policy gercek environment uzerinde MLP-PPO'ya gore ne yapiyor?

Basari kriteri:

- success rate olculur
- p95 latency olculur
- avg energy olculur
- action diversity olculur
- Edge %75 attractor azaldi mi bakilir

### Adim 5: Faz raporu

Son soru:

> Bu faz bize ne ogretti?

Basari kriteri:

- `phase_reports/Phase_8_Report.md` yazilir
- `task.md` ilgili kutular guncellenir
- Faz 9'a ne devredildigi net yazilir

---

## 10. Bu Calismanin Bize Katkisi Ne Olacak

### Teknik Katki

Sistem artik sadece vektor tabanli RL agent olmayacak.
Graph-aware karar mekanizmasina sahip olacak.

Bu, mimari olarak daha guclu bir sistem demek.

### Bilimsel Katki

Offloading problemi dogal olarak graph yapisinda oldugu icin, GNN tabanli yaklasim makale/tez acisindan daha savunulabilir bir yenilik sunar.

Yani katkimiz su hale gelir:

> LLM-guided semantic prior ile graph-aware RL policy'yi partial task offloading probleminde birlestiren bir framework.

Bu ifade, onceki "PPO ile offloading yaptik" ifadesinden cok daha gucludur.

### Deneysel Katki

Faz 8 sonunda su karsilastirmayi yapabiliriz:

- MLP-PPO
- Pretrained MLP-PPO
- Graph-aware policy
- Graph-aware policy + semantic prior

Bu sayede "graph bilgisi gercekten ise yariyor mu?" sorusuna deneyle cevap verebiliriz.

### Dokumantasyon Katkisi

Bu fazdaki dokumanlar sayesinde ileride calisma rehberi veya tez metodolojisi yazarken su sorularin cevabi hazir olacak:

- Neden graph kullandik?
- Graph state nasil tanimlandi?
- GNN policy neyi farkli yapti?
- Semantic prior graph policy ile nasil birlesti?
- Hangi metriklerde fayda gorduk?

---

## 11. Faz 8 Basarili Olursa Neyi Iddia Edebilecegiz

Faz 8 basarili olursa su iddialari kurabiliriz:

1. Offloading kararini duz vektor yerine graph olarak temsil etmek mumkundur.
2. Graph-aware policy, cihaz-edge-cloud-task iliskilerini karar surecine dahil eder.
3. Semantic prior, graph policy icinde ek karar rehberi olarak kullanilabilir.
4. Graph-aware policy, sadece success rate degil action diversity ve context-sensitive decision structure acisindan da degerlendirilebilir.
5. Faz 7'de kalan `Edge %75` agirlikli karar siniri Faz 8 ile daha iyi analiz edilebilir.

Bu iddialarin hepsi tez/makale icin kullanilabilecek turden iddialardir.

---

## 12. Faz 8 Basarisiz Olursa Bile Ne Ogreniriz

Bu da onemli.
Bir arastirma calismasinda her yeni modelin mutlaka daha iyi cikmasi gerekmez.

Graph-aware policy bekledigimiz kadar iyi cikmazsa bile sunlari ogrenmis oluruz:

- Mevcut problem boyutu icin MLP-PPO yeterli olabilir.
- Graph mimarisi daha fazla veri veya daha uzun egitim isteyebilir.
- Semantic prior fusion stratejisi degistirilmelidir.
- Graph state feature'lari yeterince ayirt edici olmayabilir.
- Edge/cloud iliskilerini daha gercekci modellemek gerekebilir.

Yani Faz 8 sadece "GNN daha iyi mi?" sorusu degil.
Ayni zamanda "bu problemde graph temsili ne kadar faydali?" sorusunu bilimsel olarak test eder.

---

## 13. Bu Fazda Dikkat Etmemiz Gereken Riskler

### Risk 1: Mevcut MLP deneylerini bozmak

Mevcut `OffloadingEnv` ve 12 boyutlu state, `PPO`, `DQN`, `A2C` ve Faz 7'deki `Pretrained + PPO` deneyleriyle uyumlu.
Graph calismasini buna zarar vermeden ayri bir yol olarak kurmaliyiz.

### Risk 2: GNN'i cok erken karmasiklastirmak

Ilk hedef devasa bir model degil.
Ilk hedef dogru graph state ve calisan forward pass.

### Risk 3: Sadece success rate'e bakmak

Faz 7'den biliyoruz ki success rate tek basina yeterli degil.
Action diversity ve decision structure da izlenmeli.

### Risk 4: PyG kurulum problemi

`requirements.txt` icinde `torch-geometric==2.5.0` var.
Ama mevcut venv'de import kontrolu basarisiz oldu.
Bu yuzden gerekirse PyTorch-only fallback ile baslayacagiz.

### Risk 5: Graph kullandik diye otomatik katkili sanmak

Graph mimarisi ancak deneyle fayda gosterirse bilimsel katkidir.
Bu nedenle Faz 8 sonunda mutlaka MLP-PPO ile karsilastirma yapilacak.

---

## 14. Not Defteri Icin Kisa Ozet

Faz 8'in tek cumlelik ozeti:

> Faz 8 oncesinde ana neural policy hattimiz `PPO`, `DQN`, `A2C` ve ozellikle Faz 7'de guclenen `Pretrained MLP-PPO` idi; bu modeller sistemi 12 sayilik duz bir vektor olarak goruyordu. Faz 8'de sistemi cihaz, edge, cloud ve task node'larindan olusan bir graph olarak temsil edip graph-aware policy ile daha baglama duyarli offloading kararlari almayi hedefliyoruz.

Neden yapiyoruz?

> Cunku task offloading problemi dogal olarak iliskisel bir problemdir; cihaz-edge-cloud-task baglantilarini modele acikca gostermek, daha iyi ve daha aciklanabilir kararlar uretmemize yardimci olabilir.

Sonunda ne elde edecegiz?

> Graph state builder, graph-aware policy, semantic prior fusion ve `MLP-PPO / Pretrained MLP-PPO` ile graph policy karsilastirmasi.

Bize katkisi ne?

> Projeyi basit RL tabanli offloading calismasindan, LLM-guided graph-aware partial task offloading framework seviyesine tasir.

Faz 8'in ana kontrol sorusu:

> Graph-aware policy, duz vektor state ile calisan `Pretrained MLP-PPO`ya gore Faz 7'den kalan Edge %75 agirlikli karar sinirini azaltip daha context-sensitive action diversity uretebiliyor mu?

---

## 15. Faz 8 Okuma Sirasi

Faz 8'i anlamak icin onerilen okuma sirasi:

1. `v2_docs/phase_8/phase_8_explaination_of_studies.md`
2. `v2_docs/phase_8/phase_8_Graph_Aware_Policy_Upgrade_plan.md`
3. Faz ilerledikce eklenecek teknik raporlar
4. Faz sonunda `phase_reports/Phase_8_Report.md`

Bu siralama once kavrami, sonra uygulama planini, sonra deney sonucunu takip etmeyi saglar.
