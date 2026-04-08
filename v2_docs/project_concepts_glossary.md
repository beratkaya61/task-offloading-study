# Proje Kavram Sozlugu

Bu sozluk, `task-offloading-study` projesinde gecen temel kavramlari tek yerde toplamak icin hazirlandi.
Amac, fazlar ilerledikce karsilasilan teknik terimleri sonradan hizlica hatirlayabilmek ve raporlari daha kolay okuyabilmektir.

Bu dokuman yalnizca Faz 5 veya Faz 6 icin degil, proje genelindeki tum teknik asamalar icin ortak referanstir.

---

## 1. Temel Proje Kavramlari

### Task Offloading
Bir gorevin cihazin kendi uzerinde mi, edge sunucuda mi, yoksa cloud tarafinda mi islenecegine karar verme problemidir.
Bu projenin cekirdek problemi budur.

### Local / Edge / Cloud
- `Local`: Gorev cihazin kendi CPU'sunda calisir.
- `Edge`: Gorev yakindaki edge sunucuya aktarilir.
- `Cloud`: Gorev merkezi bulut altyapisina gonderilir.

### Partial Offloading
Bir gorevin tamamini tek yere gondermek yerine, bir kismini localde bir kismini uzakta isleme yaklasimidir.
Bu projede ayrik aksiyonlarla temsil edilir: ornegin %25, %50, %75 edge oranlari.

### Semantic-Aware Offloading
Karar mekanizmasinin sadece fiziksel sinyalleri degil, gorevin anlamsal niteligini de dikkate almasidir.
Ornek: kritik veya acil bir gorev ile best-effort bir gorevin ayni sekilde ele alinmamasi.

### MEC (Mobile Edge Computing)
Bulut ile cihaz arasinda, gorevlere daha yakin konumlanan edge kaynaklarinin kullanildigi hesaplama modelidir.
Bu proje bu baglamda konumlanir.

### AgentVNE Esinlenmesi
Projede kullanilan asamali dusunme ve metodolojik disiplinin bir kismi AgentVNE benzeri yaklasimlardan ilham alir.
Ozellikle staged training, semantic guidance ve daha guclu deney protokolu bu cizgidedir.

---

## 2. Faz 1: Reproducibility ve Deney Omurgasi Kavramlari

### Reproducibility
Ayni config ve ayni seed ile deney tekrarlandiginda benzer sonuc alinabilmesidir.
Bilimsel savunulabilirlik icin temel gerekliliktir.

### Seed
Rastgeleligin kontrol edilmesi icin kullanilan sabit sayidir.
RL, sampling, task olusumu ve model baslangici gibi surecleri etkiler.

### Determinism
Ayni kosullarda ayni davranisin uretilmesine yakin olma durumudur.
Pratikte tam determinism zor olsa da seed ve sabit protokol ile yaklasilir.

### Config-Driven Workflow
Deney ayarlarinin dogrudan kod icine yazilmasi yerine YAML gibi config dosyalarindan yonetilmesidir.
Bu, tekrar edilebilirligi ve duzeni artirir.

### Experiment Logging
Her kosunun sonucunu CSV/JSON gibi dosyalara yazma altyapisidir.
Sonradan tablo, grafik ve raporlar bu loglardan beslenir.

### Canonical Report
Daginik ciktilar arasindan ana okunacak rapordur.
Bu proje boyunca farkli fazlarda ozet raporlar bu rolü tasir.

---

## 3. Faz 2: Environment ve Simulasyon Kavramlari

### Environment / Env
Ajanin icinde karar verdigi simule dunya veya deney ortami.
Gymnasium arayuzuyle `reset` ve `step` fonksiyonlari uzerinden kullanilir.

### Simulator-Backed Training
Tamamen mock veri yerine simulasyondan uretilmis daha fiziksel task ve cihaz akislarini kullanarak egitim yapmaktir.

### Multi-Step Episode
Tek bir episode icinde birden fazla gorevin sirali olarak islenmesidir.
Bu, tek adimli ortama gore daha gercekci bir karar baglami uretir.

### State Builder
State vektorunu standart bicimde ureten moduldur.
Amac, egitim ve degerlendirme ortamlarinin ayni state mantigini paylasmasidir.

### Reward
Ajanin yaptigi secimin ne kadar iyi oldugunu sayisal olarak ifade eden odul/ceza sinyalidir.
Bu projede gecikme, enerji, deadline, semantic uyum gibi bilesenlerden etkilenir.

### Reward Decomposition
Toplam reward'un hangi bilesenlerden olustugunu ayri ayri dusunmektir.
Bu, neden bir karar iyi veya kotu gorunuyor anlamak icin onemlidir.

---

## 4. Faz 3: Semantic Katman ve LLM Kavramlari

### LLM Guidance
Buyuk dil modelinden ya da benzeri semantic analiz katmanindan karar yardimi almaktir.

### Semantic Prior
LLM veya rule-based semantic analyzer tarafindan uretilen aksiyon egilimi bilgisidir.
Bu, tek bir one-hot tavsiye yerine aksiyonlar uzerinde dagilim veya oncelik saglayabilir.

### Confidence
Semantic analyzer'in kendi tavsiyesinden ne kadar emin oldugunu gosterir.
Yuksek confidence daha guclu semantic etki anlamina gelebilir.

### Structured Output
LLM cikisinin dogrudan parse edilebilir bir yapiyla, ornegin JSON benzeri alanlarla uretilmesidir.

### Reward Shaping
Ajanin daha hizli ve daha yonlu ogrenebilmesi icin odul fonksiyonuna ek yonlendirici sinyaller eklenmesidir.
Bu projede semantic alignment ve deadline odulleri bunun ornegidir.

### Success Bonus
Basarili task tamamlamasi durumunda verilen sparse ek oduldur.
Faz 6'da semantic bagimliligini azaltmak ve gorev basarisini daha acik odullendirmek icin eklendi.

### Explanation Bank
Ajanin kararlarina ait semantic girdi, secilen aksiyon ve sonuc loglarinin biriktirildigi yapi.
Sonradan case study veya hata analizi icin kullanilir.

---

## 5. Faz 4: Baseline ve RL Kavramlari

### Baseline
Ana modelin performansini anlamli sekilde karsilastirabilmek icin kullanilan referans politika veya algoritmadir.

### Heuristic Baseline
Ogrenmeyen, elle yazilmis karar kurallaridir.
Ornek: `LocalOnly`, `EdgeOnly`, `CloudOnly`, `Random`, `GreedyLatency`, `GeneticAlgorithm`.

### RL (Reinforcement Learning)
Ajanin, ortamla etkileserek odul sinyali uzerinden iyi karar vermeyi ogrendigi yaklasimdir.

### Policy
Ajanin bir gozlem aldiginda hangi aksiyonu sececegini belirleyen karar mekanizmasidir.
Heuristic policy olabilir, ogrenilmis neural network policy de olabilir.

### PPO
Proximal Policy Optimization.
Bu projedeki ana RL algoritmasidir.
Karar politikasini nispeten stabil sekilde guncellemesi nedeniyle sik kullanilir.

### DQN
Deep Q-Network.
Ayrik aksiyon uzaylarinda sik kullanilan klasik RL algoritmasidir.
Bu projede PPO ile kiyaslanan baseline RL ajanlarindan biridir.

### A2C
Advantage Actor-Critic.
Policy ve value tahminini birlikte kullanan baska bir klasik RL algoritmasidir.
Bu projede PPO ve DQN ile beraber karsilastirilir.

### Checkpoint
Egitilmis modelin disk uzerine kaydedilmis halidir.
Ornek: `.zip` uzantili PPO modeli.

### Artifact
Bir deney sonunda uretilebilen somut ciktilardir.
Ornek: checkpoint, CSV log, figure, rapor.

---

## 6. Faz 5: Ablation ve Sentetik Deney Kavramlari

### Synthetic Environment
Gercek trace verisi yerine simulasyon veya sentetik gorev akislari uzerinde calisan ortamdir.
Faz 5'in ana calisma zemini burasidir.

### Cloud Collapse
Ajanin neredeyse her durumda cloud secmeye cokmesi durumudur.
Faz 5'te bu davranis kirilmaya calisildi.

### Dominant Action
Bir policy'nin deney boyunca en cok sectigi aksiyondur.
Ajanin tek bir davranisa cokup cokmedigini gormek icin kullanilir.

### Ablation Study
Bir modelin icindeki bilesenleri tek tek kaldirip hangi bilesenin ne kadar katkisi oldugunu olcmektir.
Ornek: `w/o semantics`, `w/o reward shaping`, `w/o mobility features`.

### Evaluation-Only Ablation
Yeni model egitmeden, mevcut modeli alip bazi ozellikleri test aninda kapatarak bakilan ablation turudur.
Bu daha hizli ama daha zayif bir kanittir.

### Retraining-Based Ablation
Her varyanti bastan egitip sonra karsilastirmaktir.
Bu daha pahali ama daha guclu bilimsel kanittir.

### Delta Analysis
Bir varyantin `full model`e gore ne kadar iyilesme veya kotulesme yarattigini sayisal farkla gostermektir.
Ornek: `w_o_mobility_features = -9.2 puan`.

### Full Model
Tum secili bilesenleri acik olan ana model varyantidir.
Ablation tablolarinda referans noktasi olarak kullanilir.

### Multi-Seed Evaluation
Ayni egitilmis modeli farkli evaluation seed'lerinde test etmektir.
Bu, test ortami oynakligina karsi daha guvenilir olcum verir.

### Multi-Seed Retraining
Modeli farkli seed'lerle bastan tekrar tekrar egitmek ve sonra ortalama/oynaklik hesaplamaktir.
Bu, tek bir sansli egitim kosusuna dayanma riskini azaltir.

### Variance / Std
Ayni deneyin farkli seed veya tekrarlar altinda ne kadar oynadigini gosterir.
Daha dusuk varyans, daha stabil sonuc anlamina gelebilir.

### Mobility Features
Baglanti kalitesi, mesafe veya hareketlilik kaynakli fiziksel sinyallerdir.
Faz 5'te en guclu ve en stabil bilesenlerden biri olarak bulundu.

### Battery Awareness
Karar verirken cihazin pil durumunu goz onune alma mekanizmasidir.

### Queue Awareness
Edge tarafindaki yuk veya bekleme durumunu state/reward tarafina yansitma mekanizmasidir.

---

## 7. Faz 6: Trace-Driven ve Generalization Kavramlari

### Trace-Driven Training
Modelin sentetik uretim yerine trace verisinden turetilmis gorev akislariyla egitilmesidir.
Faz 6'nin ana hedefidir.

### Trace Loader
Diskte bulunan ham trace dosyalarini veya kaydedilmis splitleri yukleyen katmandir.
Loader ile preprocessing/split mantiginin ayrismasi kod mimarisi acisindan onemlidir.

### Trace Processor
Ham veya yari-islenmis trace verisini egitimde kullanilabilir episode/task yapisina ceviren katmandir.

### Trace-to-Task Mapping
Trace veri alanlarinin proje icindeki task alanlarina nasil cevrildiginin tanimidir.
Ornek: `data_size -> size_bits`, `cpu_cycles -> cpu demand`, `deadline -> relative deadline`.

### Mapping Assumptions
Trace veri setinde birebir bulunmayan alanlar icin yaptigimiz acik varsayimlardir.
Bunlar dokumante edilmezse tez ve makale acisindan zayiflik yaratir.

### Domain Shift
Bir ortamda ogrenilen davranisin, farkli veri dagilimli baska bir ortamda ayni performansi gosterememesi durumudur.
Bu projede temel soru sudur:
"Synthetic ortamda iyi gorunen model, trace ortaminda da iyi mi?"

### Domain-Shift Evaluation
Synthetic ve trace gibi farkli dagilimlar arasinda capraz test yapmaktir.
Tipik iki yon:
- synthetic train -> trace test
- trace train -> synthetic test

### Cross-Domain Asymmetry
Domain shift sonuclarinin iki yonunun ayni siddette olmamasidir.
Ornek: synthetic train -> trace test cok iyi giderken, trace train -> synthetic test belirgin sekilde bozulabilir.
Bu, iki veri dagiliminin birbirine esit derecede benzemedigini gosterir.

### Generalization
Modelin sadece kendi egitim dagiliminda degil, yeni ve farkli kosullarda da makul performans gosterebilmesidir.
Domain-shift analizi bunun pratik olcusudur.

### Validation Split
Egitim sirasinda model secimi, erken durdurma veya ara kontrol icin kullanilan ayri veri bolumudur.
Bu projede trace tarafinda `val_episodes.json` bu rolu tasir.

### Test Split
Model secimi bittikten sonra son ve tarafsiz degerlendirme icin ayrilan veri bolumudur.
Bu projede `test_episodes.json`, Faz 6 kapanisindan once ayri hold-out test olarak kullanilmasi gereken bolumdur.

### Hold-Out Evaluation
Egitimde veya model seciminde kullanilmayan veri uzerindeki son kontrol kosusudur.
Amac, validation'da gorulen basarinin gercekten genellenip genellenmedigini dogrulamaktir.

### Switching Overhead
Partial offloading yaparken koordinasyon, parcaya ayirma, aktarim veya karar degisimi nedeniyle gelen ek maliyettir.
Faz 6'da bu maliyet statik degil, gorev boyutu, link kalitesi ve onceki aksiyon degisimiyle iliskili olacak sekilde modellendi.

---

## 8. Faz 7: Two-Stage Training Kavramlari

### Oracle Label
Heuristic veya maliyet fonksiyonundan turetilmis yari-optimal karar etiketidir.
Staged training veya imitation learning icin kullanilir.

### Imitation Learning / Supervised Pretraining
Ajanin RL oncesi, etikete dayali sekilde temel davranis ogrenmesidir.

### Two-Stage Training
Once pretraining, sonra RL fine-tuning yapilan egitim rejimidir.
Faz 7 icin planlanmistir.

### Fine-Tuning
Onceden ogrenilmis bir modeli yeni veri veya yeni objective ile tekrar ayarlama surecidir.

### Sample Efficiency
Bir modelin daha az deneyimle ne kadar iyi ogrenebildigini anlatan kavramdir.
Staged training bu acidan iyilestirme saglayabilir.

---

## 9. Faz 8: Graph-Aware Policy Kavramlari

### Graph-Aware Policy
Sistemi vektor yerine grafik yapisi olarak modelleyen policy yaklasimidir.
Cihazlar, edge sunucular, cloud ve baglanti iliskileri graph uzerinden temsil edilir.

### Graph State
Node ve edge ozelliklerinden olusan durum temsilidir.
Offloading problemi dogal olarak grafik yapisina yatkindir.

### GNN
Graph Neural Network.
Graph-aware policy kurmak icin kullanilan ana model ailesidir.

### Early Fusion / Late Fusion
Semantic prior gibi dis sinyallerin modelin hangi asamasinda graph temsiline katildigini anlatir.
- `Early fusion`: erken ozellik asamasinda birlesme
- `Late fusion`: karar asamasina yakin birlesme

---

## 10. Faz 9: Metrikler, Istatistik ve GUI Kavramlari

### Success Rate
Deadline icinde basariyla tamamlanan task oranidir.

### P95 Latency
Gecikme dagiliminin %95 persentilidir.
Tail-latency davranisini anlamak icin kullanilir.

### P99 Latency
Daha da uc gecikme davranisini gosteren %99 persentildir.

### Deadline Miss Ratio
Zamaninda tamamlanamayan gorevlerin oranidir.

### Avg Energy
Ortalama enerji tuketimidir.

### QoE
Quality of Experience.
Gecikme, basari ve bazen ceza bilesenlerinin birlesik kullanici deneyimi benzeri gostergesidir.

### Fairness
Kaynaklarin veya performansin cihazlar arasinda ne kadar dengeli dagildigini anlatir.

### Jitter
Gecikmenin ne kadar oynak oldugunu gosterir.
Ozellikle zaman hassas uygulamalarda onemlidir.

### Confidence Interval
Olculen ortalama performansin belirsizlik araligini gosterir.
Coklu seed sonuclarinin daha dogru okunmasina yardim eder.

### Wilcoxon Testi
Iki yontemin farkinin istatistiksel olarak anlamli olup olmadigini test etmek icin kullanilan parametrik olmayan testlerden biridir.

### GUI Experiment Mode
Model secimi, semantic mode, trace mode ve deney panellerinin arayuzden yonetildigi kullanim modu.

### Reward Decomposition Panel
GUI veya rapor uzerinde reward'u bilesenlerine ayirarak gosteren panel/yapi.

---

## 11. Faz 10: Self-Reflection ve Bilgi Birikimi Kavramlari

### Post-Hoc Analysis
Ajan karar verdikten sonra, neden iyi veya kotu sonuc ciktigini sonradan inceleme surecidir.

### Self-Reflection
Modelin hatali veya zayif karar oruntulerini tespit edip sonraki adimlarda bundan faydalanmaya calisan ust duzey analiz yaklasimidir.

### RAG / Retrieval-Augmented Generation
Gecmis bilgi bankasindan uygun ornek veya aciklamayi cekip yeni karar/aciklama surecine dahil etme yaklasimidir.

### Explanation Memory
Daha onceki semantic kararlarin, aciklamalarin ve sonuclarin tekrar kullanilabilir hafizasi.

---

## 12. Raporlama ve Okuma Mantigi

### Full Model vs Variant
Raporlarda `full model` referans modeldir; diger satirlar onun bir bileseni kapatilmis veya degistirilmis varyantlardir.

### Figure
Grafik dosyasidir. Genelde trendleri hizli okumak icin kullanilir.

### CSV Log
Ham deney kaydidir. Raporlar ve figure'lar genelde buradan uretilir.

### Workflow Log
Belirli bir script veya deney turunun urettigi toplu kayit dosyasidir.

---

## 13. Bu Sozlugu Nasil Kullanmali?

- Bir raporda veya TODO maddesinde bilmediginiz bir terim gordugunuzde once bu dosyaya bakin.
- Yeni bir kavram ortaya cikarsa bu dosyaya ekleyelim.
- Faz raporlarinda uzun aciklama yerine burada tanim verip raporlari daha okunur tutabiliriz.
- Bu sozluk, proje ilerledikce yasayan bir dokuman gibi buyutulmelidir.

