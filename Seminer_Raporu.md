# Seminer Raporu

## LLM Destekli Derin Pekiştirmeli Öğrenme ile Yeni Nesil Edge Ağlarda Semantik Farkındalıklı Görev Aktarımı ve Dinamik Kaynak Tahsisi

## Özet

Bu seminer raporunda, IoT ve mobil edge computing ortamlarında görev aktarımı problemini ele alan çalışmamı anlatıyorum. Çalışmanın temel amacı, bir IoT cihazında oluşan görevin cihaz üzerinde mi, edge sunucuda mı, kısmi olarak edge tarafında mı, yoksa cloud üzerinde mi çalıştırılacağına daha doğru ve daha bağlama duyarlı şekilde karar verebilen bir yapay zeka sistemi geliştirmektir.

Task offloading problemi ilk bakışta basit bir yönlendirme problemi gibi görünse de gerçekte çok değişkenli ve dinamik bir optimizasyon problemidir. Karar verilirken görev boyutu, CPU ihtiyacı, deadline, cihaz bataryası, ağ kalitesi, edge sunucu yükü, enerji tüketimi, kuyruk bekleme süresi ve kullanıcı deneyimi gibi birçok ölçüt birlikte değerlendirilmelidir. Bu yüzden yalnızca sabit kurallar kullanan yöntemler dinamik ortamlarda yetersiz kalabilmektedir.

Bu çalışmada klasik heuristic yaklaşımların ötesine geçerek derin pekiştirmeli öğrenme tabanlı bir task offloading framework'ü geliştirdim. Sistemde PPO, DQN ve A2C gibi RL algoritmaları kullanıldı; LocalOnly, EdgeOnly, CloudOnly, Random, GreedyLatency ve GeneticAlgorithm gibi baseline yöntemlerle karşılaştırmalar yapıldı. Ayrıca büyük dil modeli veya semantic analyzer tarafından üretilen semantik öncül bilgi, ajan kararlarına dahil edildi. Böylece ajan yalnızca fiziksel sistem metriklerine değil, görevin anlamına ve önceliğine de duyarlı hale getirildi.

Çalışma boyunca proje kademeli olarak geliştirildi. Önce tekrarlanabilir deney altyapısı kuruldu, ardından gerçekçi simülasyon ortamı ve ortak state yapısı hazırlandı. Daha sonra semantic prior, reward shaping, baseline genişletmesi, sistematik ablation study, trace-driven training, two-stage training ve graph-aware policy çalışmaları gerçekleştirildi. Son aşamada sistem, klasik vektör tabanlı RL temsilinden graph tabanlı ilişki temsiline doğru genişletilmeye başlandı.

Elde edilen sonuçlar, semantic-aware DRL yaklaşımının özellikle PPO tabanlı modelde güçlü bir aday olduğunu göstermektedir. Sentetik deneylerde PPO full model yaklaşık %76.17 başarı oranına ulaşmış, trace-driven hold-out testte PPO modeli %99.60 başarı göstermiştir. Two-stage training aşamasında ise pretrained + PPO yaklaşımı scratch PPO'ya göre +12.20 puan başarı artışı sağlamıştır. Ablation sonuçları partial offloading ve mobility features bileşenlerinin karar kalitesi açısından kritik olduğunu göstermiştir.

Bu rapor, yapılan çalışmayı seminer dersi kapsamında anlaşılır, sunulabilir ve ileride journal makalesine dönüştürülebilir bir formatta açıklamaktadır.

## 1. Giriş

Günümüzde IoT cihazları, akıllı şehirler, otonom sistemler, sağlık uygulamaları, endüstriyel otomasyon ve mobil uygulamalar gibi birçok alanda yoğun şekilde kullanılmaktadır. Bu cihazların ortak problemi, çoğu zaman sınırlı işlem gücüne, sınırlı batarya kapasitesine ve değişken bağlantı kalitesine sahip olmalarıdır. Buna karşılık bu cihazlar üzerinde çalışan görevler giderek daha karmaşık hale gelmektedir.

Örneğin bir mobil cihazdan gelen görüntü işleme görevi, bir sensör ağından gelen anomali tespit görevi veya gerçek zamanlı bir araç takip görevi cihaz üzerinde çalıştırılabilir. Ancak cihazın bataryası düşükse, CPU kapasitesi yetersizse veya görev deadline açısından kritikse, görevin edge sunucuya ya da cloud ortamına aktarılması daha mantıklı olabilir. Bu noktada task offloading problemi ortaya çıkar.

Task offloading, bir görevin nerede çalıştırılacağına karar verme problemidir. Temel seçenekler şunlardır:

- Local processing: Görev cihazın kendi üzerinde çalıştırılır.
- Edge offloading: Görev yakındaki edge sunucuya aktarılır.
- Cloud offloading: Görev merkezi cloud altyapısına gönderilir.
- Partial offloading: Görevin bir bölümü local, bir bölümü edge/cloud tarafında çalıştırılır.

Bu çalışmada özellikle partial offloading önemli bir bileşen olarak ele alınmıştır. Çünkü gerçek sistemlerde görevleri yalnızca tamamen local veya tamamen cloud şeklinde düşünmek çoğu zaman yeterli değildir. Görevin belirli bir oranını edge tarafına aktarmak, gecikme ve enerji tüketimi arasında daha dengeli sonuçlar verebilir.

Bu problemin zor olmasının nedeni, kararın tek bir metriğe bağlı olmamasıdır. En düşük gecikmeyi seçmek her zaman en iyi çözüm olmayabilir; çünkü bu durumda enerji tüketimi artabilir. En düşük enerjiyi seçmek de her zaman iyi değildir; çünkü deadline kaçırılabilir. Aynı şekilde cloud yüksek işlem kapasitesi sunsa da iletişim gecikmesi ve aktarım maliyeti nedeniyle her görev için uygun olmayabilir.

Bu nedenle bu çalışmada problemi çok amaçlı bir karar problemi olarak ele aldım. Temel hedef, görevlerin başarıyla ve deadline içinde tamamlanmasını sağlarken gecikme, enerji tüketimi, kuyruk bekleme süresi ve kullanıcı deneyimi gibi ölçütleri dengeli şekilde optimize etmektir.

## 2. Çalışmanın Amacı ve Kapsamı

Bu çalışmanın ana amacı, IoT ve edge computing ortamlarında task offloading kararlarını daha akıllı, daha uyarlanabilir ve daha açıklanabilir hale getiren bir sistem geliştirmektir.

Çalışmanın temel araştırma sorusu şu şekilde ifade edilebilir:

> LLM tabanlı semantik görev analizi ve derin pekiştirmeli öğrenme birlikte kullanıldığında, IoT edge ağlarında task offloading kararlarının başarı oranı ve karar kalitesi artırılabilir mi?

Bu ana sorunun altında birkaç alt soru bulunmaktadır:

1. Klasik heuristic yöntemler ile RL tabanlı yöntemler arasında anlamlı bir performans farkı oluşuyor mu?
2. PPO, DQN ve A2C gibi farklı RL algoritmaları task offloading probleminde nasıl davranıyor?
3. LLM veya semantic analyzer tarafından üretilen semantic prior, ajan kararlarına katkı sağlıyor mu?
4. Partial offloading, mobility features, reward shaping ve confidence gibi bileşenler performansı ne kadar etkiliyor?
5. Sentetik ortamda öğrenilen politikalar trace-driven veriye taşındığında genelleme yapabiliyor mu?
6. Supervised pretraining + PPO fine-tuning şeklindeki two-stage training yaklaşımı scratch PPO'ya göre avantaj sağlıyor mu?
7. Sistemi düz vektör state yerine graph yapısı olarak temsil etmek, daha bağlama duyarlı kararlar üretmeye yardımcı olabilir mi?

Bu kapsamda proje yalnızca tek bir model eğitme çalışması olarak değil, aşamalı bir araştırma pipeline'ı olarak tasarlanmıştır. Her fazda önceki fazın eksikleri belirlenmiş, daha sonra bu eksikleri giderecek yeni modüller eklenmiştir.

## 3. Literatür ve Arka Plan

Task offloading literatüründe farklı yaklaşım aileleri bulunmaktadır. İlk grup sabit veya kural tabanlı yöntemlerdir. Örneğin bütün görevleri local çalıştırmak, bütün görevleri edge'e göndermek veya gecikmesi en düşük görünen yolu seçmek bu gruba girer. Bu yöntemler basit, hızlı ve açıklanabilirdir; ancak dinamik sistemlerde değişen ağ koşullarına, batarya durumuna ve sunucu yüklerine yeterince uyum sağlayamazlar.

İkinci grup optimizasyon ve meta-sezgisel yöntemlerdir. Genetic Algorithm gibi yöntemler karar uzayında arama yaparak daha iyi çözümler bulmaya çalışır. Bu yöntemler klasik heuristic yaklaşımlardan daha esnek olabilir; ancak gerçek zamanlı karar gerektiren senaryolarda hesaplama maliyeti ve genelleme kabiliyeti sınırlı kalabilir.

Üçüncü grup derin pekiştirmeli öğrenme tabanlı yöntemlerdir. DRL yaklaşımında ajan, ortamla etkileşerek hangi koşulda hangi aksiyonun daha iyi sonuç verdiğini öğrenir. Task offloading problemi ayrık aksiyon uzayına sahip olduğu için DQN gibi value-based modeller kullanılabilir. Ayrıca PPO ve A2C gibi actor-critic tabanlı yöntemler de bu problem için uygundur.

Bu projede ana model olarak PPO seçilmiştir. PPO'nun tercih edilme nedeni, politika güncellemelerini kontrollü şekilde yapması ve birçok RL probleminde dengeli/stabil sonuçlar vermesidir. DQN, ayrık aksiyon uzayı için güçlü bir temel RL baseline olarak kullanılmıştır. A2C ise PPO ile DQN arasında actor-critic yaklaşımını temsil eden bir karşılaştırma modeli olarak eklenmiştir.

SAC ve DDPG gibi modeller bu çalışmada ana deney ailesine dahil edilmemiştir. Bunun nedeni, bu modellerin daha çok continuous action space için tasarlanmış olmasıdır. Bu projede aksiyon uzayı ayrık olarak tanımlanmıştır: local, edge_25, edge_50, edge_75, edge_100 ve cloud.

Literatürdeki önemli eksiklerden biri, birçok çalışmanın yalnızca fiziksel metriklere odaklanmasıdır. Oysa görevlerin semantik özellikleri de offloading kararında önemli olabilir. Örneğin acil, kritik veya yüksek doğruluk gerektiren bir görev ile düşük öncelikli bir görev aynı şekilde ele alınmamalıdır. Bu çalışmada bu boşluğu kapatmak için semantic-aware offloading yaklaşımı kullanılmıştır.

Çalışmada ayrıca AgentVNE benzeri yaklaşımlardan ilham alınmıştır. Özellikle staged training, prior distribution, teacher policy, oracle label ve graph-aware temsil gibi fikirler bu yönde geliştirilmiştir. Amaç, klasik "PPO ile offloading yaptım" yaklaşımından daha ileri giderek LLM-guided, semantic-aware ve graph-aware bir task offloading framework'ü oluşturmaktır.

## 4. Problem Tanımı

Bu çalışmada bir IoT cihazı, edge sunucular ve cloud altyapısından oluşan bir sistem ele alınmıştır. Her karar anında sistemde bir görev oluşur ve ajan bu görevin hangi ortamda çalıştırılacağına karar verir.

Görev özellikleri genel olarak şu bilgilerden oluşur:

- Görev boyutu
- CPU cycle ihtiyacı
- Deadline
- Öncelik veya aciliyet
- İletişim ihtiyacı
- Semantik karmaşıklık

Sistem özellikleri ise şu bilgileri içerir:

- SNR veya link kalitesi
- Cihaz batarya seviyesi
- Edge server load
- Edge enerji durumu
- Kuyruk bekleme etkisi
- Cloud/edge aktarım maliyeti
- Önceki aksiyon değişimine bağlı switching overhead

Aksiyon uzayı 6 ayrık karardan oluşur:

| Aksiyon | Anlamı |
|---|---|
| 0 | Local processing |
| 1 | Edge %25 partial offloading |
| 2 | Edge %50 partial offloading |
| 3 | Edge %75 partial offloading |
| 4 | Edge %100 offloading |
| 5 | Cloud offloading |

Bu aksiyon yapısı sayesinde sistem yalnızca "local mi cloud mu?" şeklinde ikili karar vermemekte, kısmi offloading kararlarını da öğrenebilmektedir.

Amaç fonksiyonu tek bir metrikten oluşmaz. Reward yapısında genel olarak şu bileşenler dikkate alınır:

- Gecikme cezası
- Enerji tüketimi cezası
- Deadline kaçırma cezası
- Başarılı tamamlanma bonusu
- Semantic alignment etkisi
- Confidence etkisi
- Queue ve edge load etkisi
- Partial offloading maliyeti

Bu yüzden problem, çok amaçlı ve dinamik bir karar problemi olarak ele alınmıştır.

### 4.1 Kullanılan Matematiksel Modeller ve Formüller

Bu çalışmada karar ortamını gerçekçi hale getirmek için sabit gecikme veya sabit enerji değerleri kullanmak yerine, haberleşme, işlem, enerji ve kuyruk etkilerini temsil eden modeller kullanılmıştır. Bu modeller raporun ana yöntem kısmında özellikle belirtilmelidir; çünkü ajan aslında bu fiziksel modellerin ürettiği sonuçlara göre öğrenmektedir.

#### Kablosuz Haberleşme Modeli: Path Loss + Shannon-Hartley

Cihaz ile edge sunucu arasındaki veri hızı sabit kabul edilmemiştir. Önce cihaz ile edge sunucu arasındaki Öklid mesafesi hesaplanır:

```text
d = sqrt((x_device - x_edge)^2 + (y_device - y_edge)^2)
```

Daha sonra path loss modeli ile kanal kazancı hesaplanır:

```text
h = d^(-alpha)
```

Burada `alpha` path loss exponent değeridir. Güncel kodda bu değer:

```text
alpha = 4
```

olarak kullanılmaktadır. Mesafe arttıkça `h` azalır; yani cihaz edge sunucudan uzaklaştıkça kanal kalitesi düşer.

Sonra SINR hesabı yapılır:

```text
SINR = (P_tx * h) / (N0 + I)
```

Burada:

- `P_tx`: transmission power
- `h`: kanal kazancı
- `N0`: noise power
- `I`: interference

Kodda kullanılan temel değerler:

```text
P_tx = 0.5 W
N0 = 1e-13 W
B = 20 MHz
```

Bu SINR değeri Shannon-Hartley kapasite formülüne verilir:

```text
R = B * log2(1 + SINR)
```

Burada `R` anlık veri hızıdır. Yani transmission time sabit değildir; kanal kalitesine göre değişir.

```text
T_tx = task_size_bits / R
```

Bu yüzden cihaz edge'e yakınsa veri aktarımı hızlı ve ucuz olur; uzaksa aktarım süresi ve enerji maliyeti artar.

Not olarak, kodda bazı yerlerde `snr_norm` adı kullanılsa da güncel state içinde bu değer pratikte doğrudan SNR değil, Shannon modelinden gelen veri hızının normalize edilmiş halidir:

```text
snr_norm = min(1, datarate / 50e6)
```

Bu nedenle raporda bunu “SNR/link quality göstergesi” veya “normalized datarate” olarak açıklamak daha doğru olur.

#### İletim Enerjisi Modeli

Bir görevin edge veya cloud tarafına aktarılması gerekiyorsa, cihaz transmission energy harcar. Bu enerji aktarım süresine bağlıdır:

```text
E_tx = P_tx * T_tx
```

Partial offloading durumunda yalnızca görevin offload edilen kısmı aktarılır:

```text
E_tx_partial = P_tx * ((r * task_size_bits) / R)
```

Burada `r`, edge tarafına aktarılan oranı gösterir:

```text
r in {0.25, 0.50, 0.75, 1.0}
```

Bu modelin önemli sonucu şudur:

> Ağ kalitesi kötüleşirse `R` düşer, `T_tx` artar ve transmission energy yükselir.

Yani enerji tüketimi yalnızca görevin boyutuna değil, aynı zamanda cihaz-edge mesafesine, path loss'a, noise/interference etkisine ve Shannon kapasitesine bağlıdır.

#### İşlem Gecikmesi Modeli

Bir görevin işlem süresi CPU cycle ihtiyacı ve işlemci frekansına göre hesaplanır:

```text
T_comp = C / f
```

Burada:

- `C`: task CPU cycles
- `f`: işlemci frekansı

Kodda kullanılan tipik frekanslar:

```text
f_local = 1 GHz
f_edge ≈ 2 GHz
f_cloud = 5 GHz
```

Bu nedenle cloud işlem açısından hızlıdır; ancak cloud kararında ek transmission ve sabit cloud latency maliyeti vardır.

#### CPU Enerji Modeli: DVFS Tabanlı Model

Enerji modelinde kullandığımız diğer önemli model DVFS, yani Dynamic Voltage and Frequency Scaling modelidir. Bu modelde CPU enerji tüketimi frekans ve işlem süresine bağlıdır.

Edge sunucu için kullanılan form:

```text
E_comp = kappa * f^3 * T_comp
```

`T_comp = C / f` olduğu için bu ifade şu şekilde de okunabilir:

```text
E_comp = kappa * f^2 * C
```

Burada:

- `kappa`: effective switched capacitance
- `f`: CPU frequency
- `C`: CPU cycles

Kodda:

```text
kappa = 1e-28
```

olarak kullanılmaktadır.

Bu modelin anlamı şudur:

> Frekans yükseldikçe işlem süresi azalır, fakat enerji maliyeti karesel/kübik etki nedeniyle artar.

Edge server tarafında frekans yük durumuna göre değişmektedir:

```text
if current_load > 2:
    current_freq = max_freq
else:
    current_freq = 0.7 * max_freq
```

Yani edge sunucu düşük yükte daha düşük frekansla enerji tasarrufu yapar; yüksek yükte daha hızlı çalışır ama daha fazla enerji harcar.

Local işlem için cihaz enerjisi yaklaşık şu şekilde hesaplanır:

```text
E_local = kappa * f_local^2 * C
```

Simulation GUI tarafında enerji farklarını görünür yapmak için bazı yerlerde ayrıca `ENERGY_SCALE_FACTOR` kullanılmıştır. Ancak temel model yine CPU cycle ve frekans ilişkisine dayanmaktadır.

#### Partial Offloading Gecikme ve Enerji Modeli

Partial offloading bu çalışmanın önemli farklarından biridir. Eğer görevin `r` oranı edge'e aktarılıyorsa:

```text
local_cycles = (1 - r) * C
edge_cycles = r * C
edge_bits = r * task_size_bits
```

Local tarafın süresi:

```text
T_local_part = ((1 - r) * C) / f_local
```

Edge tarafın aktarım ve işlem süresi:

```text
T_edge_part = (r * task_size_bits) / R + (r * C) / f_edge + T_queue
```

Partial offloading paralel çalıştığı için toplam gecikme yaklaşık şu şekilde alınır:

```text
T_partial = max(T_local_part, T_edge_part) + T_overhead
```

Enerji ise cihaz açısından local computation energy ve transmission energy toplamıdır:

```text
E_partial = E_local_part + E_tx_partial
```

Bu yapı sayesinde ajan şunu öğrenebilir:

> Görevin tamamını cloud'a göndermek yerine bir kısmını local, bir kısmını edge tarafında çalıştırmak gecikme ve enerji arasında daha dengeli sonuç verebilir.

#### Queue ve Congestion Modeli

Edge ve cloud tarafında yük arttıkça gecikme de artmaktadır. SimPy tarafında edge server için resource ve queue mantığı kullanılmıştır. RL environment tarafında ise queue/load etkisi yaklaşık ceza olarak gecikmeye eklenmiştir:

```text
T_edge_queue = 0.015 * queue_length + 0.02 * current_load
```

Cloud için:

```text
T_cloud_congestion = 0.02 * cloud_queue + 0.03 * cloud_load
```

Bu nedenle en yakın edge her zaman en iyi seçenek olmayabilir. Eğer edge sunucu yoğun ise queue delay kararı değiştirebilir.

#### Cloud Gecikme Modeli

Cloud tarafında işlem hızlıdır; fakat sabit bir cloud latency maliyeti vardır:

```text
T_cloud = T_tx + 0.1 + C / f_cloud + T_cloud_congestion
```

Simulation tarafında cloud için gidiş ve dönüş gecikmesi de temsil edilmektedir. Bu yüzden cloud, yüksek CPU gerektiren görevlerde avantajlı olabilir; ancak küçük görevlerde veya zayıf ağ koşullarında pahalı hale gelebilir.

#### Switching Overhead Modeli

Partial offloading kararlarında ek koordinasyon maliyeti vardır. Bu maliyet görevin boyutuna, link kalitesine ve önceki aksiyonla yeni aksiyonun farklı olup olmamasına göre hesaplanır:

```text
T_overhead = coordination_factor *
             (0.01 + 0.02 * size_factor + mobility_penalty + transition_penalty)
```

Burada:

```text
size_factor = min(1, task_size_bits / 10e6)
mobility_penalty = (1 - link_quality_factor) * 0.03
transition_penalty = 0.015 if previous_action != action else 0
```

Bu model, partial offloading'in bedava olmadığını gösterir. Görevi bölmek bazen faydalıdır; ancak koordinasyon maliyeti de hesaba katılmalıdır.

#### State Normalizasyonu

RL ajanına verilen state değerleri doğrudan fiziksel birimlerle değil, normalize edilmiş değerlerle verilir:

```text
datarate_norm = min(1, R / 50e6)
size_norm = min(1, task_size_bits / 10e6)
cpu_norm = min(1, cpu_cycles / 1e10)
battery_norm = battery / 10000
load_norm = edge_current_load / 10
edge_energy_norm = edge_remaining_energy / edge_energy_budget
```

Bu normalizasyon, PPO/DQN/A2C gibi modellerin farklı ölçeklerdeki sayıları daha stabil öğrenmesini sağlar.

## 5. Önerilen Yöntem

Önerilen sistemin temel fikri, task offloading kararını yalnızca fiziksel sistem metriklerine göre değil, semantik görev bilgisi ve öğrenilmiş karar politikası ile birlikte vermektir.

Sistemin genel akışı şu şekildedir:

1. Ortamda yeni bir görev oluşur.
2. Görev ve sistem durumu state builder tarafından sayısal state'e dönüştürülür.
3. Semantic analyzer görevin anlamını analiz eder.
4. Semantic prior, 6 aksiyon üzerinde bir olasılık dağılımı üretir.
5. RL ajanı fiziksel state + semantic prior bilgisini kullanarak aksiyon seçer.
6. Ortam bu aksiyonu uygular.
7. Gecikme, enerji, deadline ve başarı bilgisine göre reward hesaplanır.
8. Ajan bu deneyimden öğrenir.

### 5.1 State Temsili

İlk aşamalarda ajan 11 boyutlu bir state kullanmıştır. Bu state iki ana parçadan oluşur:

- Fiziksel özellikler
- Semantik prior özellikleri

Fiziksel özellikler şunları temsil eder:

- SNR veya ağ kalitesi
- Task size
- CPU cycle ihtiyacı
- Battery percent
- Edge server load

Semantic prior ise 6 aksiyon için olasılık eğilimi verir:

- P(Local)
- P(Edge %25)
- P(Edge %50)
- P(Edge %75)
- P(Edge %100)
- P(Cloud)

Daha sonraki fazlarda edge energy gibi ek fiziksel sinyallerle state 12 boyutlu hale gelmiştir. Bu temsil, ajan için hem fiziksel hem de semantik bağlamı birlikte taşır.

### 5.2 Semantic Prior

Semantic prior, LLM veya rule-based semantic analyzer tarafından üretilen aksiyon eğilimidir. İlk tasarımda LLM çıktısı tek bir öneri gibi düşünülebilirdi. Ancak bu yaklaşım çok katı olduğu için daha sonra prior distribution yaklaşımına geçilmiştir.

Bu yaklaşımla semantic analyzer artık yalnızca "edge seç" veya "cloud seç" demek yerine, 6 aksiyon üzerinde bir olasılık dağılımı üretir. Örneğin deadline kritik ve görev büyükse edge veya cloud aksiyonlarının olasılığı artabilir. Eğer görev küçük ve batarya uygunsa local seçenek daha anlamlı hale gelebilir.

Confidence skoru da bu yapıya dahil edilmiştir. Semantic analyzer kararından çok eminse prior daha keskin olur; emin değilse dağılım daha dengeli hale getirilir. Böylece ajan LLM önerisine körü körüne bağlı kalmaz, ancak semantik bilgiden faydalanır.

### 5.3 Reward Shaping

Reward shaping, ajanı daha doğru davranışlara yönlendirmek için kullanılan ödül tasarımıdır. Bu projede reward yalnızca başarı/başarısızlık şeklinde verilmemiş, farklı bileşenlerle zenginleştirilmiştir.

Örneğin:

- Deadline içinde tamamlanan görevlere başarı bonusu verilir.
- Yüksek gecikme ceza getirir.
- Yüksek enerji tüketimi ceza getirir.
- Semantic prior ile uyumlu ve mantıklı kararlar desteklenir.
- Partial offloading sırasında oluşan switching overhead dikkate alınır.

Bu yapı sayesinde ajan yalnızca kısa vadeli kazanca değil, daha dengeli ve sistem açısından anlamlı kararlara yönlendirilmiştir.

### 5.4 Kullanılan RL Modelleri

Çalışmada üç ana RL modeli kullanılmıştır:

| Model | Rolü |
|---|---|
| PPO | Ana model ve en güçlü aday |
| DQN | Ayrık aksiyon uzayı için value-based baseline |
| A2C | Actor-critic karşılaştırma modeli |

PPO, özellikle stabil politika güncellemesi nedeniyle ana model olarak seçilmiştir. DQN ve A2C ise PPO'nun gerçekten güçlü olup olmadığını anlamak için karşılaştırma amaçlı kullanılmıştır.

### 5.5 Baseline Modeller

Sistemin başarısını anlamlı şekilde yorumlayabilmek için yalnızca RL modelleri değil, klasik baseline yöntemler de eklenmiştir:

- LocalOnly
- EdgeOnly
- CloudOnly
- Random
- GreedyLatency
- GeneticAlgorithm

Bu baseline'lar, öğrenen modelin basit kurallara göre ne kadar avantaj sağladığını göstermek için kullanılmıştır.

## 6. Deneysel Geliştirme Süreci

Bu proje tek seferde tamamlanan bir çalışma değildir. Aşamalı bir araştırma süreci olarak ilerlemiştir. Her fazda sistemin bir eksik tarafı ele alınmış ve bir sonraki faz için daha sağlam bir zemin hazırlanmıştır.

### 6.1 Faz 1: Reproducibility ve Kod Temizliği

İlk fazda projenin araştırma çalışmasına uygun hale getirilmesi hedeflenmiştir. Bu aşamada kod tabanı modüler hale getirilmiş, config ve results klasör yapısı düzenlenmiştir.

Yapılan temel işler:

- `src/` altında core modüller oluşturuldu.
- `configs/` klasörü ile deneylerin config-driven yürütülmesi sağlandı.
- `results/raw`, `results/figures`, `results/tables` gibi klasörler oluşturuldu.
- `set_seed()` fonksiyonu ile Python, NumPy, Torch ve Gymnasium tarafında rastgelelik kontrol altına alındı.
- Deney sonuçlarının CSV/JSON olarak kaydedilmesi için logging altyapısı kuruldu.

Bu fazın önemi, projeyi yalnızca çalışan bir kod olmaktan çıkarıp tekrarlanabilir bir araştırma altyapısına dönüştürmesidir.

### 6.2 Faz 2: Eğitim Ortamı ve Simülasyon Hizalaması

İkinci fazda RL eğitim ortamı ile simülasyon ortamı hizalanmıştır. Başlangıçta eğitim ortamında kullanılan mock yapılar ile gerçek simülasyon akışı arasında farklar bulunmaktaydı. Bu farklar ileride domain shift problemine yol açabileceği için ortak state builder yaklaşımı geliştirilmiştir.

Yapılan temel işler:

- Ortak `state_builder` yazıldı.
- RL environment ve simulation environment aynı state üretim mantığına yaklaştırıldı.
- Reward hesaplama kodu ayrı bir modüle taşındı.
- Tek adımlı episode yapısından çok adımlı episode yapısına geçildi.
- Ajanın batarya ve uzun vadeli etkileri öğrenebilmesi için episode içinde birden fazla görev çözmesi sağlandı.

Bu fazdan sonra ajan artık her görevde sıfırlanan basit bir karar verici olmaktan çıkmış, ardışık görevler üzerinde uzun vadeli politika öğrenen bir yapıya dönüşmüştür.

### 6.3 Faz 3: LLM Entegrasyonunu Gerçek Katkıya Dönüştürme

Üçüncü fazda semantic analyzer sistemin gerçek bir parçası haline getirilmiştir. Önceden LLM çıktısı daha çok açıklama veya tekil öneri gibi düşünülürken, bu fazda semantic prior distribution yaklaşımı uygulanmıştır.

Yapılan temel işler:

- 6 boyutlu action prior üretimi eklendi.
- Confidence skoru prior dağılımına dahil edildi.
- LLM çıktıları için structured JSON parsing ve fallback mekanizması geliştirildi.
- Ajan kararlarını ve LLM açıklamalarını kaydetmek için explanation bank oluşturuldu.

Bu fazla birlikte sistem, yalnızca fiziksel state ile karar veren RL ajanı olmaktan çıkıp semantik görev bilgisini de kullanan bir offloading ajanına dönüşmüştür.

### 6.4 Faz 4: Baseline Ailesinin Genişletilmesi

Dördüncü fazda karşılaştırma altyapısı güçlendirilmiştir. Bir modelin başarılı olup olmadığını anlayabilmek için güçlü ve çeşitli baseline'lara ihtiyaç vardır. Bu nedenle fixed policy, heuristic, meta-heuristic ve RL tabanlı modeller aynı değerlendirme altyapısına alınmıştır.

Eklenen baseline'lar:

- LocalOnly
- EdgeOnly
- CloudOnly
- Random
- GreedyLatency
- GeneticAlgorithm

RL tarafında ise PPO v2, DQN v2 ve A2C v2 modelleri sisteme dahil edilmiştir.

Bu fazda ayrıca Stable-Baselines3 modellerinin değerlendirilmesi sırasında oluşan batch dimension ve action type uyumsuzlukları giderilmiştir. SB3 modellerinin predict çıktıları doğru şekilde işlenmiş ve environment ile uyumlu hale getirilmiştir.

Faz 4'ün ilk sonuçları, RL tarafında anlamlı performans sinyali olduğunu göstermiştir. Ancak bu aşamada asıl soru şuydu:

> PPO başarısının ne kadarı semantic bileşenlerden, ne kadarı ortam kalibrasyonundan ve ne kadarı RL algoritmasının kendisinden geliyor?

Bu soru Faz 5 ablation study ile ele alınmıştır.

### 6.5 Faz 5: Sistematik Ablation Study

Beşinci faz, projenin bilimsel açıdan en kritik aşamalarından biridir. Çünkü bu fazda modelin içindeki bileşenler tek tek çıkarılarak hangi bileşenin gerçekten katkı sağladığı ölçülmüştür.

Bu fazda yapılan ana işler:

- PPO, DQN ve A2C için multi-seed retraining yapıldı.
- Evaluation-only ablation ve retraining-based ablation ayrımı kuruldu.
- Cloud-only collapse davranışı kırıldı.
- Edge enerji bütçesi, mobility features ve dominant action takibi eklendi.
- Full model ve ablation varyantları karşılaştırıldı.

Faz 5 sonunda full model retraining sonuçları şu şekildedir:

| Algoritma | Success Rate | P95 Latency | Avg Energy | Dominant Action |
|---|---:|---:|---:|---:|
| PPO | %76.17 ± 10.63 | 2.799 | 0.0768 | 3 |
| DQN | %74.73 ± 11.23 | 2.802 | 0.0791 | 3 |
| A2C | %74.73 ± 11.23 | 2.802 | 0.0791 | 3 |

Bu sonuçlara göre PPO genel olarak en güçlü aday olarak kalmıştır. Ancak DQN ve A2C de yakın sonuçlar üretmiştir. Üç modelin de dominant action olarak `action=3`, yani Edge %75 kararına yönelmesi, sistemin önceki cloud collapse davranışından uzaklaştığını göstermiştir.

Ablation sonuçlarında en önemli bulgu mobility features ve partial offloading bileşenlerinin kritik olmasıdır. Özellikle mobility features çıkarıldığında üç algoritmada da sistematik başarı düşüşü görülmüştür. Partial offloading ise özellikle evaluation-only analizlerde başarı ve tail-latency üzerinde güçlü etki göstermiştir.

Faz 5'in ana çıkarımı şudur:

> Sentetik ortamda metodolojik olarak daha sağlam bir deney zemini kurulmuş, cloud collapse kırılmış ve hangi bileşenlerin daha kritik olduğu sayısal olarak görünür hale getirilmiştir.

Ancak Faz 5 sonunda bazı açık noktalar kalmıştır. Özellikle reward shaping etkisi tam net ayrışmamış, semantic bileşenlerin etkisi bazı koşullarda sınırlı görünmüş ve ajanlar hâlâ tek bir dominant action etrafında toplanma eğilimi göstermiştir. Bu nedenle Faz 6'da trace-driven doğrulama aşamasına geçilmiştir.

### 6.6 Faz 6: Trace-Driven Training ve Domain Shift Analizi

Altıncı fazda sistem sentetik ortamdan çıkarılarak trace-driven veriler üzerinde test edilmiştir. Bu aşamanın amacı, sentetik ortamda elde edilen bulguların gerçek veri dağılımına daha yakın episode splitleri üzerinde ne kadar geçerli olduğunu görmektir.

Kullanılan splitler:

- `train_episodes.json`
- `val_episodes.json`
- `test_episodes.json`

Bu fazda PPO modeli trace verisiyle yeniden eğitilmiş ve `models/ppo/trace_training/ppo_v3_trace_best.zip` checkpoint'i üretilmiştir.

Trace training özet sonuçları:

- Episode sayısı: 532
- Ortalama success rate: %98.59
- Ortalama delay: 0.3702 s
- Ortalama enerji: 0.0318
- Validation success rate: %99.20

Hold-out test sonuçları:

| Split | Mean Success | Mean Delay | Mean Energy |
|---|---:|---:|---:|
| Train | %99.38 | 0.3245 s | 0.0128 |
| Val | %99.00 | 0.3453 s | 0.0207 |
| Test | %99.60 | 0.3317 s | 0.0128 |

Bu sonuçlar, trace ortamında PPO'nun oldukça stabil bir politika öğrendiğini göstermiştir. Train, validation ve test başarı oranlarının birbirine çok yakın olması, belirgin bir overfitting işareti olmadığını göstermektedir.

Faz 6'da ayrıca domain shift analizi yapılmıştır:

| Train Domain | Test Domain | Success Rate | P95 Latency | Avg Energy | Dominant Action |
|---|---|---:|---:|---:|---:|
| Synthetic | Trace | %99.60 | 0.5618 | 0.0255 | 3 |
| Trace | Synthetic | %53.20 | 4.6883 | 0.0101 | 4 |

Bu tablo önemli bir bulgu üretmiştir. Synthetic -> trace yönü güçlü çıkarken, trace -> synthetic yönü zayıf kalmıştır. Bu durum domain shift'in simetrik olmadığını, yani iki ortam arasında cross-domain asymmetry bulunduğunu göstermektedir.

Faz 6'nın ana çıkarımı şudur:

> Proje yalnızca sentetik bir RL prototipi değildir; trace-driven training, hold-out evaluation ve domain-shift analizi çalışır hale getirilmiştir.

### 6.7 Faz 7: Two-Stage Training

Yedinci fazda AgentVNE benzeri staged training yaklaşımı uygulanmıştır. Buradaki amaç, PPO ajanını doğrudan sıfırdan eğitmek yerine önce teacher-labeled oracle dataset ile ısıtmak, ardından PPO fine-tuning ile performansı artırmaktır.

Bu yaklaşım iki aşamadan oluşur:

1. Supervised pretraining: Ajan teacher label'ları taklit etmeyi öğrenir.
2. PPO fine-tuning: Ajan environment reward'una göre politikasını iyileştirir.

Faz 7'de önce oracle label dataset üretilmiştir:

- `results/raw/synthetic/pretraining/oracle_label_dataset.csv`

Coverage-aware selection ve train rebalance ile bütün aksiyonların dataset içinde temsil edilmesi sağlanmıştır:

- local
- edge_25
- edge_50
- edge_75
- edge_100
- cloud

Dört farklı teacher policy test edilmiştir:

- teacher_latency_greedy
- teacher_energy_greedy
- teacher_balanced_semantic
- teacher_contextual_reward_aligned

Final karşılaştırma sonuçları:

| Teacher Policy | Pretrain Test Acc | Scratch Success | Pretrained Success | Delta |
|---|---:|---:|---:|---:|
| latency_greedy | %77.56 | %63.00 | %77.33 | +14.33 |
| contextual_reward_aligned | %83.11 | %63.00 | %75.20 | +12.20 |
| balanced_semantic | %83.56 | %63.00 | %73.27 | +10.27 |
| energy_greedy | %79.33 | %63.00 | %72.13 | +9.13 |

En yüksek başarı artışı latency_greedy teacher ile elde edilmiştir. Ancak bu teacher final policy'yi Full Cloud ağırlıklı davranışa itmiştir. Bu nedenle yalnızca başarı oranına göre seçim yapılmamıştır.

Kanonik teacher olarak `teacher_contextual_reward_aligned` seçilmiştir. Bunun nedeni, başarı artışını korurken final policy'nin Full Cloud'a çökmesini engellemesi ve Edge %75 ağırlıklı daha savunulabilir bir decision structure sağlamasıdır.

Kanonik sonuç:

- Scratch PPO: %63.00 ± 2.46
- Pretrained + PPO: %75.20 ± 2.50
- Delta success: +12.20 puan
- P95 latency delta: -0.696 s
- QoE delta: +15.68
- Energy per success delta: -0.1197

Bu fazın ana çıkarımı şudur:

> Two-stage training, task offloading probleminde gerçek bir warm-start avantajı sağlamıştır. Ancak teacher seçimi yalnızca başarıyı değil, final policy'nin davranış yönünü de ciddi biçimde etkilemektedir.

### 6.8 Faz 8: Graph-Aware Policy Upgrade

Sekizinci fazda sistemin state temsili daha ileri bir seviyeye taşınmaya başlanmıştır. Faz 1-7 boyunca ajan kararlarını düz bir vektör state üzerinden veriyordu. Bu state faydalı olmakla birlikte, offloading probleminin doğal ilişki yapısını açıkça temsil etmiyordu.

Gerçekte task offloading problemi graph yapısına uygundur:

- Device bir node olabilir.
- Task bir node olabilir.
- Edge server'lar node olabilir.
- Cloud bir node olabilir.
- Aralarındaki bağlantılar edge olarak temsil edilebilir.

Bu nedenle Faz 8'de graph-aware policy yaklaşımına geçilmiştir.

Faz 8'in ana sorusu şudur:

> Device-task-edge-cloud ilişkilerini graph olarak temsil etmek, düz vektör state ile çalışan MLP tabanlı policy'lere göre daha bağlama duyarlı kararlar üretebilir mi?

Faz 8.1 kapsamında `GraphState` yapısı ve `graph_state_builder` geliştirilmiştir. Bu yapı şu alanları üretmektedir:

- node_features
- edge_index
- edge_features
- global_features
- action_prior
- action_mask
- node_type_ids
- edge_type_ids
- metadata
- vector_state_reference

İlk graph topology şu şekilde tasarlanmıştır:

- 1 device node
- 1 task node
- N edge server node
- 1 cloud node

Edge aileleri:

- device-edge
- task-edge
- device-cloud
- task-device
- task-cloud

Faz 8.2 kapsamında PyTorch-only ilk graph-aware policy forward path yazılmıştır. PyTorch Geometric kurulumuna bağımlı olmadan çalışabilmesi için ilk model PyTorch-only tasarlanmıştır.

`GraphPolicyNetwork` şu çıktıları üretmektedir:

- logits
- masked_logits
- action_probabilities
- graph_embedding

Bu aşamada modelin amacı henüz nihai performans üretmek değil, graph state'i okuyup 6 aksiyon için skor üretebildiğini doğrulamaktır.

Çalıştırılan testler:

- `python -m unittest tests.test_graph_state_builder`
- `python -m unittest tests.test_graph_state_builder tests.test_graph_policy`

Test sonuçları başarıyla tamamlanmıştır. Bu, graph state üretimi ve graph policy forward path'in teknik olarak çalıştığını göstermektedir.

Faz 8 sonunda henüz tamamlanacak işler:

- Semantic prior fusion stratejilerini ayırmak
- none / late / early fusion varyantlarını karşılaştırmak
- Graph-aware policy'yi MLP-PPO ve Pretrained MLP-PPO ile karşılaştırmak
- Action diversity ve context-sensitive davranışı ölçmek

## 7. Bulgular

Bu çalışmada elde edilen bulgular birkaç ana başlıkta toplanabilir.

İlk bulgu, PPO'nun genel olarak en güçlü adaylardan biri olmasıdır. Faz 5 full model retraining sonuçlarında PPO %76.17 başarı oranı ile DQN ve A2C'nin biraz üzerinde yer almıştır. Bu fark çok büyük olmamakla birlikte PPO'nun ana model olarak seçilmesini desteklemektedir.

İkinci bulgu, partial offloading ve mobility features bileşenlerinin kritik olmasıdır. Ablation çalışmaları, bu bileşenler çıkarıldığında performansta anlamlı düşüşler oluştuğunu göstermiştir. Özellikle mobility features, üç algoritmada da kararlı şekilde önemli görünmüştür.

Üçüncü bulgu, sentetik ortam ile trace ortam arasında yönlü bir genelleme farkı olmasıdır. Synthetic -> trace yönü güçlü çıkarken, trace -> synthetic yönü zayıf kalmıştır. Bu durum, veri dağılımlarının birbirine simetrik şekilde benzemediğini göstermektedir.

Dördüncü bulgu, two-stage training yaklaşımının PPO için gerçek bir avantaj sağlamasıdır. Kanonik teacher ile pretrained + PPO, scratch PPO'ya göre +12.20 puan başarı artışı sağlamıştır.

Beşinci bulgu, başarı oranının tek başına yeterli olmadığıdır. Bazı teacher policy'ler daha yüksek başarı verse bile final policy'yi Full Cloud gibi istenmeyen dominant action davranışlarına itebilmektedir. Bu yüzden action diversity, dominant action ve decision structure gibi davranışsal metriklerin de izlenmesi gerekmektedir.

Altıncı bulgu, graph-aware temsilin proje için doğal bir sonraki adım olduğudur. Çünkü offloading problemi yalnızca düz sayısal özelliklerden değil, cihaz-task-edge-cloud ilişkilerinden oluşmaktadır.

## 8. Değerlendirme ve Tartışma

Bu çalışmanın en güçlü tarafı, tek bir model sonucuna dayanmaması ve aşamalı bir araştırma metodolojisi izlemesidir. Proje önce tekrarlanabilir hale getirilmiş, sonra simülasyon ortamı iyileştirilmiş, semantic prior eklenmiş, baseline'lar genişletilmiş, ablation yapılmış, trace-driven doğrulama gerçekleştirilmiş ve staged training ile model davranışı güçlendirilmiştir.

Bu açıdan çalışma, yalnızca "bir RL modeli eğittim" düzeyinde değildir. Her bileşenin etkisi ayrı ayrı sorgulanmış ve deneysel olarak değerlendirilmiştir.

Semantic prior bileşeni teorik olarak güçlü bir katkıdır. Çünkü görevlerin anlamını karar sürecine dahil eder. Ancak ablation sonuçları semantic bileşenin etkisinin her koşulda çok net ayrışmadığını göstermiştir. Bu durum semantic prior'ın gereksiz olduğu anlamına gelmez; mevcut state, reward veya environment tasarımının semantic katkıyı her senaryoda yeterince görünür hale getirmediği anlamına gelebilir.

Reward shaping için de benzer bir yorum yapılabilir. Bazı sonuçlarda reward shaping etkisi sınırlı görünmüştür. Bu, reward tasarımının daha hassas kalibre edilmesi gerektiğini göstermektedir. Faz 9'da reward decomposition, p99 latency, deadline miss ratio, energy per success ve confidence interval gibi gelişmiş metrikler eklendiğinde bu etkiler daha iyi analiz edilebilir.

Trace-driven sonuçların çok yüksek çıkması olumlu bir bulgudur; ancak bu sonuçlar dikkatli yorumlanmalıdır. Trace ortamında train, validation ve test splitleri arasında belirgin overfitting görünmemektedir. Fakat synthetic -> trace ve trace -> synthetic sonuçlarının asimetrik olması, ortamlar arasında farklı karar kalıpları olduğunu göstermektedir. Bu nedenle gerçek genelleme iddiası daha geniş trace datasetleriyle güçlendirilebilir.

Two-stage training sonuçları oldukça değerlidir. Çünkü pretrained + PPO'nun scratch PPO'ya göre belirgin avantaj sağladığı görülmüştür. Ancak burada önemli nokta teacher seçimidir. En yüksek başarıyı veren teacher her zaman en iyi teacher değildir. Eğer teacher final policy'yi istenmeyen bir action attractor'a itiyorsa, bu sonuç bilimsel olarak daha zayıf olabilir. Bu nedenle bu çalışmada contextual_reward_aligned teacher daha dengeli bir seçim olarak tercih edilmiştir.

Graph-aware policy çalışması henüz tamamlanmamış olsa da doğru bir araştırma yönüdür. Çünkü mevcut sistemde ajan çoğu durumda Edge %75 etrafında toplanma eğilimi göstermektedir. Graph-aware temsil, sistemin ilişkisel yapısını daha açık göstereceği için action diversity ve context-sensitive decision making açısından fayda sağlayabilir.

## 9. Sınırlılıklar

Bu çalışmanın bazı sınırlılıkları bulunmaktadır.

İlk olarak, sistem henüz gerçek fiziksel edge testbed üzerinde çalıştırılmamıştır. Deneyler sentetik ortam ve trace-driven episode splitleri üzerinden yapılmıştır. Bu, akademik araştırma için geçerli bir başlangıç olsa da gerçek sistem doğrulaması ileride yapılmalıdır.

İkinci olarak, trace verilerinin task formatına dönüştürülmesi bazı varsayımlar gerektirir. Trace-to-task mapping varsayımları dokümante edilmiştir; ancak bu varsayımlar sonuçları etkileyebilir.

Üçüncü olarak, semantic analyzer tarafında kullanılan LLM/fallback yapısı sınırlı olabilir. Daha güçlü LLM'ler veya domain-specific semantic analyzer'lar ile semantic prior kalitesi artırılabilir.

Dördüncü olarak, reward shaping etkisi henüz tüm yönleriyle ayrışmamıştır. Bu nedenle daha ayrıntılı reward decomposition ve istatistiksel analiz gereklidir.

Beşinci olarak, graph-aware policy henüz ilk forward path ve test aşamasındadır. Nihai performans karşılaştırması için semantic prior fusion ve eğitim/evaluation deneylerinin tamamlanması gerekmektedir.

Altıncı olarak, action diversity problemi tamamen çözülmüş değildir. Cloud collapse kırılmış olsa da bazı fazlarda Edge %75 dominant action olarak öne çıkmaktadır.

## 10. Sonuç ve Gelecek Çalışmalar

Bu seminer çalışmasında, IoT ve edge computing ortamlarında semantic-aware task offloading problemi ele alınmıştır. Çalışma kapsamında LLM destekli semantic prior, DRL tabanlı karar verme, baseline karşılaştırmaları, ablation study, trace-driven training, two-stage training ve graph-aware policy geliştirmeleri yapılmıştır.

Çalışmanın ana katkıları şu şekilde özetlenebilir:

1. Task offloading problemi için tekrarlanabilir bir deney altyapısı kurulmuştur.
2. Fiziksel state ile semantic prior bilgisini birleştiren semantic-aware RL yaklaşımı geliştirilmiştir.
3. PPO, DQN ve A2C modelleri klasik heuristic baseline'larla karşılaştırılmıştır.
4. Partial offloading, mobility features, reward shaping ve semantic prior gibi bileşenlerin etkisi ablation study ile incelenmiştir.
5. Sentetik ortamdan trace-driven ortama geçilerek domain shift ve hold-out test analizi yapılmıştır.
6. Two-stage training ile PPO performansı scratch training'e göre anlamlı şekilde artırılmıştır.
7. Graph-aware policy için graph state builder ve ilk graph policy forward path geliştirilmiştir.

Gelecek çalışmalarda öncelikli hedefler şunlardır:

- Faz 8.3 kapsamında semantic prior fusion stratejilerinin tamamlanması
- Graph-aware policy'nin MLP-PPO ve Pretrained MLP-PPO ile karşılaştırılması
- Faz 9 kapsamında gelişmiş metriklerin eklenmesi
- p99 latency, deadline miss ratio, energy per success, jitter, fairness ve confidence interval hesaplarının raporlanması
- Daha geniş trace datasetleriyle genelleme analizinin güçlendirilmesi
- LLM self-reflection ve experience replay mekanizmasının eklenmesi
- Journal makalesi için sonuç tablolarının ve istatistiksel testlerin son haline getirilmesi

Genel olarak bu çalışma, klasik RL tabanlı task offloading yaklaşımını daha ileri taşıyarak LLM-guided, semantic-aware, two-stage trained ve graph-aware bir offloading framework'üne dönüştürmeyi hedeflemektedir.

## Kaynakça İçin Hazırlık Notları

Raporun journal formatına dönüştürülmesi aşamasında kaynakça bölümü şu çalışma gruplarından oluşturulmalıdır:

- Mobile Edge Computing ve task offloading üzerine temel çalışmalar
- DRL tabanlı edge offloading çalışmaları
- PPO, DQN ve A2C temel algoritma kaynakları
- Semantic-aware veya LLM-guided decision making çalışmaları
- AgentVNE ve staged training yaklaşımıyla ilişkili çalışmalar
- Graph Neural Network tabanlı resource allocation ve scheduling çalışmaları

## Ekler

### Ek A: Proje Fazları

| Faz | Başlık | Durum |
|---|---|---|
| Faz 1 | Reproducibility ve kod temizliği | Tamamlandı |
| Faz 2 | Training environment ve simülasyon hizalaması | Tamamlandı |
| Faz 3 | LLM/Semantic prior entegrasyonu | Tamamlandı |
| Faz 4 | Baseline ailesinin genişletilmesi | Tamamlandı |
| Faz 5 | Sistematik ablation study | Tamamlandı |
| Faz 6 | Trace-driven training | Tamamlandı |
| Faz 7 | Two-stage training | Tamamlandı |
| Faz 8 | Graph-aware policy upgrade | Devam ediyor |
| Faz 9 | Advanced metrics, statistical analysis ve GUI | Planlandı |
| Faz 10 | LLM self-reflection ve experience replay | Planlandı |

### Ek B: Kullanılan Ana Dosya ve Artefaktlar

| Tür | Dosya/Klasör |
|---|---|
| Ana görev listesi | `task.md` |
| Faz raporları | `phase_reports/` |
| Sentetik configler | `configs/synthetic/` |
| Trace configler | `configs/trace/` |
| Sentetik deneyler | `experiments/synthetic/` |
| Trace deneyleri | `experiments/trace/` |
| Model checkpointleri | `models/` |
| Ham sonuçlar | `results/raw/` |
| Grafikler | `results/figures/` |
| Graph state builder | `src/env/graph_state_builder.py` |
| Graph policy | `src/agents/graph_policy.py` |
| Graph testleri | `tests/test_graph_state_builder.py`, `tests/test_graph_policy.py` |

### Ek C: Seminerde Anlatırken Kullanılabilecek Kısa Akış

Sunumda çalışmayı şu sırayla anlatmak daha kolay olacaktır:

1. Problem: IoT cihazları görevleri nerede çalıştırmalı?
2. Zorluk: Gecikme, enerji, deadline, batarya ve ağ koşulları birlikte değişiyor.
3. Çözüm fikri: RL ajanı karar versin, LLM semantic prior ile yönlendirsin.
4. İlk altyapı: Reproducibility, state builder, reward, logging.
5. Baseline'lar: LocalOnly, EdgeOnly, CloudOnly, Random, GreedyLatency, GA.
6. RL modelleri: PPO, DQN, A2C.
7. Ablation: Hangi bileşen gerçekten önemli?
8. Trace-driven test: Sentetik sonuçlar gerçekçi veri splitlerine taşınıyor mu?
9. Two-stage training: Önce teacher'dan öğren, sonra PPO ile iyileştir.
10. Graph-aware policy: Sistem aslında graph; bunu modele açıkça gösterelim.
11. Sonuç: Semantic-aware DRL yaklaşımı umut verici, graph-aware yön gelecek katkıyı güçlendirecek.
