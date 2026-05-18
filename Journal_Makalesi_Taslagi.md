# Journal Makalesi Taslağı

## Semantic-Aware Task Offloading and Dynamic Resource Allocation in Next-Generation Edge Networks via LLM-Guided Deep Reinforcement Learning

## 1. Başlık

**Semantic-Aware Task Offloading and Dynamic Resource Allocation in Next-Generation Edge Networks via LLM-Guided Deep Reinforcement Learning**

Türkçe karşılığı:

**LLM Destekli Derin Pekiştirmeli Öğrenme ile Yeni Nesil Edge Ağlarda Semantik Farkındalıklı Görev Aktarımı ve Dinamik Kaynak Tahsisi**

Bu başlık çalışmanın üç ana katkısını doğrudan yansıtmaktadır:

- Task offloading ve dynamic resource allocation problemini ele alması
- LLM/semantic prior ile görev anlamını karar sürecine dahil etmesi
- Deep reinforcement learning ve graph-aware policy yönüne ilerlemesi

## 2. Özet

Mobile edge computing ve IoT sistemlerinde görev aktarımı, gecikme, enerji tüketimi, cihaz bataryası, ağ kalitesi, edge sunucu yükü ve deadline gibi birden fazla değişkenin birlikte optimize edilmesini gerektiren karmaşık bir problemdir. Klasik heuristic yöntemler bu dinamik yapıya her zaman uyum sağlayamadığı için, öğrenebilen ve bağlama duyarlı karar mekanizmalarına ihtiyaç duyulmaktadır.

Bu çalışmada, IoT edge ağlarında semantic-aware task offloading ve dynamic resource allocation için LLM-guided deep reinforcement learning tabanlı bir framework önerilmektedir. Önerilen sistemde görev ve sistem durumu fiziksel özelliklerle temsil edilmekte, ayrıca semantic analyzer tarafından üretilen 6 boyutlu action prior karar sürecine dahil edilmektedir. PPO, DQN ve A2C modelleri LocalOnly, EdgeOnly, CloudOnly, Random, GreedyLatency ve GeneticAlgorithm baseline'ları ile karşılaştırılmıştır. Çalışmada sentetik deneyler, sistematik ablation study, trace-driven training, domain-shift evaluation, two-stage training ve graph-aware policy altyapısı birlikte ele alınmıştır.

Deneysel sonuçlar, PPO tabanlı full modelin sentetik ortamda %76.17 başarı oranına ulaştığını, trace-driven hold-out testte PPO modelinin %99.60 başarı gösterdiğini ve two-stage training yaklaşımının scratch PPO'ya göre +12.20 puanlık başarı artışı sağladığını göstermektedir. Ablation sonuçları partial offloading ve mobility features bileşenlerinin karar kalitesi açısından kritik olduğunu ortaya koymaktadır. Ayrıca graph-aware policy çalışması, offloading probleminin cihaz-task-edge-cloud ilişkilerini açıkça temsil eden daha yapısal bir modele taşınabileceğini göstermektedir.

## 3. Anahtar Kelimeler

Task Offloading, Mobile Edge Computing, Internet of Things, Deep Reinforcement Learning, PPO, DQN, A2C, Semantic-Aware Offloading, Large Language Models, Dynamic Resource Allocation, Partial Offloading, Graph-Aware Policy, Graph Neural Networks.

## 4. Giriş

IoT cihazları günümüzde akıllı şehirler, sağlık sistemleri, endüstriyel otomasyon, mobil uygulamalar ve gerçek zamanlı izleme sistemlerinde yaygın olarak kullanılmaktadır. Bu cihazların işlem gücü, batarya kapasitesi ve ağ bağlantısı çoğu zaman sınırlıdır. Buna rağmen bu cihazlardan gelen görevler giderek daha karmaşık hale gelmektedir.

Bu noktada mobile edge computing, görevlerin yalnızca cihaz üzerinde değil, cihaza yakın edge sunucular veya cloud altyapısı üzerinde çalıştırılmasına imkan tanır. Ancak hangi görevin nerede çalıştırılacağı kararı basit değildir. Görev local çalıştırılırsa iletişim maliyeti oluşmaz, fakat cihaz bataryası ve işlem kapasitesi zorlanabilir. Edge sunucuya gönderilirse gecikme azalabilir, fakat edge yükü ve kuyruk bekleme süresi artabilir. Cloud ise güçlü işlem kapasitesi sunsa da uzaklık ve aktarım gecikmesi nedeniyle her durumda uygun olmayabilir.

Bu çalışmada task offloading problemi, local, partial edge offloading ve cloud offloading seçeneklerinden oluşan ayrık bir karar problemi olarak ele alınmıştır. Temel amaç, deadline içinde görev tamamlama oranını artırırken gecikme, enerji tüketimi ve QoE gibi metrikleri dengeli şekilde optimize etmektir.

Çalışmanın temel motivasyonu, mevcut offloading yaklaşımlarının çoğunlukla fiziksel sistem metriklerine odaklanmasıdır. Oysa görevlerin semantik özellikleri de karar sürecinde önemlidir. Örneğin acil ve kritik bir görev ile düşük öncelikli bir görev aynı offloading politikasıyla ele alınmamalıdır. Bu nedenle önerilen yöntemde semantic analyzer tarafından üretilen action prior, RL ajanının state temsiline dahil edilmiştir.

Bu çalışmanın ana araştırma sorusu şudur:

> LLM tabanlı semantik görev analizi, derin pekiştirmeli öğrenme tabanlı task offloading kararlarının başarı oranını ve karar kalitesini artırabilir mi?

Çalışmanın katkıları şu şekilde özetlenebilir:

- Semantic prior ile zenginleştirilmiş DRL tabanlı task offloading framework'ü geliştirilmiştir.
- PPO, DQN ve A2C modelleri klasik heuristic ve meta-heuristic baseline'larla karşılaştırılmıştır.
- Partial offloading, mobility features, semantic prior, confidence ve reward shaping bileşenleri ablation study ile analiz edilmiştir.
- Sentetik ortamdan trace-driven ortama geçilerek hold-out test ve domain-shift evaluation yapılmıştır.
- Teacher-labeled supervised pretraining ve PPO fine-tuning ile two-stage training yaklaşımı uygulanmıştır.
- Offloading karar yapısını graph temsiline taşımak için graph state builder ve graph-aware policy forward path geliştirilmiştir.

## 5. Literatür Taraması

Task offloading literatüründe üç ana yaklaşım ailesi öne çıkmaktadır: heuristic yöntemler, optimizasyon/meta-sezgisel yöntemler ve deep reinforcement learning tabanlı yöntemler.

Heuristic yöntemler basit ve hızlıdır. LocalOnly, EdgeOnly, CloudOnly veya GreedyLatency gibi yöntemler bu gruba örnek verilebilir. Bu yöntemler yorumlanabilir olmalarına rağmen değişken ağ koşulları, batarya seviyesi, edge yükü ve deadline gibi dinamik faktörlere yeterince esnek tepki veremeyebilir.

Optimizasyon ve meta-sezgisel yöntemler, karar uzayında daha iyi çözümler aramak için kullanılmaktadır. Genetic Algorithm gibi yöntemler heuristic yaklaşımlardan daha esnek olabilir. Ancak gerçek zamanlı sistemlerde hesaplama maliyeti ve çevrimiçi adaptasyon kabiliyeti önemli sınırlılıklar doğurabilir.

DRL tabanlı yöntemler, ajanın ortamla etkileşerek politika öğrenmesine dayanır. Task offloading problemi ayrık aksiyon uzayına sahip olduğu için DQN doğal bir baseline'dır. PPO ve A2C gibi actor-critic yöntemler ise çok amaçlı ve stokastik ortamlarda daha dengeli politika öğrenimi sunabilir.

Bu çalışmada PPO ana model olarak seçilmiştir. PPO'nun tercih edilmesinin nedeni, politika güncellemelerini clipping mekanizması ile sınırlayarak daha stabil öğrenme sağlamasıdır. DQN value-based RL yaklaşımını, A2C ise daha temel actor-critic yaklaşımını temsil etmektedir. SAC ve DDPG gibi continuous-action modeller bu çalışmanın ana kapsamına alınmamıştır; çünkü bu projede aksiyon uzayı discrete olarak tanımlanmıştır.

Literatürdeki önemli boşluklardan biri, birçok çalışmanın task offloading kararını yalnızca fiziksel metriklerle ele almasıdır. Görevin semantik niteliği, deadline kritikliğinin anlamı, öncelik seviyesi veya görevin bağlamsal önemi çoğu zaman doğrudan karar mekanizmasına dahil edilmemektedir. Bu çalışma, LLM/semantic analyzer tarafından üretilen semantic prior'ı RL state'ine ekleyerek bu boşluğa cevap vermeyi hedeflemektedir.

## 6. Problem Tanımı

Bu çalışmada bir IoT cihazı, edge sunucular ve cloud altyapısından oluşan bir sistem ele alınmaktadır. Her karar adımında sistemde bir görev oluşmakta ve ajan bu görevin hangi ortamda çalıştırılacağına karar vermektedir.

Görev özellikleri şunları içermektedir:

- Task size
- CPU cycle ihtiyacı
- Deadline
- Öncelik/aciliyet
- İletişim gereksinimi
- Semantik karmaşıklık

Sistem özellikleri şunları içermektedir:

- SNR veya link kalitesi
- Cihaz batarya seviyesi
- Edge server load
- Edge enerji durumu
- Queue waiting etkisi
- Switching overhead
- Cloud/edge aktarım gecikmesi

Aksiyon uzayı 6 ayrık karardan oluşmaktadır:

| Aksiyon | Açıklama |
|---|---|
| 0 | Local processing |
| 1 | Edge %25 partial offloading |
| 2 | Edge %50 partial offloading |
| 3 | Edge %75 partial offloading |
| 4 | Edge %100 offloading |
| 5 | Cloud offloading |

Amaç, görevin deadline içinde başarıyla tamamlanmasını sağlarken latency, energy consumption, queue waiting ve QoE metriklerini dengeli şekilde optimize etmektir.

Reward fonksiyonu genel olarak şu bileşenlerden etkilenmektedir:

- Başarı bonusu
- Gecikme cezası
- Enerji cezası
- Deadline miss cezası
- Semantic alignment etkisi
- Confidence etkisi
- Partial offloading maliyeti
- Queue/load etkisi

### 6.1 Matematiksel Sistem Modeli

Önerilen ortamda gecikme ve enerji değerleri sabit katsayılarla değil, haberleşme, işlem, kuyruk ve partial offloading modellerinden türetilmektedir. Bu nedenle modelin fiziksel temeli aşağıdaki bileşenlerden oluşur.

#### 6.1.1 Kablosuz Kanal ve Shannon Kapasitesi

Cihaz ile edge sunucu arasındaki mesafe Öklid uzaklığı ile hesaplanır:

```text
d = sqrt((x_i - x_j)^2 + (y_i - y_j)^2)
```

Path loss modeli ile kanal kazancı:

```text
h = d^(-alpha)
```

Burada güncel implementasyonda `alpha = 4` olarak kullanılmaktadır. SINR:

```text
SINR = (P_tx * h) / (N0 + I)
```

Shannon-Hartley kapasitesi:

```text
R = B * log2(1 + SINR)
```

Kullanılan temel parametreler:

```text
P_tx = 0.5 W
N0 = 1e-13 W
B = 20 MHz
```

Aktarım süresi:

```text
T_tx = S / R
```

Burada `S` task size bits değeridir. Bu model sayesinde cihaz-edge mesafesi, path loss, noise ve interference etkisi transmission delay ve transmission energy üzerinde doğrudan etkili olur.

Not: Kodda bazı değişkenler tarihsel olarak `snr_norm` adıyla tutulmuştur. Güncel state temsili açısından bu değer doğrudan ham SNR değil, Shannon modelinden elde edilen veri hızının normalize edilmiş göstergesidir:

```text
datarate_norm = min(1, R / 50e6)
```

Bu nedenle makale metninde bu değişkenin “normalized datarate/link-quality indicator” olarak tanımlanması daha doğrudur.

#### 6.1.2 Transmission Energy Model

Offloading sırasında cihazın harcadığı iletim enerjisi:

```text
E_tx = P_tx * T_tx
```

Partial offloading için:

```text
E_tx_partial = P_tx * ((r * S) / R)
```

Burada `r`, edge'e aktarılan görev oranıdır:

```text
r ∈ {0.25, 0.50, 0.75, 1.00}
```

Bu formül, ağ kalitesi düştüğünde veri hızının azalacağını, bunun da aktarım süresini ve enerji tüketimini artıracağını ifade eder.

#### 6.1.3 Computation and DVFS Energy Model

İşlem gecikmesi:

```text
T_comp = C / f
```

Burada:

- `C`: CPU cycles
- `f`: CPU frequency

Tipik frekanslar:

```text
f_local = 1 GHz
f_edge ≈ 2 GHz
f_cloud = 5 GHz
```

CPU enerji modeli DVFS mantığına dayanır:

```text
E_comp = kappa * f^3 * T_comp
```

`T_comp = C / f` olduğu için:

```text
E_comp = kappa * f^2 * C
```

Güncel implementasyonda:

```text
kappa = 1e-28
```

Edge sunucularda frekans yük durumuna göre değişir:

```text
if current_load > 2:
    f_edge = f_max
else:
    f_edge = 0.7 * f_max
```

Bu yapı, düşük yükte enerji tasarrufu, yüksek yükte ise daha hızlı işlem davranışını temsil eder.

#### 6.1.4 Local, Edge, Cloud ve Partial Offloading Gecikmeleri

Local işlem:

```text
T_local = C / f_local
E_local = kappa * f_local^2 * C
```

Cloud işlem:

```text
T_cloud = T_tx + T_cloud_fixed + C / f_cloud + T_cloud_congestion
E_cloud_device = E_tx
```

Edge/partial offloading için:

```text
local_cycles = (1 - r) * C
edge_cycles = r * C
edge_bits = r * S
```

Local parça:

```text
T_local_part = ((1 - r) * C) / f_local
```

Edge parça:

```text
T_edge_part = (r * S) / R + (r * C) / f_edge + T_queue
```

Partial offloading toplam gecikmesi:

```text
T_partial = max(T_local_part, T_edge_part) + T_overhead
```

Partial offloading cihaz enerjisi:

```text
E_partial = kappa * f_local^2 * ((1 - r) * C)
            + P_tx * ((r * S) / R)
```

Bu model, görevin local ve edge parçalarının paralel yürütülebileceğini ve toplam gecikmenin yavaş kalan kola göre belirleneceğini varsayar.

#### 6.1.5 Queue, Congestion ve Switching Overhead

Edge queue etkisi yaklaşık olarak:

```text
T_edge_queue = 0.015 * queue_length + 0.02 * current_load
```

Cloud congestion etkisi:

```text
T_cloud_congestion = 0.02 * cloud_queue + 0.03 * cloud_load
```

Partial offloading için koordinasyon/switching overhead:

```text
T_overhead = coordination_factor *
             (0.01 + 0.02 * size_factor + mobility_penalty + transition_penalty)
```

Bu bileşenler, offloading kararını daha gerçekçi hale getirir. Çünkü pratikte görev bölme, edge kuyruk bekleme ve aksiyon değiştirme maliyetsiz değildir.

## 7. Önerilen Yöntem

Önerilen yöntem, fiziksel sistem durumunu semantic prior ile birleştiren DRL tabanlı bir offloading ajanına dayanmaktadır.

Genel karar akışı şu şekildedir:

1. Ortamda yeni bir görev oluşur.
2. Görev ve sistem durumu state builder tarafından normalize edilmiş state'e dönüştürülür.
3. Semantic analyzer görev özelliklerinden action prior üretir.
4. Fiziksel state ve semantic prior birlikte RL ajanına verilir.
5. Ajan local, partial edge veya cloud aksiyonlarından birini seçer.
6. Ortam aksiyonu uygular ve latency, energy, success bilgilerini üretir.
7. Reward hesaplanır.
8. Ajan policy'sini günceller.

İlk state temsili fiziksel özellikler ve semantic prior olmak üzere iki parçadan oluşmuştur. Fiziksel özellikler SNR, task size, CPU cycles, battery percent ve edge load gibi değerleri içerirken semantic prior 6 aksiyon için olasılık dağılımı sunmaktadır.

Semantic prior, tek bir kesin tavsiye yerine aksiyonlar üzerinde dağılım olarak kullanılmıştır. Böylece LLM/semantic analyzer ajana rehberlik eder, ancak karar tamamen LLM tarafından belirlenmez. Confidence skoru da bu dağılımın keskinliğini ayarlamak için kullanılmıştır.

RL modeli tarafında PPO ana modeldir. DQN ve A2C ise karşılaştırma amacıyla dahil edilmiştir. PPO'nun ana model olarak seçilmesi, stabil politika güncellemesi ve çok amaçlı RL problemlerindeki dayanıklılığı ile ilişkilidir.

## 8. Sistem Mimarisi

Sistem mimarisi birkaç ana modülden oluşmaktadır:

- Environment: Görevleri, cihazı, edge/cloud kaynaklarını ve reward dinamiğini temsil eder.
- State Builder: Ortam durumunu ajan için sayısal state'e dönüştürür.
- Semantic Analyzer: Görevin anlamını analiz ederek semantic prior üretir.
- Reward Module: Latency, energy, deadline ve semantic alignment bileşenlerinden reward hesaplar.
- RL Agents: PPO, DQN ve A2C politikalarını içerir.
- Baseline Policies: LocalOnly, EdgeOnly, CloudOnly, Random, GreedyLatency ve GeneticAlgorithm yöntemlerini içerir.
- Evaluation Pipeline: Modelleri ortak metriklerle değerlendirir.
- Logging and Reporting: CSV, figure ve phase report üretir.
- Graph State Builder: Offloading karar durumunu node-edge yapısı olarak temsil eder.
- Graph Policy Network: GraphState girdisinden 6 aksiyon için logits ve action probabilities üretir.

Proje aşamalarına göre mimari olgunlaşma şu şekilde ilerlemiştir:

- Faz 1: Reproducibility, config ve logging altyapısı
- Faz 2: Ortak state builder ve simulator-backed environment
- Faz 3: Semantic prior ve LLM output parsing
- Faz 4: Baseline ve RL model ailesi
- Faz 5: Ablation ve multi-seed retraining
- Faz 6: Trace-driven pipeline ve domain shift
- Faz 7: Oracle label, supervised pretraining ve PPO fine-tuning
- Faz 8: Graph state builder ve graph-aware policy

## 9. Deneysel Kurulum

Deneyler iki ana ortamda yürütülmüştür:

- Synthetic environment
- Trace-driven environment

Synthetic environment, kontrollü deney ve ablation çalışmaları için kullanılmıştır. Trace-driven environment ise daha gerçekçi episode splitleri üzerinden modelin genelleme kabiliyetini değerlendirmek için kullanılmıştır.

Kullanılan modeller:

- PPO
- DQN
- A2C

Kullanılan baseline'lar:

- LocalOnly
- EdgeOnly
- CloudOnly
- Random
- GreedyLatency
- GeneticAlgorithm

Kullanılan ana metrikler:

- Success rate
- Average reward
- P95 latency
- Average energy
- QoE
- Dominant action
- Deadline miss ratio
- Energy per success
- Domain-shift success

Reproducibility için seed kontrolü uygulanmıştır. Deney sonuçları `results/raw/` altında CSV olarak, grafikler `results/figures/` altında, faz raporları ise `phase_reports/` altında tutulmuştur.

## 10. Sonuçlar

### 10.1 Sentetik Full Model Sonuçları

Faz 5 full model retraining sonuçları:

| Algorithm | Success Rate | P95 Latency | Avg Energy | Dominant Action |
|---|---:|---:|---:|---:|
| PPO | %76.17 ± 10.63 | 2.799 | 0.0768 | 3 |
| DQN | %74.73 ± 11.23 | 2.802 | 0.0791 | 3 |
| A2C | %74.73 ± 11.23 | 2.802 | 0.0791 | 3 |

Bu sonuçlar PPO'nun ana aday olarak kalmasını desteklemektedir. DQN ve A2C yakın sonuçlar üretmiş olsa da PPO en yüksek ortalama başarıyı vermiştir.

### 10.2 Trace-Driven Sonuçlar

Trace training sonucunda PPO modeli validation tarafında %99.20 başarı oranına ulaşmıştır. Hold-out test sonuçları ise şu şekildedir:

| Split | Mean Success | Mean Delay | Mean Energy |
|---|---:|---:|---:|
| Train | %99.38 | 0.3245 s | 0.0128 |
| Validation | %99.00 | 0.3453 s | 0.0207 |
| Test | %99.60 | 0.3317 s | 0.0128 |

Train, validation ve test sonuçlarının yakın olması, trace tarafında belirgin bir overfitting işareti olmadığını göstermektedir.

### 10.3 Domain-Shift Sonuçları

| Train Domain | Test Domain | Success Rate | P95 Latency | Avg Energy | Dominant Action |
|---|---|---:|---:|---:|---:|
| Synthetic | Trace | %99.60 | 0.5618 | 0.0255 | 3 |
| Trace | Synthetic | %53.20 | 4.6883 | 0.0101 | 4 |

Bu sonuç, domain shift'in simetrik olmadığını göstermektedir. Synthetic -> trace yönünde model güçlü kalırken, trace -> synthetic yönünde performans belirgin şekilde düşmektedir.

### 10.4 Two-Stage Training Sonuçları

Teacher sensitivity sonuçları:

| Teacher Policy | Pretrain Test Acc | Scratch Success | Pretrained Success | Delta |
|---|---:|---:|---:|---:|
| teacher_latency_greedy | %77.56 | %63.00 | %77.33 | +14.33 |
| teacher_contextual_reward_aligned | %83.11 | %63.00 | %75.20 | +12.20 |
| teacher_balanced_semantic | %83.56 | %63.00 | %73.27 | +10.27 |
| teacher_energy_greedy | %79.33 | %63.00 | %72.13 | +9.13 |

Kanonik teacher olarak `teacher_contextual_reward_aligned` seçilmiştir. Bu teacher, yüksek başarı artışı sağlarken final policy'nin Full Cloud collapse davranışına kaymasını azaltmıştır.

## 11. Ablation Study

Ablation çalışmasında full modelden belirli bileşenler çıkarılarak performans etkileri ölçülmüştür. İncelenen varyantlar şunlardır:

- w/o semantics
- w/o semantic prior
- w/o reward shaping
- w/o confidence
- w/o battery awareness
- w/o queue awareness
- w/o mobility features
- w/o partial offloading

Faz 5 sonucunda en güçlü ve kararlı etki mobility features ve partial offloading üzerinde görülmüştür.

Evaluation-only ablation sonuçlarında:

| Variant | Delta vs Full |
|---|---:|
| w/o partial offloading | -8.23 puan |
| w/o mobility features | -8.80 puan |
| w/o semantics | yaklaşık 0.00 puan |
| w/o reward shaping | yaklaşık 0.00 puan |

Retraining-based ablation daha güçlü kanıt olarak değerlendirilmiştir. Bu sonuçlarda mobility features üç algoritmada da sistematik performans düşüşüne neden olmuştur. Semantic bileşenin katkısı PPO ve A2C tarafında görünür hale gelmiş, DQN tarafında daha zayıf ayrışmıştır. Reward shaping etkisi ise mevcut protokolde tam netleşmemiştir.

Ablation study'nin ana çıkarımı:

> Partial offloading ve mobility features, offloading karar kalitesi için kritik bileşenlerdir. Semantic ve reward shaping bileşenlerinin etkisi ise daha gelişmiş metriklerle yeniden analiz edilmelidir.

## 12. Tartışma

Deneysel sonuçlar, semantic-aware DRL yaklaşımının task offloading probleminde umut verici olduğunu göstermektedir. PPO modeli sentetik ortamda en güçlü adaylardan biri olmuş, trace-driven ortamda ise oldukça yüksek başarı oranlarına ulaşmıştır.

Ancak sonuçlar yalnızca success rate üzerinden değerlendirilmemelidir. Faz 5 ve Faz 7 bulguları, dominant action ve decision structure metriklerinin de kritik olduğunu göstermiştir. Bazı teacher policy'ler daha yüksek başarı oranı verse bile final policy'yi Full Cloud gibi istenmeyen davranışlara yöneltebilmektedir.

Semantic prior'ın etkisi bazı ablation senaryolarında sınırlı görünmüştür. Bu durum semantic prior'ın değersiz olduğu anlamına gelmemektedir. Daha doğru yorum, mevcut environment ve reward tasarımının semantic katkıyı her koşulda yeterince ayrıştırmadığıdır. Bu nedenle gelişmiş metrikler ve daha hassas deney protokolleri gereklidir.

Trace-driven sonuçların çok yüksek olması olumlu olmakla birlikte dikkatli yorumlanmalıdır. Domain-shift analizinde synthetic -> trace ve trace -> synthetic yönleri arasında ciddi fark görülmüştür. Bu, farklı veri dağılımlarının model davranışını yönlü şekilde etkilediğini göstermektedir.

Graph-aware policy çalışması, Faz 7 sonunda görülen Edge %75 ağırlıklı karar yapısını daha iyi analiz etmek ve iyileştirmek için doğal bir sonraki adımdır. Offloading problemi cihaz, task, edge server ve cloud ilişkilerinden oluştuğu için graph temsili bu probleme yapısal olarak uygundur.

## 13. Sınırlılıklar

Bu çalışmanın bazı sınırlılıkları vardır:

- Sistem henüz gerçek fiziksel edge testbed üzerinde doğrulanmamıştır.
- Trace-to-task mapping bazı varsayımlar içermektedir.
- Semantic analyzer sınırlı model/fallback yapısı ile çalışmaktadır.
- Reward shaping etkisi tüm koşullarda net ayrışmamıştır.
- Graph-aware policy henüz ilk forward path ve unit test aşamasındadır.
- Action diversity problemi tamamen çözülmemiştir.
- İstatistiksel analiz ve confidence interval kapsamı Faz 9'a devredilmiştir.

Bu sınırlılıklar çalışmanın değerini azaltmaktan çok, journal yayını öncesinde güçlendirilmesi gereken noktaları göstermektedir.

## 14. Sonuç ve Gelecek Çalışmalar

Bu çalışmada IoT ve edge computing ortamlarında LLM-guided semantic-aware task offloading problemi ele alınmıştır. Geliştirilen sistem, fiziksel state bilgisi ile semantic prior'ı birleştiren DRL tabanlı bir karar mekanizması sunmaktadır.

Çalışmanın temel sonuçları şunlardır:

- PPO, sentetik full model retraining sonucunda %76.17 başarı oranına ulaşmıştır.
- Trace-driven hold-out testte PPO modeli %99.60 başarı göstermiştir.
- Two-stage training, scratch PPO'ya göre +12.20 puanlık başarı artışı sağlamıştır.
- Partial offloading ve mobility features bileşenleri ablation analizinde kritik bulunmuştur.
- Domain-shift analizinde synthetic -> trace ve trace -> synthetic yönleri arasında asimetri gözlenmiştir.
- Graph-aware policy için graph state builder ve ilk policy forward path başarıyla geliştirilmiştir.

Gelecek çalışmalar:

- Graph-aware policy eğitim ve evaluation deneylerinin tamamlanması
- Semantic prior fusion stratejilerinin karşılaştırılması
- MLP-PPO, Pretrained MLP-PPO ve Graph-aware policy karşılaştırması
- p99 latency, deadline miss ratio, energy per success, fairness, jitter ve QoE metriklerinin eklenmesi
- 5-seed protokolü, confidence interval ve istatistiksel testlerin uygulanması
- Daha geniş trace datasetleriyle genelleme analizinin güçlendirilmesi
- LLM self-reflection ve experience replay mekanizmasının eklenmesi

## 15. Kaynakça ve Ekler

Bu taslak journal makalesine dönüştürülürken kaynakça aşağıdaki çalışma gruplarından oluşturulmalıdır:

- Mobile edge computing ve task offloading çalışmaları
- DRL-based edge offloading çalışmaları
- PPO, DQN ve A2C temel algoritma makaleleri
- Semantic-aware veya LLM-guided decision making çalışmaları
- AgentVNE ve staged training yaklaşımıyla ilişkili çalışmalar
- Graph neural network tabanlı resource allocation ve scheduling çalışmaları

Eklerde verilmesi önerilen materyaller:

- Hyperparameter tabloları
- Config dosyaları
- Trace-to-task mapping varsayımları
- Ek ablation tabloları
- Ek grafikler
- Model checkpoint listesi
- GraphState sözleşmesi
- Reproducibility komutları
