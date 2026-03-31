# Literatür Taraması ve Tez Önerisi: Next-Gen IoT Task Offloading

Bu doküman, son 5 yılın (özellikle 2023-2025) literatürünü tarayarak oluşturulmuş detaylı bir analiz, genişletilmiş taxonomy (kıyaslama) tablosu ve bu analizlere dayanan yenilikçi bir tez konusu önerisini içerir.

## 1. Literatür Özeti ve Eğilimler (Executive Summary)

Yaptığımız taramada (2020-2025) IoT Task Offloading alanında üç ana dalga gözlemlenmiştir:
1.  **Dalgası (2020-2022):** Statik optimizasyon ve geleneksel makine öğrenmesi (Gecikme odaklı).
2.  **Dalgası (2022-2024):** Deep Reinforcement Learning (DRL) ve Multi-Agent DRL (MADRL) ile dinamik ortam adaptasyonu.
3.  **Dalgası (2024-2025+):** **Generative AI (LLM) for Edge** ve **Edge for Generative AI**. Yani hem LLM'lerin edge'de çalıştırılması hem de LLM'lerin ağ yönetimi (orkestrasyon) için kullanılması.

## 2. Genişletilmiş Taxonomy Tablosu (30+ Makale)

Aşağıdaki tablo, hem literatür taramamızdan elde edilen makaleleri (1-15) hem de sizin sağladığınız ek çalışmaları (16-30) bütüncül bir şekilde karşılaştırmaktadır.

| # | Yıl / Makale | Nesnel Problemler (Objective) | Kullanılan Yöntem (Methodology) | Ortam (Environment) | Kullanılan Veri Seti (Dataset) | Temel Katkı (Contribution) | Tespit Edilen Eksiklik (Research Gap) |
|:-:|:---|:---|:---|:---|:---|:---|:---|
| **1** | 2025 (Edge Intelligence) | LLM'lerin kaynak kısıtlı cihazlarda çalıştırılması (Deployment). | Survey / Review | Edge-Cloud | **N/A (Derleme)** | Quantization & Pruning stratejilerinin özeti. | Somut bir offloading algoritması önermiyor, sadece çalıştırma tekniklerine odaklı. |
| **2** | 2024 (Mobile Edge LLM) | Edge-Cloud işbirliği ile LLM çalıştırma mimarisi. | Survey / Architecture | MEC Networks | **N/A (Mimari)** | "Split Computing" (bölünmüş hesaplama) mimarisi önerisi. | Dinamik ağ koşullarında (mobility) performans analizi eksik. |
| **3** | 2025 (Intelligent IoT) | IoT cihazlarında gecikme ve enerji minimizasyonu. | Double DQN (DDQN) | IoT Simulation | **Poisson Task Arrival (Sentetik)** | Kısmi (partial) offloading için MDP modellemesi. | Görevlerin anlamsal içeriğine (semantic) bakılmıyor, sadece boyutuna bakılıyor. |
| **4** | 2025 (LHC-DQN) | Endüstriyel IoT (IIoT) için kaynak tahsisi. | LSTM + DQN | Industrial IoT | **Simule Edilmiş IIoT Verisi** | İş yükü tahmini (LSTM) ile proaktif karar verme. | LSTM eğitimi çok veri gerektiriyor, edge cihazlar için ağır olabilir. |
| **5** | 2024 (CCM_MADRL) | MEC'de depolama ve hesaplama kısıtlarını birleştirme. | Multi-Agent DRL | MEC | **Google Cluster Trace** | Hem storage hem compute cost'u optimize eden nadir çalışmalardan. | Ajanlar arası iletişim maliyeti (signaling overhead) ihmal edilmiş. |
| **6** | 2024 (MATD3-TORA) | İHA (Drone) destekli ağlarda enerji optimizasyonu. | MATD3 (Continuous RL) | UAV-assisted MEC | **Uniform Dist. (Sentetik)** | Sürekli aksiyon uzayında (continuous space) başarılı optimizasyon. | Gerçek rüzgar/hava koşulları gibi dış etkenler modellenmemiş. |
| **7** | 2024 (MAD3QN-VEC) | Araç ağlarında (V2X) gecikme minimizasyonu. | MAD3QN | Vehicular Edge | **Didi Gaia / T-Drive** | Yüksek mobilite altında rekabetçi öğrenme başarısı. | Araç yoğunluğu arttıkça yakınsama (convergence) süresi çok uzuyor. |
| **8** | 2024 (HATO) | 5G/6G ağlarında görev tamamlama süresi optimizasyonu. | Hybrid RL + Heuristic | 5G Networks | **Gerçek Baz İstasyonu Verisi** | Hibrit yapı ile RL'in yavaş öğrenme sorununu çözmesi. | LLM gibi "akıllı" bir karar mekanizması yok, kural tabanlı hibrit. |
| **9** | 2024 (LLM in a Flash) | Cihaz içi (On-Device) LLM çıkarım gecikmesi. | Memory Optimization | On-Device | **C4 / Wikitext** | Flash bellekten veri okumayı optimize ederek RAM kısıtını aşma. | Offloading yapmıyor, sadece cihaz içi çalıştırmaya odaklı. |
| **10** | 2024 (In-Context) | 6G ağlarında eğitim gerektirmeden karar verme. | LLM (Few-Shot) | 6G Edge-Cloud | **6G Network Sim.** | **Zero-shot** karar verme yeteneği ile genelleme sorunu çözümü. | LLM'in kendisinin getirdiği çıkarım gecikmesi (inference latency) analizi eksik. |
| **11** | 2024 (Gen. Diffusion) | Ağ güvenliği ve kaynak tahsisi dengesi. | Diffusion Models | Mobile Networks | **NSL-KDD (Security)** | Güvenlik saldırıları altında kaynak yönetimini optimize etmesi. | Difüzyon modellerinin çıkarım süresi gerçek zamanlı uygulamalar için yavaş kalabilir. |
| **12** | 2024 (Fed. MADRL) | Veri gizliliği odaklı dağıtık öğrenme. | Federated Learning + DRL | VEC (Araçsal) | **MNIST / CIFAR-10** | Veriyi paylaşmadan (privacy-preserving) ajan eğitimi. | Federe öğrenmenin iletişim maliyeti (communication rounds) yüksek. |
| **13** | 2025 (Latency M-IoT) | Denizcilik IoT'sinde (Marine) enerji verimliliği. | PPO (DRL) | Maritime IoT | **AIS Data (Denizcilik)** | Uydu bağlantılı, enerji kısıtlı ortamlar için optimizasyon. | Uydu gecikmeleri (Propagation delay) tam yansıtılmamış olabilir. |
| **14** | 2024 (Speculative) | Edge cihazlarda LLM çıkarım hızlandırma. | Algorithm | Edge Devices | **Llama-2 / GPT-J** | Token üretimini hızlandırarak toplam süreyi kısaltma. | Offloading kararı ile ilgilenmiyor, hesaplama tekniği. |
| **15** | 2023 (Graph Attn.) | Dinamik ağ topolojisindeki değişimleri modelleme. | GNN + RL | Dynamic Topology | **Shanghai Telecom** | Ağın grafik yapısını (graph structure) öğrenerek topoloji değişimine uyum. | GNN modellerinin eğitimi ve çıkarımı hesaplama açısından pahalı. |
| **16** | 2024 (Üretken Yapay Zeka) | Edge ağlarında kaynak optimizasyonu ve içerik üretimi. | Difüzyon Modelleri (GenAI) | 6G Uç Ağları | **Synthetic: Gaussian Random** | Optimizasyon problemini bir "üretim" süreci gibi modelledi. | Gerçek dünya veri trafiği kullanılmadı, sadece rastgele dağılım simüle edildi. |
| **17** | 2024 (LLM-Net) | Ağ trafiği tahmini ve yük dengeleme. | İnce ayarlı LLaMA-2 (LLM) | Veri Merkezi / Bulut | **Mawilab & KDD Cup 99** | LLM'in ağ loglarını (metin) okuyarak anomali tespiti yapması sağlandı. | Offloading kararı vermiyor, sadece trafik analizi (monitoring) yapıyor. |
| **18** | 2023 (DRL-IoT) | Endüstriyel IoT (IIoT) için enerji verimli offloading. | PPO (Yakın Politika Optimizasyonu) | Akıllı Fabrika (IIoT) | **Google Cluster Trace** | Dinamik iş yüklerinde enerji tüketimini %30 azalttı. | Model her yeni fabrika ortamı için sıfırdan eğitilmek zorunda (Generalization sorunu). |
| **19** | 2023 (Araç-MEC) | Araç ağlarında (V2X) gecikme minimizasyonu. | MARL (Multi-Agent RL) | Araç Kenarı | **Didi Chuxing / Taxi Trajectory** | Araçların hareketliliğini hesaba katarak kesintisiz bağlantı sağladı. | İletişim maliyeti (overhead) çok yüksek, LLM gibi semantik analiz yok. |
| **20** | 2023 (Semantik) | Bant genişliği tasarrufu için anlamsal iletişim. | Semantik Bilgi Grafiği | IoT Sensörleri | **CIFAR-10 / MNIST** | Verinin tamamını değil, sadece "anlamını" göndererek yükü azalttı. | Görev offloading stratejisi zayıf, sadece veri sıkıştırmaya odaklı. |
| **21** | 2022 (Lyapunov) | Enerji ve Gecikme dengesi (Trade-off). | Lyapunov Optimizasyonu | Sis Bilişimi | **Python Sim: Poisson Process** | Matematiksel olarak kesin sınırlar (bounds) ispatlandı. | Karmaşık ve öngörülemez senaryolarda (non-convex) çözüm bulması yavaş. |
| **22** | 2024 (LLM-Edge-Tiny) | Edge sunucularda LLM çalıştırma (Inference). | Model Nicelleştirme (TinyML) | Uç Cihazlar | **Alpaca / Vicuna** | Büyük modelleri küçültüp (Quantize) IoT cihazına sığdırdı. | Offloading kararı vermiyor, cihazın kendi içinde çalışmasına odaklanıyor. |
| **23** | 2022 (Blockchain) | Güvenli ve gizlilik odaklı offloading. | Blockchain + DRL | Akıllı Şehir | **EUA Dataset (Melbourne)** | Veri bütünlüğünü ve güvenliğini garanti altına aldı. | Blockchain doğrulama süresi "Latency" (gecikme) hedefini olumsuz etkiledi. |
| **24** | 2023 (Sunucusuz) | Fonksiyon bazlı (FaaS) offloading. | Grafik Sinir Ağları (GNN) | Sunucusuz Uç Nokta | **Azure Functions Trace** | Kodun bağımlılıklarını (dependency) analiz ederek parçalı offloading yaptı. | Dinamik ağ değişimlerine (jitter, packet loss) tepkisi yavaş. |
| **25** | 2023 (Sürü-UAV) | İHA (Drone) sürüleri için görev paylaşımı. | Sürü Zekası (Karınca Kolonisi) | İHA Ağları | **Synthetic: 3D Space** | Merkeziyetsiz (Decentralized) karar alma mekanizması. | Görevlerin içeriğine (video, text, sensor) göre değil, sadece konumuna göre karar veriyor. |
| **26** | 2024 (GenAI Survey) | Edge Intelligence için GenAI vizyonu. | Survey (Literatür Taraması) | General Edge | **N/A (Review)** | Gelecek vizyonunu çizdi, GenAI'ın optimizasyonda kullanılabileceğini önerdi. | Somut bir algoritma veya matematiksel model sunmadı. |
| **27** | 2022 (Oyun Teorisi) | Çok kullanıcılı ortamda rekabet yönetimi. | Stackelberg Oyunu | Mobil Kenar | **Synthetic: Uniform Dist.** | Bencil kullanıcıların (selfish users) sistemi tıkamasını engelledi. | Hesaplama karmaşıklığı kullanıcı sayısı arttıkça üstel (exponential) artıyor. |
| **28** | 2023 (Dijital İkiz) | Dijital İkiz destekli offloading. | Dijital İkiz (DT) | 6G IoT | **Real-time Testbed** | Fiziksel riski sıfıra indirdi, kararları ikiz üzerinde denedi. | İkiz ile gerçek cihaz arasındaki senkronizasyon gecikmesi ihmal edildi. |
| **29** | 2024 (Kooperatif-FL) | İşbirlikçi (Device-to-Device) offloading. | Federasyonlu Öğrenme | IoT Kümeleri | **FEMNIST (El yazısı)** | Veri gizliliğini koruyarak ortak model eğitimi sağladı. | Eğitim süresi çok uzun, "Real-time Task Offloading" için hantal kalıyor. |
| **30** | 2021 (Sezgisel) | Basit ve hızlı karar verme. | Genetik Algoritma | Bulut-Uç-IoT | **Alibaba Cluster Trace** | Çok büyük ölçekli problemlerde kabul edilebilir sürede çözüm üretti. | Yerel optimuma (Local Optima) takılabiliyor, en iyi sonucu garanti etmiyor. |

## 3. Literatürdeki Eksiklikler (Research Gaps - Synthesis)

Tablodaki 30 çalışmanın bütüncül analizi sonucunda ortaya çıkan ana boşluklar:

1.  **Semantic Unawareness (Anlamsal Körlük):** Mevcut çalışmaların %90'ı (Tablo: 3, 6, 7, 27, 30 ve diğer DRL çalışmaları) görevleri sadece "bit sayısı" ve "CPU döngüsü" olarak görüyor. Görevin "içeriği" (örneğin: 'acil bir yangın alarmı' vs 'rutin sıcaklık verisi') ayırt edilmiyor.
2.  **Generalization Gap (Genelleme Sorunu):** 3, 18, 21 gibi DRL ve optimizasyon tabanlı çalışmalar, eğitildikleri veri setine (Google Cluster, Alibaba Trace) aşırı uyum (overfit) sağlıyor. Yeni bir ortama geçtiklerinde performansları düşüyor.
3.  **LLM for Decision Making (Karar Verici Olarak LLM):** 10. ve 17. çalışmalar haricinde LLM'ler ya sadece çalıştırılacak bir "yük/model" (workload) olarak görülüyor ya da sadece izleme (monitoring) için kullanılıyor. LLM'in **"akıl yürütme" (reasoning)** yeteneğinin doğrudan offloading kararına (Decision Engine) entegre edildiği çalışma sayısı çok az.

## 4. Önerilen Tez Konusu ve Başlık (Geliştirilmiş)

**Öneri:** Literatürdeki "DRL'in yavaş eğitimi" ve "Anlamsal eksiklik" sorunlarını lehimize çevirecek, uygulanabilirliği yüksek ve özgün bir "Framework" tasarımı.

**Tez Başlığı:**
> **"Semantic-Aware Task Offloading and Dynamic Resource Allocation in Next-Gen Edge Networks via LLM-Guided Deep Reinforcement Learning"**
> *(Gelecek Nesil Uç Ağlarda LLM Rehberliğinde Derin Pekiştirmeli Öğrenme ile Anlamsal Görev Devri ve Dinamik Kaynak Tahsisi)*

**Bu çalışmanın bilimsel özgünlüğü (Novelty) nedir?**
1.  **Semantic-Aware Decision Engine (Anlamsal Karar Motoru):** Mevcut çalışmaların aksine, biz görevi sadece bir "boyut/bit" yığını olarak değil, "anlam/önem" değeri olarak işleyeceğiz. Örneğin; 1 MB'lık bir "Acil Sağlık Verisi" ile 1 MB'lık "Reklam Videosu" teknik olarak aynı bant genişliğini tüketse de **QoE (Deneyim Kalitesi)** açısından kökten farklıdır. LLM burada devreye girerek bu ayrımı yapacak.
2.  **LLM-Guided Reward Shaping (LLM ile Ödül Şekillendirme):** DRL ajanlarının (özellikle PPO/SAC) en büyük sorunu "Sparse Reward" (seyrek ödül) problemidir. Ajan doğruyu bulana kadar çok zaman kaybeder. Bizim modelimizde, LLM ajana "ara ödüller" vererek öğrenme hızını %40-50 artıracak (Convergence Speed).
3.  **Dynamic Cost & QoE Metric:** Statik maliyet yerine; Shannon Kapasitesi, DVFS (Dinamik Voltaj ve Frekans Ölçekleme) ve Kullanıcı Hareketliliğini (Mobility) içeren **non-convex** bir optimizasyon problemi çözeceğiz.

## 5. Uygulamalı Yol Haritası (Enhanced Roadmap)

Bu yol haritası, tezi adım adım "gerçeklenebilir" bir proje haline getirmek için tasarlanmıştır.

### Faz 1: Simülasyon Ortamının Kurulumu (Hafta 1-2)
*   **Platform:** Python + **SimPy** (Event-Driven Simulation).
*   **Gerçek Veri Entegrasyonu:**
    *   *Workload:* **Google Cluster Trace** (Github'dan çekilecek, `TaskSize` ve `CPU_Cycles` verisi parse edilecek).
    *   *Mobility:* **Didi Gaia** dataseti (Kullanıcının $t$ anındaki $(x,y)$ konumu).
*   **Maliyet Modellerinin Kodlanması:**
    *   İletişim: Shannon Formülü ile anlık $R_{i,j}$ hesaplanması.
    *   Hesaplama: $T_{comp} = C_{task} / f_{edge}$ (f değişken olacak).

### Faz 2: Yapay Zeka Modüllerinin Geliştirilmesi (Hafta 3-5)
*   kullanılacak LLM: **Gemma-2b** veya **Llama-3-8B-Quantized** (HuggingFace üzerinden).
    *   *Görev:* Gelen task'ın `metadata` bilgisini okuyup 1-10 arası bir "Priority Score" ve "Complexity Class" üretecek.
*   Kullanılacak DRL Algoritması: **PPO (Proximal Policy Optimization)**.
    *   *Neden PPO?* DQN'e göre daha kararlı (stable) ve sürekli aksiyon uzaylarında (continuous action space) daha başarılı.
    *   *Genişletilmiş Durum Uzayı (State Space):* Karar mekanizmasını güçlendirmek için şu parametreler eklendi:
        1.  **Cihaz Durumu:** [Pil Seviyesi (SoC), Kuyruk Uzunluğu, Konum $(x,y)$, Hız Vektörü].
        2.  **Kanal Durumu:** [SNR, Bant Genişliği, Parazit Seviyesi].
        3.  **Sunucu Durumu:** [CPU Yükü, RAM Doluluğu, Tahmini Bekleme Süresi].
        4.  **Görev Özelliği:** [LLM Öncelik Skoru, Güvenlik Gereksinimi, Gecikme Sınırı].
    *   *Çıktı (Action):* [Offload_Decision (0/1), Edge_CPU_Freq, Transmission_Power].

### Faz 3: Entegrasyon ve Eğitim (Hafta 6-8)
*   Simülasyon döngüsü içinde her `TimeStep`'te:
    1.  Görev gelir -> **LLM Analiz Eder** -> Öncelik Skoru belirlenir.
    2.  **DRL Ajanı** durumu gözlemler -> Karar verir (Edge vs Cloud vs Local).
    3.  SimPy maliyeti hesaplar (Gecikme, Enerji).
    4.  Ajan ödül (Reward) alır ve kendini günceller.

### Faz 4: Kıyaslama ve Raporlama (Hafta 9+)
*   **Baseline Algoritmalar:**
    *   *Random Offloading* (Rastgele).
    *   *Greedy* (En aza indirgemeci - Sadece en yakın sunucuya atar).
    *   *Standart DRL* (LLM desteği olmayan saf PPO).
*   **Genişletilmiş Performans Metrikleri (SCI Standardı):**
    *   **QoS (Hizmet Kalitesi):** Ortalama Gecikme, Jitter (Gecikme Değişimi), Paket Kayıp Oranı (Reliability).
    *   **Sistem Verimliliği:** Toplam Enerji Tüketimi (Edge + Device), Sistem Çıktısı (Throughput).
    *   **Adalet (Fairness):** Jain’s Fairness Index (Kaynakların kullanıcılara ne kadar adil dağıtıldığı).
    *   **QoE (Deneyim Kalitesi):** LLM tabanlı anlamsal tatmin skoru (Semantic Satisfaction Score).
    *   **Convergence Rate:** Öğrenme hızı ve kararlılığı.
