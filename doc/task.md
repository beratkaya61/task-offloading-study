# Araştırma ve Tez Planlaması: IoT Task Offloading

Bu dosya, "Next-Gen IoT Task Offloading" konulu araştırma süreci ve simülasyon tasarımı için oluşturulmuştur.

> [!NOTE]
> Tüm dokümantasyon dosyaları karmaşıklığı önlemek için `./doc/` klasörüne taşınmıştır.

- [x] **Temel Kavramların Açıklanması**
    - [x] Task Offloading konseptinin (Device-Edge-Cloud) analojilerle anlatımı.
    - [x] Katmanların ve problemin oluşumunun açıklanması.
- [x] **Literatür Taraması (Son 5 Yıl, SCI/Uluslararası)**
    - [x] Anahtar kelimelerin belirlenmesi (Keywords).
    - [x] 15+ Makalenin tespiti (LLM, Resource Allocation, Latency odaklı).
    - [x] Makalelerin Taxonomy Tablosu ile kıyaslanması (Amaç, Yöntem, Veri Seti, Sonuçlar).
- [x] **Faz 1: Simülasyon Ortamının Kurulumu** (Tamamlandı)
    - [x] `simulation_env.py`: SimPy tabanlı temel ortamın kurulması.
    - [x] `Dynamic Models`: Shannon, DVFS, Battery ve Jitter modellerinin kodlanması.
    - [x] `Data Loaders`: Google Cluster Trace ve Didi Gaia veri okuyucularının yazılması (Mock/Real).
    - [x] **Visualization (GUI)**: PyGame ile simülasyonun harita üzerinde (Real-time) görselleştirilmesi.
    - [x] **GUI Enhancement**: Legend, pil göstergesi, animasyonlar, istatistik paneli eklenmesi.
    - [x] **Real Data Integration**: Google Cluster Trace CSV okuyucu ve entegrasyonu.
- [x] **Faz 2: Yapay Zeka (AI) Entegrasyonu**
    - [x] `LLM Semantic Analyzer`: Task önceliklendirmesi için LLM entegrasyonu.
    - [x] `AI Decision Log`: Karar mekanizmasını açıklayan detaylı panel (Geliştirildi).
    - [x] **Academic Knowledge Base**: `src/progress` altında makale formatında dokümantasyon oluşturulması (01-04 Tamamlandı).
    - [x] **GUI Professional Redesign**: Booklet ve Feed ekranlarının ayrılması, ikonların düzeltilmesi ve Glassmorphism uygulanması.
    - [x] **Deep System Architecture**: Roadmap şemalarının tüm derinliği ve parametre alışverişiyle güncellenmesi.
    - [x] **Unified Execution**: Simülasyonun PPO ajanını otomatik yüklemesi ve `run_simulation.bat` entegrasyonu.
    - [x] `PPO Agent`: Deep Reinforcement Learning ajanının (Stable-Baselines3) kurulumu ve eğitimi.
        - [x] Gymnasium ortamı (`rl_env.py`) ve Reward Shaping mantığı kodlandı.
        - [x] Eğitim betiği (`train_agent.py`) hazırlandı.
    - [x] `Reward Shaping`: LLM destekli ödül fonksiyonunun tasarlanması ve simülasyona entegrasyonu.
- [ ] **Faz 3: Test, Analiz ve GUI İyileştirmeleri**
    - [x] `GUI SCROLL`: Karar akışının mouse wheel ile kaydırılabilir yapılması.
    - [x] `UI Visibility`: Dark mode ile çakışan siyah yazıların ve görünmeyen barların düzeltilmesi.
    - [x] `GUI LAYOUT`: Sol Methodology paneli eklenmesi (Screen Width 1800).
    - [x] `GUI ZOOM`: Harita alanında mouse wheel ile zoom/pan yeteneği.
    - [x] `NODE STATS`: Offload istatistiklerinin Edge-N bazlı kırılımı.
    - [x] `HEALTH STATUS`: Device ve Edge bazlı spesifik sağlık göstergeleri.
    - [x] `UI POLISH`: Node Health ve Stats panellerindeki çakışmanın giderilmesi.
    - [x] `METHODOLOGY+`: Teknik panelin detaylandırılması ve fontların büyütülmesi.
    - [x] `AI Humanized Logic`: Karar loglarının "LLM Analizi" vs "AI Önerisi" olarak netleştirilmesi.
- [x] **Faz 4: Baselines ve Karşılaştırmalı Analiz**
    - [x] `Shadow Baselines`: PPO ile eş zamanlı olarak Random ve Greedy maliyetlerinin hesaplanması.
    - [x] `Performance Drawer`: Sayfanın altına kazanç sidebar/paneli eklenmesi.
    - [x] `Gain Analysis`: PPO'nun yüzde kaç daha verimli olduğunun (Latency/Energy) anlık gösterimi.
    - [x] `Explainable Results`: Kazancın nedenlerini (örn. "PPO kuyrukları önceden tahmin etti") açıklayan detaylı metinler.
    - [x] `Metrics Engine`: Fairness, Jitter ve QoE hesaplama mantığı.
