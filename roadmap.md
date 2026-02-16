# Araştırma ve Tez Yol Haritası (Roadmap)

Bu dosya, "LLM-Empowered Semantic Task Offloading" projesinin görsel ve metin tabanlı uygulama planıdır.

## 1. Görsel Zaman Çizelgesi (Gantt Chart)

```mermaid
gantt
    title Tez ve Makale Çalışma Takvimi
    dateFormat  YYYY-MM-DD
    section Hazırlık
    Konsept & Literatür (30+ Makale) :done, des1, 2026-02-15, 2d
    Tez Konusu & Gap Analizi         :done, des2, 2026-02-16, 1d
    Yol Haritası & Mimari            :active, des3, 2026-02-16, 1d

    section Faz 1: Simülasyon (Python)
    Gerçek Veri Seti (Google/Didi)   :crit, sim1, 2026-02-17, 2d
    Dinamik Modeller (Shannon/DVFS)  :sim2, 2026-02-18, 2d
    SimPy Çekirdek Kurulumu          :sim3, 2026-02-20, 3d

    section Faz 2: Yapay Zeka (AI)
    LLM Entegrasyonu (Gemma/Llama)   :ai1, 2026-02-23, 4d
    Deep RL (PPO) Ajanı Eğitimi      :ai2, 2026-02-27, 5d
    
    section Faz 3: Analiz & Yazım
    Baseline Kıyaslamaları           :test1, 2026-03-05, 3d
    Jain's Fairness & QoE Analizi    :test2, 2026-03-08, 3d
    Makale Yazımı (IEEE Trans.)      :write1, 2026-03-12, 10d
```

## 2. Sistem Mimarisi (System Architecture)

Sistem; Gerçek Veri, Simülasyon Çekirdeği ve Yapay Zeka Katmanı olmak üzere 3 ana bloktan oluşur.

```mermaid
flowchart TD
    %% 1. Veri Katmanı
    A[Google Trace\nTask Profiles] --> B(Task Generator)
    C[Didi Gaia\nUser Locations] --> D(Mobility Manager)

    %% 2. Simülasyon & Karar (Decision)
    B --> E{Decision Engine}
    D --> E
    
    %% 3. Zeka Katmanı Inputs
    D -.-> G{Deep RL Agent}
    E -.-> F[LLM Analyzer]
    F --> G
    G -->|Action| E

    %% 4. Kuyruklar (Queues)
    E -- Local --> H[IoT Device]
    E -- Edge --> I[Edge Server]
    E -- Cloud --> J[Cloud Server]

    %% 5. Maliyetler (Cost Models)
    K[Energy Model] -.-> H & I
    L[Latency/Shannon] -.-> I & J
    
    %% 6. Sonuçlar
    H & I & J --> R[Results]
    R --> N[Metrics: QoE, Fairness]
```

## 3. Uygulama Adımları (Actionable Plan)

Bu bölüm, projeyi hayata geçirmek için kodlanacak modülleri listeler.

### Adım 1: Temel Simülasyon (`simulation_env.py`)
- [ ] **Data Loaders:** `google_trace_loader.py` ve `mobility_loader.py` (CSV okuma).
- [ ] **Channel Model:** Shannon kapasitesi ile bant genişliği hesaplayan fonksiyon.
- [ ] **Device Class:** Pil durumu ve konumu tutan sınıf.
- [ ] **Edge Server Class:** DVFS destekli işlemci sınıfı.

### Adım 2: Zeka Entegrasyonu (`ai_agent.py`)
- [ ] **LLM Interface:** HuggingFace `AutoModel` ile `prompt engineering` (Task -> Priority).
- [ ] **RL Environment:** Gymnasium (OpenAI Gym) uyumlu özel `Env` sınıfı.
- [ ] **PPO Agent:** `Stable-Baselines3` kütüphanesi ile PPO kurulumu.

### Adım 3: Çalıştırma ve Test (`main.py`)
- [ ] **Senaryo Koşumu:** 100 cihaz ve 5 Edge sunuculu senaryonun başlatılması.
- [ ] **Veri Toplama:** Her saniyenin `csv` olarak kaydedilmesi.
- [ ] **Grafik Çizimi:** `matplotlib` ile Gecikme, Enerji ve Fairness grafiklerinin çizilmesi.
