# Phase 4 Raporu: Baseline Ailesinin Genişletilmesi ve İlk RL Deneyleri

## 📋 Yapılan Çalışmalar
Bu fazda, projenin kıyaslama altyapısını güçlendirmek ve derin pekiştirmeli öğrenme modellerini yeni mimariye entegre etmek için aşağıdaki adımlar tamamlanmıştır:

1.  **Baseline Genişletme:**
    *   `LocalOnly`, `EdgeOnly`, `CloudOnly`, `Random`, `GreedyLatency` baselineları standartlaştırıldı.
    *   **Genetic Algorithm (GA)** tabanlı meta-sezgisel baseline sisteme entegre edildi.
2.  **PPO v2 Eğitimi:**
    *   11 boyutlu (5 fiziksel + 6 semantik) yeni durum uzayına (state space) uyumlu **PPO v2** modeli sıfırdan eğitilmiştir.
    *   NumPy 2.0 ve Torch 2.2.2 uyumluluk sorunları, NumPy 1.26.4 sürümüne dönülerek ve DType zorlamaları (float32) yapılarak çözülmüştür.
3.  **Hata Giderme:**
    *   `GeneticAlgorithm` ve diğer baselinelardaki `numpy.ndarray` hashleme hataları (`unhashable type`) giderilmiştir.
    *   SB3 modellerinin `predict` metodunda vektörize edilmiş gözlem beklentisi ile bireysel gözlem arasındaki uyuşmazlıklar analiz edilmiştir.

## 📊 Deneysel Sonuçlar (Özet)
Son yapılan testlerde (150 görev üzerinden) elde edilen başarı oranları ve ödül (reward) değerleri:

| Model | Başarı Oranı (%) | Ortalama Ödül |
| :--- | :--- | :--- |
| **Genetic Algorithm** | **%58.00** | **2540.98** |
| **Cloud Only** | %48.00 | 2378.51 |
| **Greedy Latency** | %49.33 | 2347.33 |
| **Random** | %41.33 | 1477.28 |
| **Edge Only** | %36.67 | 1645.70 |
| **Local Only** | %20.67 | -1220.14 |

*Not: PPO v2 modeli eğitim aşamasında %35.67 başarı oranı ile başlamış, ancak değerlendirme aşamasındaki teknik bir uyuşmazlık (obs shape) nedeniyle tabloya tam yansıtılamamıştır.*

## 🚧 Karşılaşılan Sorunlar ve Çözümler
*   **DType Sorunu:** NumPy 2.0 ile Torch arasındaki uyumsuzluk, `state_builder.py` içerisinde `.astype(np.float32)` zorlaması ile aşıldı.
*   **PPO Predict Hatası:** SB3 PPO modelinin tekli gözlemleri (11,) yerine (1, 11) şeklinde batch formatında beklemesi, değerlendirme scriptinde revizyon gerektirmektedir.

## 📍 Mevcut Durum ve Gelecek Adım
Faz 4'ün büyük bir kısmı (Baseline entegrasyonu ve sıfırdan RL eğitimi) başarıyla tamamlanmıştır.

**Nerede Kaldık:**
*   PPO v2'nin `run_baselines.py` içerisinde sorunsuz koşturulması için gözlem boyutunun (batch dimension) ayarlanması gerekiyor.
*   **Faz 5 (Sistematik Ablation Study)** adımına geçmeye hazırız. Bu aşamada semantik bilgilerin (LLM analizleri) ve farklı reward bileşenlerinin ajan üzerindeki etkisini ölçeceğiz.
