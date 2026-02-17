# Walkthrough: Phase 2 - AI & Reinforcement Learning Integration

Bu çalışma, IoT görev atama (task offloading) problemini çözmek için **PPO (Proximal Policy Optimization)** algoritmasını ve **LLM** tabanlı ödül şekillendirmeyi (Reward Shaping) sisteme entegre eder.

## 1. Yapılan Yenilikler
- **RL Environment (`src/rl_env.py`)**: Simülasyon, standart bir `Gymnasium` ortamına dönüştürüldü.
- **PPO Agent (`src/train_agent.py`)**: Stable-Baselines3 kullanılarak zeki bir ajan eğitilmeye başlandı.
- **Reward Shaping**: LLM'den gelen semantik skorlara göre ödül fonksiyonu dinamik olarak değişir.
- **Profesyonel GUI**: Side Panel dikey olarak ayrıldı (Dashboard + Decision Feed).

## 2. Ajan Eğitimi Nasıl Çalıştırılır?
Ajanı eğitmek için terminalde şu komutu çalıştırabilirsiniz:
```powershell
src\venv\Scripts\python.exe src\train_agent.py
```
*Bu işlem, cihaz/görev özelliklerine göre en iyi atama stratejisini (Local, Edge, Cloud) öğrenen bir model üretir.*

## 3. Simülasyonu Çalıştırma
Görsel Dashboard'u görmek için:
```powershell
.\run_simulation.bat
```

## 4. Dosya Yapısı
- **`doc/`**: Tüm proje dökümantasyonu (Task, Roadmap, Plan).
- **`src/`**: Ana kod dizini.
- **`src/requirements.txt`**: Tüm kütüphaneler burada merkezi olarak tutulur.
- **`src/progress/`**: Akademik makale taslakları.

---
*Proje Durumu: Faz 2 Tamamlandı. Faz 3 (Test ve Analiz) için her şey hazır.*
