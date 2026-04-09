Bkz. ortak kavram sozlugu: v2_docs/project_concepts_glossary.md

# Phase 4 Raporu: Baseline Ailesinin Genisletilmesi ve Ilk RL Deneyleri

## Yapilan Calismalar
Bu fazda, projenin kiyaslama altyapisini guclendirmek ve derin pekistirmeli ogrenme modellerini yeni mimariye entegre etmek icin asagidaki adimlar tamamlanmistir:

1. Baseline Genisletme:
- `LocalOnly`, `EdgeOnly`, `CloudOnly`, `Random`, `GreedyLatency` baseline'lari standartlastirildi.
- `Genetic Algorithm (GA)` tabanli meta-sezgisel baseline sisteme entegre edildi.

2. PPO v2 Egitimi ve Entegrasyonu:
- 11 boyutlu (5 fiziksel + 6 semantik) yeni state space ile uyumlu `PPO v2` modeli sifirdan egitildi.
- NumPy 2.0 ve Torch 2.2.2 uyumluluk sorunlari cozuldu.

3. Kritik Hata Giderme:
- Sorun: `PPO_v2` degerlendirmede `unhashable type: numpy.ndarray` hatasi verdi.
- Cozum:
  - `src/core/evaluation.py` icine SB3 model detection fonksiyonu eklendi (`_is_sb3_model()`)
  - SB3 modelleri icin batch dimension eklendi: `obs[np.newaxis, :]`
  - SB3 `predict()` ciktilari integer'a donusturuldu: `action = int(action)`
- Sonuc: `PPO_v2` basariyla degerlendirildi.

## Deneysel Sonuclar (Faz 4 Son Run)

| Model | Basari Orani (%) | Ortalama Odul | Durum |
| :--- | ---: | ---: | :--- |
| **A2C_v2** | **100.00%** | **69.23** | Actor-Critic etkisi cok yuksek |
| **PPO_v2** | 66.67% | 70.56 | Optimize edildi |
| **DQN_v2** | 66.67% | 68.41 | Q-Learning basarisi |
| **EdgeOnly** | 66.67% | 61.31 | Fixed policy |
| **GeneticAlgorithm** | 66.67% | 53.05 | Meta-heuristic baseline |
| **GreedyLatency** | 33.33% | 39.68 | Heuristic baseline |
| **CloudOnly** | 0.00% | 20.40 | Zayif |
| **Random** | 0.00% | 38.11 | Dusuk |
| **LocalOnly** | 0.00% | -89.75 | Zayif |

## Faz 4 Tamamlama Kontrol Listesi

### Yapilan Isler
- [x] `src/baselines.py` olusturuldu ve 6+ baseline modeli entegre edildi
- [x] `src/evaluation.py` olusturuldu ve CSV loglama sistemi kuruldu
- [x] `src/metrics.py`, `src/config.py` modulleri hazirlandi
- [x] PPO v2 modeli 11-boyutlu state ile egitildi
- [x] Tum baseline'lar basariyla degerlendirildi
- [x] Ozet rapor `v2_docs/phase_5/offloading_experiment_report.md` altindaki kanonik rapor yapisina baglandi

## Karsilasilan Sorunlar ve Cozumleri

### Problem 1: Batch Dimension Uyumsuzlugu
Custom baseline'lar `(11,)` vektor ile calisirken, Stable Baselines3 modelleri `(1, 11)` batch formati bekliyordu.

Cozum:
```python
def _is_sb3_model(model):
    return hasattr(model, 'policy') and hasattr(model, 'learn')

obs_batch = obs[np.newaxis, :]
action, _ = model.predict(obs_batch, deterministic=True)
action = int(action)
```

### Problem 2: Action Type Uyumsuzlugu
SB3 `predict()` numpy array donuyordu ve `env.step()` bunu hash key olarak kullanmaya calisiyordu.

Cozum:
```python
action = int(action)
```

## Istatistiksel Yorum
- Bu faz, baseline ailesinin calisir hale geldigini gosteren ilk asama oldu.
- RL tarafinda anlamli ayrisma sinyali alindi.
- Faz 5'e gecis icin gerekli baseline ve evaluation omurgasi tamamlandi.

## Mevcut Durum
- Faz 4 tamamlandi.
- Tum temel baseline'lar calisiyor ve degerlendiriliyor.
- CSV/Markdown loglama sistemi operasyonel.
- Sonraki dogru adim Faz 5 ablation study idi.

## Faz 5 Icin Hazirlik Notu
Bu faz sonunda su soru acik kaldi:
- `PPO_v2` basarisinin ne kadari semantic bilesenlerden geliyor?

Bu nedenle Faz 5'te sistematik `ablation study` acilmasi planlandi.
