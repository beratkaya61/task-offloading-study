# 📚 Faz 6: Trace-driven Training - Detaylı Plan ve Altyapı

**Tarih:** 31 Mart 2026  
**Durum:** 🔨 ALTYAPI KURULDU, TRAINING HAZIR  
**Hedef:** %68-77 başarı oranı (Faz 5: %62.4 → +%5-15 iyileştirme)

---

## 🎯 Özet

Faz 5'teki ablation çalışmalarından önemli bulgular elde ettik:
- **Reward shaping KRITIK** (kapat = %0 başarı)
- **Partial offloading YÜKSEKetki** (-%23.4)
- **LLM semantics nominal** (<1% etki)
- **Battery/Mobility yeniden ayarlama fırsatları** (+%2.6, +%5.2)

Faz 6'da **trace-driven training** kullanarak bu bulguları gerçek(lenmiş) task dağılımları üzerinde doğrulayacak ve başarı oranını %68-77'ye çıkaracağız.

---

## 📋 Faz 6 Kapsam Tanımı

### Hedefler
| Hedef | Metriktleri | Başarı Kriteri |
|--------|-----------|---|
| **Başarı Oranı** | Success rate % | ≥ %68 (Faz 5'ten +%5.6) |
| **İyileştirme** | Phase5 vs Phase6 delta | +%5-15 beklenen |
| **Trace Realism** | Episode türü | Sentetik Didi Gaia mobility |
| **Stability** | Training convergence | NaN/Inf yok, düzgün kurva |

### Girdiler (Input)
- ✅ Phase 5 ablation bulguları (9 çalışma sonuçları)
- ✅ Trace processor (`src/core/trace_processor.py`)  
- ✅ Training config (`configs/train_trace_ppo.yaml`)
- ✅ Orchestrator (`experiments/run_trace_training.py`)
- ✅ Sentetik trace episodları (100 episode, 5000 task)

### Çıktılar (Output)
- Phase_6_Report.md (bulgular ve analiz)
- model checkpoint'leri (PPO_v3 ağırlıkları)
- Training metrics (TensorBoard logs)
- Comparison table (Faz 5 vs Faz 6)

---

## 🏗️ Altyapı Detayları

### 1. Trace Processor (`src/core/trace_processor.py`)

**Sorumluluğu:**
- Gerçek/sentetik trace verilerini yükler
- Görevleri (task) episode'lara dönüştürür
- Veri ön-işlemes (normalizasyon, outlier temizliği)
- Train/val/test split

**Algoritmik Akış:**
```
Raw Traces (CSV/JSON) 
    ↓
Load & Parse (pd.read_csv)
    ↓
Preprocess (normalize, filter outliers)
    ↓
Generate Episodes (collect tasks, sort by arrival_time)
    ↓
Split (80% train, 10% val, 10% test)
    ↓
Save JSON (reproducibility)
```

**Sentetik Trace Karakteristikleri:**
- **Devices:** 20 cihaz (simüle Didi Gaia araçları)
- **Tasks:** 5000 görev, 100 episode'a bölünmüş
- **Arrival pattern:** Poisson dağılımı (λ=0.5)
- **Task kompleksitesi (CPU cycles):** Exponential (mean=587M)
- **Veri boyutu:** Exponential (mean=574.5 KB)
- **Deadline:** Uniform [0.5s, 5s]
- **Öncelik:** Kategorik [0=LOW, 1=MEDIUM_LOW, 2=MEDIUM_HIGH, 3=HIGH]

**Test Sonuçları:**
```
✅ Generated 100 training episodes
✅ 5000 tasks successfully created
✅ Episode split: 80/10/10 (train/val/test)
✅ Statistics computed (mean, std, distribution)
```

### 2. Training Konfigürasyonu (`configs/train_trace_ppo.yaml`)

**PPO Hiperparametreleri (Faz 5'ten verify edildi):**
```yaml
learning_rate: 3.0e-4          # Faz 4/5'te optimal
gamma: 0.99                    # Standart RL discount
gae_lambda: 0.95               # Advantage estimation
clip_ratio: 0.2                # PPO clipping
entropy_coef: 0.01             # Exploration bonus
value_coef: 0.5                # Value function weight
max_grad_norm: 0.5             # Gradient clipping

batch_size: 64                 # Experience batch
n_steps: 2048                  # Faz 5'ten artırıldı (stability)
n_epochs: 10                   # Mini-batch epoch'ları
```

**Environment Flagları (Faz 5 Ablation Insights):**
```yaml
use_reward_shaping: true         # ✅ KRITIK (-62.4% impact)
use_partial_offloading: true     # ✅ YÜKSEKetki (-23.4%)
use_semantic_features: true      # ✅ Açıklanabilirlik (nominal etki)
use_confidence_weighting: false  # ❌ Counter-intuitive (+2.2%)
use_battery_awareness: true      # ✅ Yeniden ayarlanmış (+2.6% anomali)
use_mobility_features: true      # ✅ Yeniden ayarlanmış (+5.2% anomali)
use_queue_awareness: false       # ❌ Düşük etki (+1.0%)
```

**Reward Bileşenleri:**
```yaml
reward_weights:
  latency: 1.0          # Temel latency cezası
  energy: 0.5           # Enerji verimliliği bonusu
  priority: 2.0         # Yüksek-öncelik görev tamamlama
  deadline: 3.0         # Deadline tutma bonusu (semantic)
  fairness: 0.5         # Kuyruk adaleti cezası
```

### 3. Orchestrator (`experiments/run_trace_training.py`)

**5 Adımlı Pipeline:**

```
Adım 1: Prepare Traces
├─ Load/generate traces
├─ Preprocess (normalize, filter)
├─ Generate episodes (100 total)
└─ Split (80/10/10)

Adım 2: Train Model
├─ Create OffloadingEnv_v2
├─ Initialize PPOTrainer
├─ Loop: 500 episodes (early stopping)
├─ Eval every 10 episodes
└─ Save best checkpoint

Adım 3: Evaluate
├─ Load best model
├─ Run on validation set (10 episodes)
├─ Compute metrics (success rate, latency, etc.)
└─ Compare vs Faz 5 baseline

Adım 4: Validate Ablations
├─ Test config: full_model
├─ Test config: no_reward_shaping
├─ Test config: no_partial_offloading
└─ Verify Faz 5 bulguları trace data'da

Adım 5: Generate Report
├─ Compile metrics
├─ Create comparison tables
├─ Write Phase_6_Report.md
└─ Summary of findings
```

**Checkpoint Yönetimi:**
- Best model otomatik kaydedilir (success rate arttığında)
- Early stopping: 50 patience, 0.5% improvement threshold
- Logs: `logs/phase6_trace_training/`
- Checkpoints: `models/ppo_v3_trace/`

---

## 🧪 Trace-Driven Training Hipotezleri

### H1: Gerçekçi Task Dağılımı Daha İyi Policy Learning Sağlar
**Mantık:**
- Faz 5 ablation: frozen PPO_v2 policy (statik)
- Faz 6: PPO_v2'yi trace data üzerinde fine-tune edecek
- Trace gerçekçi Poisson arrivals, exponential complexity sağlayacak
- **Beklenti:** Model daha iyi reward calibration → +%5-8 başarı

### H2: Reward Shaping Criticality Trace Data'da da Doğrulanacak
**Mantık:**
- Faz 5'te: disable_reward_shaping = %0 başarı (frozen policy)
- Faz 6'ta: policy retraining ile yeniden test edecek
- Trace data realistic task variations sağlayacak
- **Beklenti:** Reward shaping criticality preserve edilecek

### H3: Partial Offloading Flexibility Battery Constraints ile İnteraksiyon
**Mantık:**
- Faz 5'te: partial offloading = -%23.4 impact
- Battery + partial offloading trade-off'u trace data'da ortaya çıkabilir
- Realistic mobility patterns → distance-based penalties más relevant
- **Beklenti:** Split offloading önemini teyit edecek ama ince tuneability görecek

### H4: Mobility Heuristic Re-tuning Pozitif Anomali Düzeltecek
**Mantık:**
- Faz 5'te: disable_mobility = +%5.2 (!) (anomali)
- Edge proximity heuristic suboptimal bias yaratıyor
- Trace-driven training distance weights'i optimize edebilir
- **Beklenti:** Mobility features soft'ened versiyonu daha iyi sonuç verecek

---

## 📊 Beklenen Sonuçlar

### Performance Gains
```
Faz 5 Baseline:        62.40% ✓
Faz 6 Target Min:      68.00% (+5.60%)
Faz 6 Target Max:      77.00% (+14.60%)
Faz 6 Expected Best:   72.00% (+9.60% orta-nokta)
```

### Metrikler Tablosu
| Metrik | Faz 5 | Faz 6 Expected | Geliştirme |
|--------|--------|---|---|
| Success Rate | %62.40 | %70.00±3% | +%7.60 |
| Avg Latency | TBD | -10% | İnişleme |
| Deadline Miss | %37.60 | %30.00 | İnişleme |
| Energy Efficiency | TBD | +5-8% | Yükselişleme |
| Priority Satisfaction | TBD | >%80 | Yüksek |

### Convergence Eğrisi Beklentisi
```
Episode 0:     %62.4 (Faz 5 baseline, başlangıç)
              ↑ ↑ ↑
              rapid improvement (episode 50-150)
              ↑ ↑
Episode 200:  %70.0 (expected midpoint)
              ↑ plateau (learning saturation)
              ↑ ↑ ↑
Episode 400:  %72.0 (expected max)
              flat (early stopping triggered @ep ~420)
```

---

## 🔗 Çalışma Dosyaları

### Oluşturulan Dosyalar
- ✅ `src/core/trace_processor.py` (520 satır)
  - `TraceProcessor` class: load/generate/preprocess/split
  - `TraceTask`, `TraceEpisode` dataclass'ları
  - Istatistik hesaplama
  
- ✅ `configs/train_trace_ppo.yaml` (60+ satır)
  - Training, environment, data, evaluation configs
  - Phase 5 insights entegre
  
- ✅ `experiments/run_trace_training.py` (650+ satır)
  - `TraceTrainingOrchestrator` class
  - 5 step pipeline
  - Report generation
  
- ✅ `data/traces/{train,val,test}_episodes.json`
  - Sentetik episodes (JSON dumped)
  - Train: 80 episodes, Val: 10 episodes, Test: 10 episodes

- ✅ `phase_reports/Phase_6_Report_DRAFT.md`
  - Template başlama

### Referans Dosyaları (Faz 5)
- `phase_reports/Phase_5_Report.md` (160+ satır, Türkçe)
- `results/tables/ablation_comparison.md` (component impacts)
- `configs/ablation.yaml` (9 ablation flags)
- `experiments/run_ablation_study.py` (orchestration)

---

## ⚙️ Teknik Derinlik: Trace Processing Detayları

### Episode Oluşturma Algoritması
```python
def generate_episodes(traces, tasks_per_episode=50, n_episodes=100):
    """
    1. Tüm trace'leri birleştir (all_tasks)
    2. Her episode için:
       a. Random sample: tasks_per_episode kadar görev
       b. Arrival time'a göre sırala (temporal consistency)
       c. TraceTask objelerine dönüştür
       d. TraceEpisode objesi oluştur (metadata ile)
    3. Return: episodes list
    """
```

### Trace Task Özellikleri Mapping
```python
CSV/JSON trace → TraceTask:

task_id          → task_id (unique identifier)
device_id        → device_id (0-19)
arrival_time     → arrival_time (seconds)
data_size        → data_size (KB)
cpu_cycles       → cpu_cycles (M cycles)
priority         → priority (0-3)
location_(x,y)   → location (tuple)
deadline         → arrival_time + deadline_delta

Computed:
completion_time  = None (outcome)
success          = None (binary)
latency          = completion_time - arrival_time
```

### Statistics Fonksiyonu Çıktısı
```json
{
  "n_episodes": 100,
  "n_tasks_total": 5000,
  "data_size": {
    "mean": 574.5 KB,
    "std": 476.6 KB,
    "min": 100.0 KB,
    "max": 3946.0 KB
  },
  "cpu_cycles": {
    "mean": 587130481,
    "std": 469685542,
    "min": 100005817,
    "max": 2913173783
  },
  "priority_distribution": {
    "0": 1046,    # LOW
    "1": 1431,    # MEDIUM_LOW
    "2": 1609,    # MEDIUM_HIGH
    "3": 914      # HIGH
  },
  "deadline": {
    "mean": 2.85 seconds,
    "std": 1.23 seconds,
    "min": 0.51 seconds,
    "max": 4.99 seconds
  }
}
```

---

## 🚀 Başlatma Komutu (Ready to Run)

```bash
# Faz 6 training başlat
cd d:\task-offloading-study
python experiments/run_trace_training.py

# Beklenen süre: 15-30 dakika (GPU/CPU'ya bağlı)
# Çıktı: logs/phase6_trace_training/Phase_6_Report.md
```

---

## 📝 Başarı Kriteri

Faz 6 **BAŞARILI** sayılmak için:

✅ **Teknik:**
- [ ] Training NaN/Inf olmadan tamamlanır
- [ ] Validation success rate ≥ %68
- [ ] Early stopping ≤ 450 episode'da tetiklenir (convergence)
- [ ] Checkpoint'ler kaydedilir

✅ **Bilimsel:**
- [ ] Faz 5 ablation findings validate edilir (reward shaping criticality)
- [ ] Performance improvement ≥ %5 vs Faz 5 baseline
- [ ] Trace realism effect gözlemlenebilir

✅ **Dokümantasyon:**
- [ ] Phase_6_Report.md oluşturulur (bulgular + analiz)
- [ ] Comparison table: Faz 5 vs Faz 6 metrics
- [ ] Recommendations: Faz 7+ için insights

---

## 🔮 Faz 6 → Faz 7 Geçişi

**Faz 6 sonrası beklenen çıktılar:**
1. Yüksek-fidelity trace dataset preference (synthetic vs real)
2. Battery/Mobility heuristic'ler için tuning önerileri
3. Two-stage training stratejisi için baseline
4. Advanced metrics framework (percentiles, fairness, QoE)

**Faz 7 Hedefi:** %77-85 başarı (graph NN'ler veya multi-agent coordin ile)

---

## 📚 Referanslar

- **Phase 5 Report:** `phase_reports/Phase_5_Report.md` (Faz 5 ablation findings)
- **Trace Processor Source:** `src/core/trace_processor.py` (implementation)
- **Config:** `configs/train_trace_ppo.yaml` (hyperparameters)
- **Orchestrator:** `experiments/run_trace_training.py` (pipeline code)
- **AgentVNE Paper:** "Semantic-aware RL for task offloading" (scientific foundation)

---

**Hazırlanmış:** 31 Mart 2026  
**Yazarı:** GitHub Copilot  
**Durum:** 🟢 ALTYAPI TAM, TRAINING İÇİN HAZIR

*Bu dokümantasyon Faz 6'nın detaylı planını ve altyapısını açıklar. Training başlanmasının `python experiments/run_trace_training.py` komutu ile yapılması beklenir.*
