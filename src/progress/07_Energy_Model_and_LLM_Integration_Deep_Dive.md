# Energy Model, DVFS, Shannon Kanali ve LLM Integration - Detaylı Açıklama

## 1️⃣ Enerji Tüketimi: Dinamik Mi Sabit Mi?

### **CEVAP: DINAMIK (Görev özelliklerine bağlı)**

Enerji tüketimi **sabit bir sayı DEĞİL**, görevin özellikleri ve network durumuna göre değişiyor.

#### **Enerji Hesaplanması**

```python
# LOCAL İŞLEM (Action 0)
local_comp_energy_full = KAPPA * (DEFAULT_CPU_FREQ ** 2) * task.cpu_cycles * ENERGY_SCALE_FACTOR
self.battery -= local_comp_energy_full
```

**Formül:**

```
E_local = κ × f² × C_cpu

Burada:
- κ (KAPPA) = 0.5 (power coefficient)
- f = frequency (1 GHz)
- C_cpu = görevin CPU cycle'ı (değişken!)
```

**Örnek:**

```
Task 1: cpu_cycles = 5×10^8 → E = 0.5 × 1² × 5×10^8 × 50 = 12.5 MJ (YAŞAR)
Task 2: cpu_cycles = 1×10^10 → E = 0.5 × 1² × 1×10^10 × 50 = 250 MJ (ÇABUCAK ÖLDÜRÜR!)
```

---

### **PARTIAL Offloading Enerji**

```python
# Local kısmı
local_cycles = (1 - ratio) * task.cpu_cycles
local_energy = KAPPA * (DEFAULT_CPU_FREQ**2) * local_cycles * ENERGY_SCALE_FACTOR

# İletim kısmı (Shannon modeli ile hesaplanan datarate kullanarak)
edge_bits = ratio * task.size_bits
tx_time = edge_bits / datarate  # ← datarate DINAMIK (Shannon'dan geliyor!)
tx_energy = TRANSMISSION_POWER * tx_time * ENERGY_SCALE_FACTOR

# Toplam
total_energy = local_energy + tx_energy
self.battery -= total_energy
```

**Bu demek ki:**

- Görev büyük → daha fazla enerji
- Network hızlı (Shannon'dan yüksek datarate) → transmission hızlı → az TX enerji
- Network yavaş (Shannon'dan düşük datarate) → transmission yavaş → çok TX enerji

---

## 2️⃣ Shannon Kanali Modeli - Network Hızı

### **Shannon-Hartley Kapasitesi Formülü**

```python
def calculate_datarate(self, device, edge_server):
    # Path Loss: h = d^(-alpha)
    d = distance(device, edge_server)
    h = d ** (-PATH_LOSS_EXPONENT)  # (-2)

    # SINR hesaplaması
    sinr = (TRANSMISSION_POWER * h) / (NOISE_POWER + interference)

    # Shannon Kapasitesi
    datarate = BANDWIDTH * log2(1 + sinr)
    # R = 20 MHz × log2(1 + SINR)
```

**Ne demek:**

- **Yakındaki Edge (d=100m)**: h = 100^(-2) = 0.0001 → yüksek SINR → hızlı datarate → az transmission süresi
- **Uzak Edge (d=500m)**: h = 500^(-2) = 0.000004 → düşük SINR → yavaş datarate → uzun transmission süresi
- **Noise/Interference**: Arttıkça SINR düşer, datarate düşer, enerji artar

**Enerji açısından:**

```
tx_energy = TRANSMISSION_POWER * (task.size_bits / datarate) * ENERGY_SCALE_FACTOR
                                                 ↑
                            Shannon'dan gelen dinamik datarate!
```

---

## 3️⃣ DVFS (Dynamic Voltage and Frequency Scaling) Modeli

### **Edge Server'da Frekans Ayarlaması**

```python
# Edge Server processing_task() fonksiyonunda
def process_task(self, task):
    # DVFS: Load'a göre frekans değişir
    if self.current_load > 2:
        self.current_freq = self.max_freq  # Full speed
    else:
        self.current_freq = self.max_freq * 0.7  # 70% speed

    # İşlem süresi frekansa bağlı
    processing_time = task.cpu_cycles / self.current_freq

    # Enerji (Edge'de), KUBIK (f³) bağımlı!
    energy = KAPPA * (self.current_freq ** 3) * processing_time
```

**Ne demek:**

- **Düşük load**: 0.7×max_freq → daha yavaş ama çok az enerji (f³ ile azalır!)
- **Yüksek load**: 1.0×max_freq → daha hızlı ama çok fazla enerji (f³ ile artar!)

**Örnek:**

```
f₁ = 0.7 GHz: E = κ × (0.7)³ × t = κ × 0.343 × t
f₂ = 1.0 GHz: E = κ × (1.0)³ × t = κ × 1.0 × t
Fark = 3x fark! (f³ nedeniyle)
```

**Enerji Tüketimi Dinamiktir:**

- Edge kuyruk uzun → full frequency → 3x enerji
- Edge kuyruk kısa → lower frequency → 1x enerji

---

## 4️⃣ Özet: Enerji Tüketimi Nasıl Hesaplanıyor?

### **Sistem Şeması**

```
┌─────────────────────────────────────────────────────────────┐
│                    TASK ARRIVES                              │
└───────────────────────┬─────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
     LOCAL (0)        PARTIAL        FULL EDGE/CLOUD
     Action 0      (1-4)            (5)
        │               │               │
        └───────────────┼───────────────┘
                        │

┌───────────────────────────────────────────────────────────────┐
│                ENERGY CALCULATION                              │
├───────────────────────────────────────────────────────────────┤
│                                                                 │
│ LOCAL Processing:                                               │
│ ├─ E_cpu = κ × f² × C_cpu × SCALE                              │
│ │          (görev işlem döngüsüne bağlı)                       │
│ └─ NO transmission energy                                       │
│                                                                 │
│ PARTIAL Offloading (ratio = 0.25/0.5/0.75):                    │
│ ├─ E_local = κ × f² × (1-ratio) × C_cpu × SCALE                │
│ ├─ E_tx = P_tx × (ratio × size / Shannon_datarate) × SCALE     │
│ │         (Shannon kapasitesine bağlı, dinamik!)               │
│ └─ E_edge = κ × (f_edge)³ × processing_time                    │
│              (DVFS'ye bağlı, load dinamik!)                    │
│                                                                 │
│ FULL EDGE (ratio = 1.0):                                        │
│ ├─ E_tx = P_tx × (size / Shannon_datarate) × SCALE             │
│ └─ E_edge = κ × (f_edge)³ × processing_time                    │
│              (tüm işlem edge'de, DVFS etki eder)               │
│                                                                 │
│ CLOUD (Action 5):                                               │
│ ├─ E_tx = P_tx × (size / Shannon_datarate) × SCALE × LTE       │
│ │         (internet gateway'e kadar, uzun mesafe!)             │
│ └─ E_edge = 0 (Cloud'un enerji tüketimi device'da sayılmaz)    │
│                                                                 │
└───────────────────────────────────────────────────────────────┘
```

### **Dinamik Faktörler**

| Faktör                     | Tip                     | Etki                    |
| -------------------------- | ----------------------- | ----------------------- |
| **task.cpu_cycles**        | Fiziksel (görev tipi)   | E_local doğru orantılı  |
| **task.size_bits**         | Fiziksel (görev tipi)   | E_tx doğru orantılı     |
| **distance(device, edge)** | Fiziksel (konum)        | Shannon datarate → E_tx |
| **edge.current_load**      | Dinamik (sistem durumu) | DVFS frekansı → E_edge  |
| **Shannon SINR**           | Fiziksel (fizik kanali) | datarate → E_tx         |

---

## 5️⃣ LLM Integration - Model Training Sorusu

### **Sorunuz: "RL modeline LLM çıktısını verip tekrar mı eğiteceksin?"**

**CEVAP: EVET, doğru anlıyorsunuz!**

### **Şu Anki Durum (Before)**

```
PPO Model:
├─ Input: 5 feature
│  ├─ SNR (normalized datarate)
│  ├─ size_norm (görev boyutu)
│  ├─ cpu_norm (CPU ihtiyacı)
│  ├─ batt_norm (batarya %)
│  └─ load_norm (edge kuyruk)
│
└─ Output: Action (0-5)
```

**Problem:** PPO, LLM'nin "local processing önerildiğini" bilmiyor!

### **Yeni Durum (After)**

```
PPO Model (RETRAINING GEREKLI):
├─ Input: 6 feature ← ⚠️ DEĞİŞTİ!
│  ├─ SNR (normalized datarate)
│  ├─ size_norm (görev boyutu)
│  ├─ cpu_norm (CPU ihtiyacı)
│  ├─ batt_norm (batarya %)
│  ├─ load_norm (edge kuyruk)
│  └─ llm_local_score ← YENI!
│                        (1.0 = local öneriliyor)
│                        (0.1 = edge öneriliyor)
│                        (-0.5 = cloud öneriliyor)
│
└─ Output: Action (0-5)
```

### **Training Süreci**

```python
# RL_ENV initialization
class RLEnvironment:
    def _get_obs(self):
        # ... 5 feature ...
        llm_rec = task.semantic_analysis['recommended_target']
        llm_local_score = 1.0 if llm_rec == 'local' else ...

        # 6 feature return et
        return np.array([..., llm_local_score], dtype=np.float32)

# PPO Training başladığında:
PPO_MODEL = PPO('MlpPolicy', env)
PPO_MODEL.learn(total_timesteps=100000)  # ← Bu 6 feature'ı öğrenecek
```

### **Neden Retraining Gerekli?**

1. **Neural Network Input Layer**: 5 neuron için train edilmiş
2. **Yeni Input**: 6 neuron
3. **Ağ mimari uyuşmuyor** → Random weights atanacak 6. neuron'a
4. **Eğitim gerekli** → Model yeni feature'ı (LLM input) faydalı olduğunu öğrenecek

---

## 6️⃣ Tutarlılık Nasıl Sağlanacak?

### **LLM ↔ PPO Alignment Mekanizması**

```python
# training esnasında:
# PPO görecek ki:
#
# "LLM = local önerdiğinde" (llm_local_score = 1.0)
# ve "batarya düşük" (batt_norm = 0.2)
# → Action 0 (Local) seçilirse REWARD = +15!
#
# Tersine:
# "LLM = local önerdiğinde" (llm_local_score = 1.0)
# ama "Action 5 (Cloud) seçilirse" → REWARD = -30!
```

**Sonuç:** PPO öğrenecek ki, LLM doğru tavsiye veriyor!

### **Üç Adımlı Plan**

```
1. TRAINING PHASE
   - 6-feature model eğit (100k+ timestep)
   - LLM score'ları reward'larla bağla
   - Tutarlılığı maximize et

2. TESTING PHASE (Şu anda yapılıyor)
   - Simülasyonda gözlemle:
     * LLM success rate: %?
     * LLM ↔ PPO Alignment: %?
     * Ortalama enerji tasarrufu: %?

3. PRODUCTION
   - PPO + LLM hybrid sistem canlıya al
   - Kararları her ikisinden de kontrol et
```

---

## 7️⃣ Pratik Örnek: Bir Görev Yaşam Döngüsü

### **Senaryo: HIGH_DATA Task (50MB, 10GHz CPU)**

```
T=2.5s: TASK OLUŞTURULDU
├─ size_bits = 50e6 bits
├─ cpu_cycles = 1e10
├─ task_type = HIGH_DATA
└─ deadline = 5.0s

T=2.55s: LLM ANALYSIS
├─ bandwidth_need = 0.63 (50MB vs 10MB normalization)
├─ complexity = 1.0 (çok yüksek CPU)
├─ urgency = 0.17 (5s deadline yeterince uzun)
└─ recommended_target = "CLOUD" (karmaşık, büyük → cloud'a gönder)

T=2.56s: RL ENVIRONMENT OBSERVATION
├─ snr_norm = 0.4 (orta hızda network)
├─ size_norm = 1.0 (çok büyük)
├─ cpu_norm = 1.0 (çok yüksek CPU)
├─ batt_norm = 0.75 (batarya iyi)
├─ load_norm = 0.5 (edge orta yoğun)
└─ llm_local_score = -0.5 ← LLM "CLOUD" dedi!

T=2.57s: PPO DECISION
├─ Model "yüksek CPU + yüksek size + batarya iyi + llm=-0.5"
│  gördü
├─ "Action 5 (CLOUD)" seçti
└─ reward = -(50*0.01) - (energy*2) + 5 (cloud penalty azaldı çünkü LLM de cloud dedi!)

T=2.58s: EXECUTION
├─ tx_energy = P_tx × (50e6 bits / Shannon_datarate) × SCALE
│            = 1.0 × (50e6 / 5e7) × 50  ← Datarate = 50 Mbps (Shannon'dan)
│            = 1.0 × 1.0 × 50 = 50 J
├─ cloud_processing = negligible (cloud'da enerji harcanmıyor)
└─ battery -= 50 J

T=2.75s: COMPLETION
├─ completion_time = 2.75s
├─ latency = 2.75 - 2.5 = 0.25s
├─ deadline met? = YES (0.25 < 5.0)
└─ reward += 5.0 (deadline bonus!)
```

---

## 8️⃣ Cevaplarınızın Özeti

### **Soru 1: "RL modeline LLM çıktısını vereceksin ve tutarlı olması için onu tekrar mı eğiteceksin?"**

**Cevap:** ✅ EVET

- Observation 5→6 feature'a yükseltti
- PPO bunu öğrenmek için yeniden eğitilmeli
- Training sırasında "LLM input" ile "reward" arasında korelasyon kurulacak

### **Soru 2: "Device enerjisi görevin ne kadar enerji harcadığına bağlı olarak değişmiyor mu?"**

**Cevap:** ✅ EVET, DEĞIŞIR - DINAMIK

- LOCAL: `E = κ × f² × C_cpu` (görev CPU'sine doğru orantılı)
- PARTIAL: `E = κ × f² × (1-r) × C_cpu + P_tx × (r × size / Shannon_datarate)`
- EDGE: `E = κ × (f_edge)³ × t` (DVFS'ye + kuyruk durumuna bağlı)

### **Soru 3: "DVFS modeli, Shannon modeli kullanılmıyor mu?"**

**Cevap:** ✅ KULLANILIYOR

- **DVFS**: Edge Server'da `process_task()` → frequency adjustment
- **Shannon**: `calculate_datarate()` → transmission hızı
- **Enerji**: Her ikisi de dinamik enerji hesaplamalarına etki eder

### **Soru 4: "Enerji sabit mi, dinamik mi?"**

**Cevap:** ✅ TAMAMEN DİNAMİK

```
E = f(task_properties, network_state, server_load, Shannon_SINR, DVFS)
```

---

## 📊 Önerilen Sonraki Adımlar

1. **Trained Model'i Sil** (observation değiştiği için)

   ```bash
   rm src/models/ppo_offloading_agent.zip
   ```

2. **Yeni Model Eğit** (6-feature observation ile)

   ```bash
   python src/train_agent.py  # ← 100k+ timestep
   ```

3. **Simülasyonu Çalıştır** ve metrikleri gözlemle:
   - LLM Success Rate: % ?
   - LLM ↔ PPO Alignment: % ?
   - LOCAL offloading: % ? (sıfırdan yüksek mi?)
   - Battery drain: daha yavaş mı?
