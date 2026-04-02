# Faz 5 — Sistematik Ablation Study Hazırlık Planı

## 🎯 Faz 5 Hedefi
Semantik bileşenlerin (LLM prior, reward shaping, confidence vb.) task offloading sistemine katkısını sistematik olarak ölçmek.

**Ana Soru:** "PPO_v2'nin %62.67 başarısı nereden geliyor? LLM + Semantic bileşenlerin toplam katkısı ne kadar?"

---

## 📋 Yapılacaklar (Sıralı)

### Adım 1: Ablation Konfigürasyonları Hazırlama ✅
- [x] `configs/synthetic/ablation.yaml` oluşturuldu (9 farklı ablation)
- [ ] Faz 4 sonuçlarıyla karşılaştırma için baseline kaydedilecek

**Hazırlanmış Ablation'lar:**
| # | Ablation | Purpose | Expected Impact |
|---|----------|---------|-----------------|
| 1 | Full Model | Baseline (all features on) | ~62% success |
| 2 | w/o Semantics | LLM contribution | -5 to -15% |
| 3 | w/o Reward Shaping | Semantic reward impact | -3 to -8% |
| 4 | w/o Semantic Prior | Action bias from LLM | -2 to -5% |
| 5 | w/o Confidence | Confidence calibration | -1 to -3% |
| 6 | w/o Partial Offloading | Split offloading value | -10 to -20% |
| 7 | w/o Battery Awareness | Battery preservation | -5 to -10% |
| 8 | w/o Queue Awareness | Queue congestion avoidance | -3 to -8% |
| 9 | w/o Mobility Features | Distance/proximity impact | -2 to -5% |

### Adım 2: Ablation Runner Script Yazma
**Dosya:** `experiments/synthetic/run_ablation_study.py`

**İçerik:**
```python
# Pseudo-code
def run_ablation_study():
    ablation_config = load_yaml('configs/synthetic/ablation.yaml')
    
    for ablation_name, ablation_spec in ablation_config['ablation_studies'].items():
        print(f"[ABLATION] Running: {ablation_name}")
        
        # 1. Modeli konfigüre et (feature'ları aç/kapat)
        env = OffloadingEnv(
            disable_semantics=not ablation_spec['semantics'],
            disable_reward_shaping=not ablation_spec['reward_shaping'],
            # ... diğer flags
        )
        
        # 2. PPO_v2 modelini yükle ve test et
        policy = PPO.load('models/ppo/single_run_synthetic/ppo_offloading_agent_v2.zip', env=env)
        
        # 3. 10 episode üzerinde değerlendir
        results = evaluate_policy(env, policy, num_episodes=10, run_name=ablation_name)
        
        # 4. CSV'ye kaydet
        log_to_csv(results, ablation_name)
    
    # 5. Karşılaştırma tablosu oluştur
    generate_ablation_comparison_table()
    generate_ablation_impact_charts()
```

### Adım 3: RL Environment Modifikasyonları
**Dosya:** `src/env/rl_env.py`

Gerekli değişiklikler:
- [ ] Constructor'a ablation flags ekle (disable_semantics, disable_reward_shaping, etc.)
- [ ] `_get_observation()` içinde disable_semantics kontrolü
- [ ] `_compute_reward()` içinde disable_reward_shaping kontrolü
- [ ] `_compute_reward()` içinde confidence weighting disable etme
- [ ] State builder'da feature'ları dinamik olarak açma/kapama

**Örnek:**
```python
class OffloadingEnv(gym.Env):
    def __init__(self, ..., disable_semantics=False, disable_reward_shaping=False, ...):
        self.disable_semantics = disable_semantics
        self.disable_reward_shaping = disable_reward_shaping
        ...
    
    def _get_observation(self):
        obs = self.state_builder.build_state(...)
        if self.disable_semantics:
            obs = obs[:5]  # Keep only physical features, drop semantic prior
        return obs
    
    def _compute_reward(self, action, task, offload_target):
        reward = base_reward(task, offload_target)
        
        if not self.disable_reward_shaping:
            reward += self.semantic_analyzer.compute_bonus(task, action)
        
        return reward
```

### Adım 4: Metrikleri Genişletme
**Dosya:** `src/core/metrics.py` (yeni)

Gerekli metrikler:
- [ ] `avg_latency` — Ortalama görev latency'si
- [ ] `p95_latency` — 95. persentil latency
- [ ] `deadline_miss_ratio` — Deadline'ı kaçanların oranı
- [ ] `avg_energy` — Ortalama enerji tüketimi
- [ ] `fairness_score` — Cihazlar arasında fairness (Jain index)
- [ ] `jitter` — Latency varyasyonu (std dev)
- [ ] `qoe` — Quality of Experience skoru

**Pseudo-code:**
```python
def compute_metrics(episode_logs):
    metrics = {
        'avg_latency': np.mean(episode_logs['latencies']),
        'p95_latency': np.percentile(episode_logs['latencies'], 95),
        'deadline_miss_ratio': sum(1 for l in episode_logs['latencies'] if l > deadline) / len(...),
        'avg_energy': np.mean(episode_logs['energy_consumed']),
        'fairness_score': jain_fairness_index(episode_logs['device_energy']),
        'jitter': np.std(episode_logs['latencies']),
        'qoe': calculate_qoe(episode_logs),
    }
    return metrics
```

### Adım 5: Ablation Runner Yazma
**Dosya:** `experiments/synthetic/run_ablation_study.py`

### Adım 6: Sonuçları Analiz Etme
**Çıktılar:**
- [ ] `results/tables/offloading_experiment_report.md` — Kanonik markdown rapor
- [ ] `results/figures/synthetic_ablation_<algorithm>_<scope>_success_rate.png` — Bar/line chart'lar
- [ ] `results/raw/ablation_experiments.csv` — Detaylı logs

### Adım 7: Faz 5 Raporunu Yazma
**Dosya:** `phase_reports/Phase_5_Report.md`

**İçerik:**
- Ablation tasarımı ve metodoloji
- Her ablation'ın sonuçları
- Katkı dağılımı (% olarak)
- İstatistiksel anlamlılık testleri
- Yayın-ready bulguları

---

## 🔧 Teknik Detaylar & Kararlar

### Karar 1: Ablation Sırasında PPO_v2 Model Yeniden Eğitilecek mi?
**Seçenek A:** Her ablation için yeni modeli train et (gerçekçi ama zaman alıcı)
**Seçenek B:** Tek bir eğitilmiş model kullanıp özellik'leri devre dışı bırak (hızlı)

**Tavsiye:** **Seçenek B** (hızlı inceleme için), sonra önemli ablation'ları Seçenek A ile validate et

### Karar 2: İstatistiksel Testler Yapılacak mı?
**Evet!** Her ablation için:
- [ ] Mean ± 95% CI
- [ ] Paired t-test vs. Full Model
- [ ] p-value raporlanacak

### Karar 3: Diğer Baseline'lar ile Ablation Karşılaştırması?
**Evet!** GA vs PPO_v2 ablation'lar:
- GA hala %50'de kalır (statik)
- PPO_v2 ablation'ları düşüyor
- Bu PPO'nun adaptivitesini gösterir

---

## 📚 Bağlantılar

- **Faz 5 Tanımı:** [TODO_ANTIGRAVITY_TASK_OFFLOADING_UPGRADE.md#Faz 5](file:///...)
- **Ablation Config:** `configs/synthetic/ablation.yaml`
- **Önceki Sonuçlar:** [Phase_4_Report.md](phase_reports/Phase_4_Report.md)

---

## 🎓 Yayın Katkısı

### "Tez'de Nereye Gider?"
**Bölüm 6 (Ablation Analysis):**
- Table: "Component Ablation Results" (9 satır × 8 sütun)
- Figure: "Impact of Semantic Components on Success Rate"
- Text: "Our ablation reveals that semantic prior accounts for X% of performance gain..."

### "Tek Başına Paper Olabilir mi?"
**Potansiyel Paper:** "Ablation Study of LLM-Guided Task Offloading"
- Venue: IEEE IoT Journal, ACM Transactions on Mobile Computing
- Title: "Understanding the Role of Semantic Priors in RL-based Mobile Edge Computing"

---

## ✅ Faz 5 Tamamlama Kriterleri

- [ ] 9 ablation çalıştı ve sonuçlar CSV'ye kaydedildi
- [ ] Her ablation için Mean ± 95% CI raporlandı
- [ ] Ablation'lar arasında p-value (paired t-test) hesaplandı
- [ ] Markdown karşılaştırma tablosu oluşturuldu
- [ ] Impact chart'ları (bar/line) oluşturuldu
- [ ] Phase_5_Report.md yazıldı ve tamamlandı
- [ ] **Bulgular:** Hangi bileşenler en çok katkı sağlıyor? (Expected: Semantics > Reward Shaping > Partial Offloading > Confidence)
- [ ] Sonuçlar AgentVNE paperi ile karşılaştırıldı

---

## 📝 Notlar

### İlgili Alanlara Tekrar Bakılması Gereken Kısımlar
1. **State Builder:** Semantics disable edildiğinde state boyutu değişecek (11 → 5). Bu OK mi?
   - Cevap: OK! PPO'nun flexibility'si bu kadar değişimi handle eder

2. **Env.reset() & Seeding:** Tüm ablation'lar aynı seed ile çalışmalı (reproducibility)
   - Cevap: Already done (seed=42 in configs)

3. **GPU Memory:** PPO model loading × 9 ablation = Bellek kullanma?
   - Cevap: Tek bir model load et, inference sırasında feature'ları mask et

---

## 🚀 Faz 5 Başlamaya Hazır!

Tüm hazırlıklar tamamlandı. Sonraki adım:
1. `src/env/rl_env.py`'de ablation flag'larını ekle
2. `experiments/synthetic/run_ablation_study.py` yaz
3. Çalıştır: `python experiments/synthetic/run_ablation_study.py`
4. Rapor yaz: `phase_reports/Phase_5_Report.md`
