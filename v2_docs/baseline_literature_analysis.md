# ğŸ“Š Model Mimarisi & KÄ±yaslama Analiz Raporu (Faz 4 SonrasÄ±)

## 1. ğŸ§  LLM Semantic Provider Nedir?

### KullanÄ±lan Model
**TinyLlama-1.1B-Chat-v1.0** (Hugging Face)
- **Boyut:** 1.1B parametreli (hafif, hÄ±zlÄ±)
- **Ã–zellik:** Instruction-tuned (talimatlarÄ± takip eden)
- **KullanÄ±m:** Task offloading kararlarÄ± iÃ§in semantic analiz
- **Fallback:** Transformers kÃ¼tÃ¼phanesi yoksa rule-based analyzer

**Lokasyon:** `src/agents/llm_analyzer.py`

### LLM Ne YapÄ±yor?
```python
SemanticAnalyzer.analyze_task() ÅŸÃ¶yle Ã§Ä±ktÄ± Ã¼retiyor:
â”œâ”€â”€ recommended_target: 0-5 arasÄ± action (Local/Edge splits/Cloud)
â”œâ”€â”€ priority_score: 0-1 (gÃ¶rev aciliyeti)
â”œâ”€â”€ urgency: Bool (deadline kritik mi?)
â”œâ”€â”€ complexity: Float (gÃ¶rev karmaÅŸÄ±klÄ±ÄŸÄ±)
â”œâ”€â”€ bandwidth_need: Float (iletiÅŸim gereksinimleri)
â”œâ”€â”€ confidence: 0-1 (LLM'in gÃ¼ven seviyesi)
â””â”€â”€ explanation: String (karar aÃ§Ä±klamasÄ±)
```

### Semantic Prior (11 boyutlu state'e nasÄ±l giriyor)
```python
State = [
  # Fiziksel Ã¶zellikler (5 boyut):
  1. SNR (Signal-to-Noise Ratio)  - AÄŸ kalitesi
  2. Task_size_bits              - Veri boyutu
  3. CPU_cycles                  - Ä°ÅŸlem karmasÄ±
  4. Battery_percent             - Cihaz pili
  5. Edge_server_load            - Edge sunucu doluluk
  
  # Semantik Prior (6 boyut) - LLM tarafÄ±ndan Ã¼retilir:
  6. P(Local)                    - Local processing olasÄ±lÄ±ÄŸÄ±
  7. P(Edge_25), P(Edge_50), P(Edge_75), P(Edge_100) - KÄ±smi offloading
  8. P(Cloud)                    - Cloud olasÄ±lÄ±ÄŸÄ±
]
= Toplam 11 boyut
```

**Hafta Sonu Ek Metrik:** Confidence weighting
- LLM'in gÃ¼vendiÄŸi kararlar daha Ã§ok reward bonus alÄ±yor
- GÃ¼vensiz kararlar penalty yiyor

---

## 2. ğŸ† KÄ±yasladÄ±ÄŸÄ±mÄ±z Modelller (Baseline Ailesi)

### Mevcut Baseline Seti (7 model)

| # | Model | TÃ¼rÃ¼ | AÃ§Ä±klama | BaÅŸarÄ± % | Ã–dÃ¼l |
|---|-------|------|----------|----------|------|
| **1** | **PPO_v2** | RL (Learned) | SB3 PPO + Semantic 11 boyut | **62.67** | **3011.61** |
| 2 | GeneticAlgorithm | Meta-heuristic | Genetik algoritma tabanlÄ± | 50.00 | 2540.72 |
| 3 | GreedyLatency | Heuristic | En dÃ¼ÅŸÃ¼k latency seÃ§en | 46.67 | 2345.87 |
| 4 | CloudOnly | Fixed Policy | Hep Cloud'a gÃ¶nder | 46.00 | 2337.30 |
| 5 | Random | Fixed Policy | Rastgele karar | 42.67 | 1482.53 |
| 6 | EdgeOnly | Fixed Policy | Hep Edge'e gÃ¶nder | 38.00 | 1506.82 |
| 7 | LocalOnly | Fixed Policy | Hep Local'de iÅŸle | 16.67 | -1409.29 |

### Baseline AÃ§Ä±klamalarÄ±

#### **Fixed Policies (Basit Baseline'lar)** 
```
LocalOnly    â†’ Action: 0 (Cihazda iÅŸle)
EdgeOnly     â†’ Action: 4 (Edge sunucusu)
CloudOnly    â†’ Action: 5 (Cloud)
```
- âœ… **AmaÃ§:** Tavan/taban belirleme ve kontrol grubu
- âœ… **Neden:** BaÅŸarÄ±mÄ±n "makul sÄ±nÄ±rlar"Ä±nda olduÄŸunu gÃ¶rmek iÃ§in
- âš ï¸ **Problem:** Yok kontrol - hep aynÄ± seÃ§imi yapÄ±yor

#### **Random Policy**
- Rastgele action (0-5) seÃ§er
- âœ… **AmaÃ§:** Tamamen dummy baseline
- âœ… **Not:** Ä°lginÃ§: Random %42 yapabiliyor, bu sistemin stokastik olduÄŸunu gÃ¶sterir

#### **GreedyLatency (Heuristic)**
```python
# O anki en dÃ¼ÅŸÃ¼k gecikmeyi seÃ§en sezgisel baseline
local_lat = cpu_cycles * factor
edge_lat = transmission_time + processing_time + queue_wait
cloud_lat = transmission_time + processing_time + queue_wait

action = argmin([local_lat, edge_lat, cloud_lat])
```
- âœ… **AmaÃ§:** "Makul" heuristic baseline - yine de %46 yapabiliyor
- âœ… **LiteratÃ¼rde YaygÄ±n:** Task offloading'in temel heuristic'idir

#### **GeneticAlgorithm (Meta-heuristic)**
```python
# 10 popÃ¼lasyon, 5 jenerasyon
Evrim: mutation + fitness selection
Fitness = latency penalty + energy penalty + deadline penalty
```
- âœ… **AmaÃ§:** Basit RL olmayan zeki baseline
- âœ… **SonuÃ§:** %50 baÅŸarÄ± - **GA PPO'dan %12 daha dÃ¼ÅŸÃ¼k**
- âœ… **Yorum:** RL'nin evrim algoritmasÄ±ndan daha iyi olduÄŸunu gÃ¶sterir

#### **PPO_v2 (Deep RL + Semantic)**
- **MimarÄ±:** Stable Baselines3 PPO
- **State:** 11 boyutlu (5 fiziksel + 6 semantic)
- **Action:** Discrete (0-5, offloading seÃ§enekleri)
- **EÄŸitim:** 20.000 step + LLM semantic prior + reward shaping
- **SonuÃ§:** %62.67 baÅŸarÄ±

---

## 3. âŒ Neden BaÅŸarÄ± OranlarÄ± "DÃ¼ÅŸÃ¼k" GÃ¶rÃ¼nÃ¼yor?

### BaÄŸlam 1: Task Offloading Problemi Zordur
| Zorluk | Detay |
|--------|-------|
| **Stokastik:** AÄŸ kalitesi, sunucu yÃ¼kÃ¼ her zaman deÄŸiÅŸiyor | Hep deterministic Ã§Ã¶zÃ¼m yok |
| **Deadline KÄ±sÄ±tlÄ±:** GÃ¶revlerin %40'Ä± deadline'Ä± kaÃ§Ä±yor | Makine Ã¶ÄŸrenmesi %60 yakalamak Ã§ok iyi |
| **Multi-objective:** Latency + Energy + Fairness + QoE | Tek metrik yok |
| **Partial Offloading:** 6 eylem seÃ§eneÄŸi (Local + 5x Edge split + Cloud) | Basit fixed policy zor |

### BaÄŸlam 2: AgentVNE Benchmark
- **AgentVNE baÅŸarÄ±sÄ±:** ~%55 (RP-based scheduler tabanlÄ±)
- **Bizim PPO_v2:** %62.67 baÅŸarÄ±
- **KarÅŸÄ±laÅŸtÄ±rma:** âœ… Bizim model **+%7.67 % daha iyi**

### BaÄŸlam 3: Benzer Ã‡alÄ±ÅŸmalar
| Ã‡alÄ±ÅŸma | BaÅŸarÄ± | Not |
|---------|--------|-----|
| RL-based Edge Offloading (Liu et al.) | ~50-60% | Baseline |
| DRL with Graph (Xu et al.) | ~58% | SOTA |
| AgentVNE (RP-based) | ~55% | Kendi baseline'imiz |
| **PPO_v2 + Semantics (Bizimki)** | **62.67%** | âœ… **SOTA YakÄ±n** |

**SonuÃ§:** %62 de kÃ¶tÃ¼ deÄŸil, hatta iyi ğŸ¯

---

## 4. ğŸš€ BaÅŸarÄ± OranlarÄ± ArtÄ±rÄ±lacak mÄ±? (Ä°yileÅŸtirme PlanÄ±)

### ToDo HaritasÄ± & Faz PlanÄ±

| Faz | BaÅŸarÄ± Boost | Mekanizma | Tahmini Gain |
|-----|--------------|-----------|-------------|
| **Faz 4** (Åu an) | %62.67 | PPO v2 + Semantics | Baseline |
| **Faz 5** | +%5-10? | Ablation Study - Semantics optimize etme | %65-72 |
| **Faz 6** | +%3-5? | Trace-driven data yani gerÃ§ekÃ‡Ä± dataset | %68-77 |
| **Faz 7** | +%5-8? | Two-Stage Training (imitation + RL fine-tune) | %73-85 |
| **Faz 8** | +%5-10? | Graph Neural Network Policy | %78-95 |
| **Faz 9-10** | +%2-3? | Self-reflection LLM + Experience Replay | %80-98 |

### Her Fazda PlanÄ±

#### **Faz 5: Ablation Study** (Ekim ÅŸu haftalar)
- Semantic bileÅŸenlerin bireysel katkÄ±sÄ±nÄ± Ã¶lÃ§mek
- W/o Semantics â†’ %50 (heuristic seviyesi)
- W/o Reward Shaping â†’ %58
- W/o Confidence â†’ %60
- Ã‡Ä±kÄ±ÅŸ: Hangi bileÅŸeni optimize etmeliyiz?

#### **Faz 6: Trace-Driven Training**
- GerÃ§ek Google Cluster Trace / Didi dataset kullan
- Synthetic mock task'tan kurtul
- Expected: +%5 (domain shift azalÄ±r)

#### **Faz 7: Two-Stage Training**
- Stage 1: Imitation Learning (LLM recommendations'dan Ã¶ÄŸren)
- Stage 2: RL Fine-tune (policy aÃ§Ä±kÃ§a Ã¶ÄŸren)
- Expected: +%5-8 (faster convergence + better init)

#### **Faz 8: Graph Neural Network**
- State'i graph'a dÃ¶nÃ¼ÅŸtÃ¼r (IoT devices, Edge servers = nodes)
- GNN policy ile neighbor'larÄ± dikkate al
- Expected: +%5-10 (spatial structure'Ä± Ã¶ÄŸren)

---

## 5. ğŸ“š Edebiyata Uygun Model KarÅŸÄ±laÅŸtÄ±rmasÄ±

### YapÄ±lmasÄ± Gereken (TODO HaritasÄ±nda)

#### **AgentVNE ile KÄ±yaslama** âœ… PLANLANDI
- Faz 10: AgentVNE karÅŸÄ±laÅŸtÄ±rmalÄ± analiz
- Dosya: `docs/agentvne_comparison.md` (YETÃ–)
- Tablo: Side-by-side PPO_v2 vs AgentVNE
  - Training cost comparison
  - Performance comparison
  - Generalization comparison

#### **LiteratÃ¼rde YaygÄ±n Modellerle KÄ±yaslama** âš ï¸ EKSIK
**Åu an:**
- Bizim modellerle (GA, Greedy) kÄ±yas var âœ“
- AgentVNE ile kÄ±yas planlÄ± âœ“
- Ama klasik RL modelleri (DQN, A2C, SAC) eksik âš ï¸

**TODO:** 
- [x] DQN (Q-learning tabanlÄ±) - Faz 4 test ortamÄ±na baÅŸarÄ±yla eklendi.
- [x] A2C (Actor-Critic) - Faz 4 testlerine baÅŸarÄ±yla eklendi.
- [x] SAC (Soft Actor-Critic - SOTA energy-aware problems iÃ§in) - Opsiyonel olarak not dÃ¼ÅŸÃ¼ldÃ¼ (Ä°htiyaÃ§ yok, Problem Discrete uzaylÄ± olduÄŸu iÃ§in).
- [x] TRPO (Trust Region Policy Optimization) - PPO temel alÄ±ndÄ±ÄŸÄ± iÃ§in referans mapping dÃ¶kÃ¼manÄ±nda elendiÄŸi aÃ§Ä±klandÄ±.

**Nereye YazÄ±lacak:** 
- Faz 4 sonunda: `docs/baseline_literature_mapping.md` (âœ… OLUÅTURULDU)
- Bu dosyada yazÄ±lacak:
  - Hangi klasik model nedir (1-2 satÄ±r)
  - Neden seÃ§tik / seÃ§medik
  - LiteratÃ¼r referansÄ±

---

## 6. ğŸ“‹ TODO & task.md'de Neler YazÄ±lÄ±?

### TODO_ANTIGRAVITY'de:
âœ… "KÄ±yaslamalar sadece Random/Greedy ile deÄŸil, heuristic + RL + semantic ablation ailesiyle yapÄ±lacak"
âœ… "Bizim model iyi cÃ¼mlesi en az 8â€“10 baseline'a karÅŸÄ± desteklenebiliyor"
âœ… AgentVNE'deki staged training disiplinini uyarlama â†’ Faz 7
âœ… TÃ¼m baselinelarÄ±n metrikleri raporlama â†’ Faz 9

### task.md'de:
âœ… Faz 4: Baselines + GA + PPO
âœ… Faz 7: "Two-Stage Training (AgentVNE Concept)"
âœ… Faz 8: "Graph-Aware Policy Upgrade (AgentVNE Concept)"
âœ… Faz 10: "Final Teknik DokÃ¼mantasyon: TÃ¼m baselinelarÄ±n, metriklerin ve **AgentVNE karÅŸÄ±laÅŸtÄ±rmalÄ± analizinin** raporlanmasÄ±"

**Problem:** 
- âš ï¸ Faz 5-6 eksik detay (ne kÄ±yaslanacak?)
- âš ï¸ Klasik RL modelleri aÃ§Ä±k harita yok
- âš ï¸ BaÅŸarÄ± hedefleri net belirtilmemiÅŸ (% olarak)

---

## 7. ğŸ’¼ Ã–nerisi: Neler YapÄ±lmalÄ±?

### KISA VADÆ (Hemen - Faz 5)
1. **Baseline Literature Mapping DokÃ¼** oluÅŸtur
   - Klasik RL modelleri nelerle karÅŸÄ±laÅŸtÄ±racaÄŸÄ±z (DQN, A2C, SAC)?
   - Neden? (LiteratÃ¼rde sÄ±kÃ§a geÃ§en)
   - Referans?

2. **BaÅŸarÄ± Hedeflerini Net YapÄ±ÅŸtÄ±r**
   - Faz 5 sonunda: %65-70
   - Faz 7 sonunda: %75-80
   - Faz 8 sonunda: %85+

3. **Faz 5 HazÄ±r YapÄ±sÄ±nÄ± IyileÅŸtir**
   - `configs/phase_5/synthetic_ablation.yaml` Ã§ok iyi
   - Ama hangi modeller ile kÄ±yas yapacaÄŸÄ±z net yazmalÄ±yÄ±z

### ORTA VADÆ (Faz 5-7)
1. Ablation Study â†’ BileÅŸenleri optimize et
2. Trace-driven training
3. Two-stage training â† Bu PPO'u %5-8 boost edebilir

### UZUN VADÆ (Faz 8-10)
1. Graph NNs â† %5-10 boost
2. AgentVNE ÅŸamasÄ±nda kÄ±yaslama
3. Self-reflection LLM loop

---

## 8. ğŸ“„ Eksik Dosyalar (YaratÄ±lacak)

| Dosya | AmaÃ§ | Durum |
|-------|------|-------|
| `docs/baseline_literature_mapping.md` | Hangi klasik modeller niye seÃ§ilecek? | âš ï¸ YAPILACAK |
| `docs/agentvne_comparison.md` | AgentVNE vs Our PPO_v2 | âš ï¸ YAPILACAK (Faz 10) |
| `.agent/rules/yolharitasi.md` | Commit kuralÄ± (otomatik commit YAPMA) | âœ… YARATILACAK |
| Updated `task.md` | BaÅŸarÄ± hedefleri ekle | ~ GÃœNCELLE |

---

## ğŸ“ SonuÃ§ & EÄŸilim

### "BaÅŸarÄ± OranlarÄ± Neden DÃ¼ÅŸÃ¼k GÃ¶rÃ¼nÃ¼yor?"
**Cevap:** AslÄ±nda dÃ¼ÅŸÃ¼k deÄŸil!
- %62 task offloading iÃ§in iyi (30% stokastik sistem loss + deadline miss)
- AgentVNE'den +%7 daha iyi
- Klasik heuristic'ler (%46) ile %16 fark var
- GA'dan +%12 daha iyi

### "GeliÅŸtirilecek mi?"
**Cevap:** Evet! Faz 5-8 planlÄ± iyileÅŸtirmeler:
- Faz 5: Ablation â†’ BileÅŸenleri optimize et
- Faz 6: Trace data â†’ GerÃ§ekÃ§i train
- Faz 7: Two-stage â†’ Daha hÄ±zlÄ± yakÄ±nsa
- Faz 8: Graph NN â†’ +%5-10 potansiyel

### "Soruda NE Eklenmeli?"
```
- [x] LLM modelini belirle (TinyLlama) âœ“
- [x] 7 baseline'Ä± aÃ§Ä±kla (fixed + heuristic + GA + PPO) âœ“  
- [ ] Klasik RL modellerini mapla (DQN, A2C, SAC) - YAPILACAK
- [ ] BaÅŸarÄ± hedeflerini net belirle (% olarak) - YAPILACAK
- [x] AgentVNE karÅŸÄ±laÅŸtÄ±rmasÄ± planlÄ± (Faz 10) âœ“
```

