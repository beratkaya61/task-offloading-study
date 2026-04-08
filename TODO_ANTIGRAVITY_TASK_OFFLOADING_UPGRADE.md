Bkz. ortak kavram sozlugu: v2_docs/project_concepts_glossary.md

# TODO â€” Antigravity Upgrade Plan for `task-offloading-study`

> Guncel durum notu (2026-04-02):
> Faz 1-5 tamamlandi ve Faz 5 sentetik taraf donduruldu.
> Faz 6 aktif asamadadir.
> `src/core/trace_loader.py` implement edildi ve `experiments/trace/train_ppo.py` akisina baglandi.
> Faz 6'da acik kalan ana maddeler artik `Success Bonus`, switching overhead, domain-shift analizi ve final trace artefaktlaridir.

Bu dosya, projeyi seminer seviyesinden **tez + makale + gÃ¼Ã§lÃ¼ demo** seviyesine Ã§Ä±karmak iÃ§in hazÄ±rlanmÄ±ÅŸ sÄ±ralÄ± geliÅŸtirme planÄ±dÄ±r.

AmaÃ§ sadece birkaÃ§ kÃ¼Ã§Ã¼k iyileÅŸtirme yapmak deÄŸil; mevcut sistemi ÅŸu Ã§izgiye taÅŸÄ±maktÄ±r:

> **LLM-guided, trace-driven, graph-aware, partial task offloading framework with rigorous evaluation, ablation studies, and publication-ready methodology**

---

# 0. Kuzey YÄ±ldÄ±zÄ± (North Star)

## Nihai hedef

AÅŸaÄŸÄ±daki baÅŸlÄ±ÄŸa yaklaÅŸan bir sistem Ã¼retmek:

**LLM-Guided Graph Reinforcement Learning for Semantic-Aware Partial Task Offloading in Dynamic Mobile Edge Computing**

## Nihai katkÄ± paketi

- LLM yalnÄ±zca aÃ§Ä±klama veren modÃ¼l deÄŸil, **action prior / semantic bias generator** olacak.
- PPO yalnÄ±zca reward shaping ile Ã§alÄ±ÅŸan model deÄŸil, **gerÃ§ek simÃ¼lasyonla hizalanmÄ±ÅŸ** bir policy olacak.
- EÄŸitim, doÄŸrudan random mock taskâ€™larla deÄŸil, **trace-driven ve staged training** ile yapÄ±lacak.
- KÄ±yaslamalar sadece Random/Greedy ile deÄŸil, **heuristic + RL + semantic ablation** ailesiyle yapÄ±lacak.
- Metrikler yalnÄ±zca ortalama latency deÄŸil; **deadline miss, p95 latency, energy per success, fairness, jitter, QoE, decision overhead** olacak.
- Sistem mevcut vektÃ¶r stateâ€™ten ileride **graph-aware policy** seviyesine taÅŸÄ±nacak.

---

# 1. Mevcut Durum Ã–zeti

## Kod tabanÄ±nda gÃ¶rÃ¼len ana bileÅŸenler

- `src/rl_env.py`
  - 6 aksiyonlu partial offloading yapÄ±sÄ± var.
  - Durum vektÃ¶rÃ¼ 8 boyutlu.
  - LLM Ã¶nerisi observationâ€™a one-hot ile giriyor.
  - Reward iÃ§inde LLM alignment bonus/penalty var.
- `src/llm_analyzer.py`
  - Rule-based + LLM-based analiz desteÄŸi var.
  - `recommended_target`, `priority_score`, `urgency`, `complexity`, `bandwidth_need`, `confidence` Ã¼retiyor.
- `src/simulation_env.py`
  - SimPy tabanlÄ± akÄ±ÅŸ var.
  - `NUM_DEVICES = 20`, `NUM_EDGE_SERVERS = 3` benzeri Ã§ok cihazlÄ± ortam var.
  - Fairness, jitter, QoE takibi mevcut.
- `src/train_agent.py`
  - PPO eÄŸitimi var.
  - Ancak eÄŸitim ortamÄ± sade/mock ve gerÃ§ek simÃ¼lasyonla tam hizalÄ± gÃ¶rÃ¼nmÃ¼yor.
- `src/gui.py`
  - GÃ¼Ã§lÃ¼ bir demo arayÃ¼zÃ¼ var.

## Åu anki kritik eksikler

- Train environment ile gerÃ§ek simulation environment arasÄ±nda boÅŸluk var.
- LLM katkÄ±sÄ± reward shaping aÄŸÄ±rlÄ±klÄ±; **prior distribution** olarak kullanÄ±lmÄ±yor.
- Baseline seti geniÅŸ deÄŸil.
- Ablation yapÄ±sÄ± sistematik deÄŸil.
- Trace-driven deney akÄ±ÅŸÄ± net deÄŸil.
- Reproducibility zayÄ±f.
- Evaluation pipeline ayrÄ± ve tekrarlanabilir deÄŸil.
- Karar overheadâ€™i (LLM sÃ¼resi + PPO inference sÃ¼resi) net Ã¶lÃ§Ã¼lmÃ¼yor.
- Graph-based policy henÃ¼z yok.

---

# 2. Uygulama Stratejisi

Bu planÄ± **9 fazda** yÃ¼rÃ¼teceÄŸiz.

Ã–nerilen sÄ±ra:

1. Stabilizasyon ve deney tekrarlanabilirliÄŸi
2. Ortam hizalama
3. LLM entegrasyonunu gÃ¼Ã§lendirme
4. Baseline ve ablation paketi
5. GerÃ§ek veri / trace entegrasyonu
6. GeliÅŸmiÅŸ metrik ve analiz
7. Two-stage training
8. Graph-aware yÃ¼kseltme
9. Tez / makale Ã¼retim paketi

---

# 3. Faz 1 â€” Reproducibility ve Kod TemizliÄŸi

## Hedef

Projeyi â€œÃ§alÄ±ÅŸÄ±yorâ€ seviyesinden â€œtekrarlanabilir araÅŸtÄ±rma prototipiâ€ seviyesine Ã§Ä±karmak.

## YapÄ±lacaklar

### 3.1. Proje klasÃ¶r yapÄ±sÄ±nÄ± standardize et

- [ ] `src/` iÃ§inde aÅŸaÄŸÄ±daki yeni modÃ¼lleri oluÅŸtur:
  - [ ] `src/baselines.py`
  - [ ] `src/evaluation.py`
  - [ ] `src/metrics.py`
  - [ ] `src/config.py`
  - [ ] `src/trace_loader.py`
  - [ ] `src/semantic_prior.py`
  - [ ] `src/pretrain_policy.py`
  - [ ] `src/utils/reproducibility.py`
- [ ] `configs/` klasÃ¶rÃ¼ oluÅŸtur:
  - [ ] `configs/synthetic/rl_training.yaml`
  - [ ] `configs/synthetic/policy_evaluation.yaml`
  - [ ] `configs/synthetic/ablation.yaml`
  - [ ] `configs/trace/ppo_training.yaml`
- [ ] `results/` klasÃ¶rÃ¼ oluÅŸtur:
  - [ ] `results/raw/`
  - [ ] `results/processed/`
  - [ ] `results/figures/`
  - [ ] `results/tables/`

### 3.2. Seed ve determinism kontrolÃ¼

- [ ] `numpy`, `random`, `torch`, `gymnasium` iÃ§in tek seed fonksiyonu yaz.
- [ ] Her deneyde seed loglansÄ±n.
- [ ] `README.md` iÃ§ine â€œHow to reproduceâ€ bÃ¶lÃ¼mÃ¼ ekle.

### 3.3. Experiment logging altyapÄ±sÄ±

- [ ] Her run sonunda JSON/CSV log Ã¼ret:
  - [ ] run_id
  - [ ] timestamp
  - [ ] config hash
  - [ ] seed
  - [ ] model type
  - [ ] semantic mode
  - [ ] total tasks
  - [ ] success rate
  - [ ] avg latency
  - [ ] p95 latency
  - [ ] avg energy
  - [ ] fairness
  - [ ] jitter
  - [ ] qoe
  - [ ] decision overhead
- [ ] `evaluation.py` ile bu loglarÄ± tek tabloda birleÅŸtir.

## Done kriteri

- [ ] AynÄ± config + seed ile aynÄ± sonuÃ§ aralÄ±ÄŸÄ±na yakÄ±n tekrar alÄ±nabiliyor.
- [ ] TÃ¼m deneyler `results/` altÄ±nda standart biÃ§imde kaydediliyor.

---

# 4. Faz 2 â€” Train Environment ve GerÃ§ek SimÃ¼lasyon HizalamasÄ±

## Hedef

`rl_env.py` ile `simulation_env.py` arasÄ±ndaki farkÄ± azaltmak.

## Kritik sorun

Åu an `rl_env.py` reset sÄ±rasÄ±nda mock task ve mock device Ã¼retiyor. Bu pratik ama tez aÃ§Ä±sÄ±ndan zayÄ±f. Ã–ÄŸrenilen politika gerÃ§ek ortam dinamiklerine tam maruz kalmÄ±yor.

## YapÄ±lacaklar

### 4.1. Ortak state builder yaz

- [ ] `src/state_builder.py` oluÅŸtur.
- [ ] Hem `rl_env.py` hem `simulation_env.py` aynÄ± state Ã¼retim fonksiyonunu kullansÄ±n.
- [ ] State iÃ§inde ÅŸu alanlar yer alsÄ±n:
  - [ ] normalized SNR
  - [ ] task size
  - [ ] cpu cycles
  - [ ] battery
  - [ ] nearest edge load
  - [ ] queue length
  - [ ] predicted cloud latency
  - [ ] mobility indicator / distance-to-edge
  - [ ] task deadline tightness
  - [ ] LLM semantic prior

### 4.2. Mock yerine simulator-backed training

- [ ] PPO training iÃ§in â€œlightweight simulator modeâ€ oluÅŸtur.
- [ ] `OffloadingEnv.reset()` rastgele mock Ã¼retmek yerine simÃ¼latÃ¶rden Ã¶rnek alsÄ±n.
- [ ] Task oluÅŸturma mantÄ±ÄŸÄ± `simulation_env.py` ile uyumlu hale gelsin.

### 4.3. Reward fonksiyonunu tek yerde topla

- [ ] `src/reward.py` oluÅŸtur.
- [ ] Reward bileÅŸenlerini fonksiyonlaÅŸtÄ±r:
  - [ ] latency term
  - [ ] energy term
  - [ ] deadline term
  - [ ] semantic alignment term
  - [ ] battery preservation term
  - [ ] cloud cost term
  - [ ] partial offloading utility term
- [ ] Reward weights config dosyasÄ±ndan gelsin.

### 4.4. Episode tanÄ±mÄ±nÄ± iyileÅŸtir

- [ ] Åu an step sonunda `done = True`; bunu geliÅŸtir.
- [ ] Ã‡ok adÄ±mlÄ± episode tasarla:
  - [ ] tek episode = birden fazla gÃ¶rev
  - [ ] veya sliding window task sequence
- [ ] PPOâ€™nun uzun ufuklu davranÄ±ÅŸ Ã¶ÄŸrenmesi saÄŸlansÄ±n.

## Done kriteri

- [ ] EÄŸitim ve deÄŸerlendirme aynÄ± fiziksel mantÄ±ÄŸÄ± paylaÅŸÄ±yor.
- [ ] Tek adÄ±mlÄ±k aÅŸÄ±rÄ± sade RL ortamÄ±ndan Ã§Ä±kÄ±lmÄ±ÅŸ oluyor.

---

# 5. Faz 3 â€” LLM Entegrasyonunu GerÃ§ek KatkÄ±ya DÃ¶nÃ¼ÅŸtÃ¼r

## Hedef

LLMâ€™i sadece reward shaping yapan yardÄ±mcÄ± modÃ¼l olmaktan Ã§Ä±karÄ±p, karar mekanizmasÄ±na **structured prior** saÄŸlayan bileÅŸen haline getirmek.

## YapÄ±lacaklar

### 5.1. `recommended_target` yerine action prior Ã¼ret

- [ ] `src/semantic_prior.py` oluÅŸtur.
- [ ] LLM / rule-based analyzer aÅŸaÄŸÄ±daki daÄŸÄ±lÄ±mÄ± Ã¼retsin:
  - [ ] `P(local)`
  - [ ] `P(edge_25)`
  - [ ] `P(edge_50)`
  - [ ] `P(edge_75)`
  - [ ] `P(edge_100)`
  - [ ] `P(cloud)`
- [ ] Observation iÃ§ine tek one-hot yerine bu 6 boyutlu prior vektÃ¶rÃ¼ girsin.

### 5.2. Confidence kullanÄ±mÄ±nÄ± gÃ¼Ã§lendir

- [ ] Semantic confidence yalnÄ±zca reward scalingâ€™de deÄŸil, aÅŸaÄŸÄ±daki yerlerde kullanÄ±lsÄ±n:
  - [ ] policy prior blending
  - [ ] action masking veya action bias
  - [ ] explanation confidence
- [ ] `confidence calibration` iÃ§in ayrÄ± analiz yap.

### 5.3. LLM Ã§Ä±ktÄ±sÄ±nÄ± parse edilebilir JSON haline getir

- [ ] `llm_analyzer.py` iÃ§inde structured output zorunlu hale getir.
- [ ] Parse baÅŸarÄ±sÄ±zsa fallback mekanizmasÄ± loglansÄ±n.
- [ ] `analysis_method` ile birlikte `parse_success`, `fallback_reason` tutulmalÄ±.

### 5.4. Semantic explanation bank

- [ ] Her karar iÃ§in explanation log Ã¼ret:
  - [ ] raw semantic input
  - [ ] analyzer output
  - [ ] selected action
  - [ ] reward decomposition
  - [ ] final outcome
- [ ] Bu log ileride tezde case study olarak kullanÄ±lacak.

## Done kriteri

- [ ] LLM katkÄ±sÄ± yalnÄ±zca â€œbonus/penaltyâ€ deÄŸil, aÃ§Ä±kÃ§a state/action prior Ã¼reten modÃ¼l oluyor.
- [ ] Semantic prior ve confidence grafiklerle gÃ¶sterilebiliyor.

---

# 6. Faz 4 â€” Baseline Ailesini GeniÅŸlet

## Hedef

KÄ±yaslamayÄ± ciddi hale getirmek.

## Minimum baseline seti

### Zorunlu baselineâ€™lar

- [ ] `LocalOnly`
- [ ] `EdgeOnly`
- [ ] `CloudOnly`
- [ ] `RandomPolicy`
- [ ] `GreedyLatency`
- [ ] `GreedyEnergy`
- [ ] `GeneticAlgorithm` (Meta-Heuristic / Eksik 1)
- [ ] `DQN` (Classical RL / Eksik 2)
- [ ] `A2C` (Classical RL / Eksik 2)
- [ ] `PPO_NoSemantics`
- [ ] `PPO_WithOneHotLLM`
- [ ] `PPO_WithSemanticPrior`  â† hedef ana model

### Orta seviye ek baselineâ€™lar

- [ ] `DQN` veya `DoubleDQN`
- [ ] `SAC/DDPG` tarzÄ± continuous ratio policy (opsiyonel ama Ã§ok deÄŸerli)

## Kod gÃ¶revleri

- [ ] `src/baselines.py` iÃ§ine bÃ¼tÃ¼n baselineâ€™larÄ± tek arayÃ¼z altÄ±nda topla.
- [ ] Hepsi aynÄ± task stream ve aynÄ± seed ile deÄŸerlendirilsin.
- [ ] SonuÃ§lar ortak evaluator ile Ã¶lÃ§Ã¼lsÃ¼n.

## Done kriteri

- [ ] â€œBizim model iyiâ€ cÃ¼mlesi en az 8â€“10 baselineâ€™a karÅŸÄ± desteklenebiliyor.

---

# 7. Faz 5 â€” Sistematik Ablation Study

## Hedef

KatkÄ±nÄ±n nereden geldiÄŸini gÃ¶stermek.

## YapÄ±lacak ablationâ€™lar

- [ ] `Full Model`
- [ ] `w/o Semantics`
- [ ] `w/o Reward Shaping`
- [ ] `w/o Semantic Prior`
- [ ] `w/o Confidence`
- [ ] `w/o Partial Offloading` (yalnÄ±zca Local/Edge/Cloud)
- [ ] `w/o Battery Awareness`
- [ ] `w/o Queue Awareness`
- [ ] `w/o Mobility Features`

## Analiz eksenleri

- [ ] avg latency
- [ ] p95 latency
- [ ] deadline miss ratio
- [ ] success ratio
- [ ] avg energy
- [ ] fairness
- [ ] jitter
- [ ] QoE

## Done kriteri

- [ ] Her ana tasarÄ±m kararÄ±nÄ±n katkÄ±sÄ± tablo ve grafikle ayrÄ±ÅŸtÄ±rÄ±labiliyor.

---

# 8. Faz 6 â€” GerÃ§ek Veri / Trace-Driven Deney Paketi

## Hedef

Synthetic-only gÃ¶rÃ¼nÃ¼mden Ã§Ä±kmak.

## YapÄ±lacaklar

### 8.1. Trace loader

- [ ] `src/trace_loader.py` yaz.
- [ ] Google Cluster Trace iÃ§in:
  - [ ] task arrival pattern
  - [ ] cpu demand
  - [ ] duration / runtime approximation
- [ ] Didi Gaia iÃ§in:
  - [ ] mobility trajectory
  - [ ] distance-to-edge profile
  - [ ] handover benzeri hareketlilik etkileri

### 8.2. Trace-to-task mapping

- [ ] Task fields ile trace fieldâ€™larÄ± net eÅŸleÅŸtir:
  - [ ] `size_bits`
  - [ ] `cpu_cycles`
  - [ ] `deadline`
  - [ ] `task_type`
- [ ] Eksik alanlar iÃ§in aÃ§Ä±k mapping assumptions dosyasÄ± yaz:
  - [ ] `doc/trace_mapping_assumptions.md`

### 8.3. Ä°ki deney modu oluÅŸtur

- [ ] `synthetic mode`
- [ ] `trace-driven mode`

### 8.4. Domain shift testi

- [ ] Syntheticâ€™te eÄŸit, traceâ€™de test et.
- [ ] Traceâ€™de eÄŸit, syntheticâ€™te test et.
- [ ] Generalization tablosu oluÅŸtur.

## Done kriteri

- [ ] Proje artÄ±k sadece â€œgÃ¼zel simÃ¼lasyonâ€ deÄŸil, trace-driven Ã§alÄ±ÅŸma haline geliyor.

---

# 9. Faz 7 â€” GeliÅŸmiÅŸ Metrik ve Ä°statistiksel Analiz

## Hedef

Makale seviyesinde Ã¶lÃ§Ã¼m disiplini kurmak.

## Zorunlu metrikler

- [ ] `Average Latency`
- [ ] `p95 Latency`
- [ ] `p99 Latency`
- [ ] `Deadline Miss Ratio`
- [ ] `Task Completion Ratio / Acceptance Ratio`
- [ ] `Average Energy Consumption`
- [ ] `Energy per Successful Task`
- [ ] `Battery Lifetime / Battery Depletion Rate`
- [ ] `Average Queue Waiting Time`
- [ ] `Jain's Fairness Index`
- [ ] `Jitter`
- [ ] `QoE Score`
- [ ] `Decision Overhead`
  - [ ] LLM inference time
  - [ ] PPO inference time
  - [ ] end-to-end controller latency

## Ä°statistiksel analiz

- [ ] Her deney en az 5 seed ile koÅŸsun.
- [ ] Ortalama Â± std ver.
- [ ] 95% gÃ¼ven aralÄ±ÄŸÄ± ver.
- [ ] Wilcoxon signed-rank testi uygula.
- [ ] Gerekirse Friedman + Nemenyi ekle.

## GÃ¶rseller

- [ ] latency boxplot
- [ ] p95 latency bar chart
- [ ] energy-success scatter
- [ ] fairness-jitter comparison
- [ ] QoE violin plot
- [ ] CDF of latency

## Done kriteri

- [ ] SonuÃ§ kÄ±smÄ± artÄ±k yalnÄ±zca tek sayÄ±dan oluÅŸmuyor; istatistiksel olarak savunulabilir hale geliyor.

---

# 10. Faz 8 â€” Two-Stage Training (En GÃ¼Ã§lÃ¼ Orta Vadeli KatkÄ±)

## Hedef

AgentVNEâ€™deki staged training disiplinini offloading problemine uyarlamak.

## YapÄ±lacaklar

### 10.1. Oracle / heuristic label Ã¼retimi

- [ ] Basit maliyet fonksiyonu ile yarÄ±-optimal karar etiketi Ã¼ret:
  - [ ] latency-aware oracle
  - [ ] energy-aware oracle
  - [ ] weighted objective oracle
- [ ] Bu etiketleri supervised pretraining iÃ§in dataset haline getir.

### 10.2. Policy pretraining

- [ ] `src/pretrain_policy.py` oluÅŸtur.
- [ ] PPO policy aÄŸÄ±nÄ± Ã¶nce imitation / supervised ÅŸekilde Ä±sÄ±t.
- [ ] Sonra PPO ile fine-tune et.

### 10.3. KarÅŸÄ±laÅŸtÄ±rma

- [ ] `PPO from scratch`
- [ ] `Pretrained + PPO`
- [ ] convergence speed
- [ ] sample efficiency
- [ ] final performance

## Done kriteri

- [ ] EÄŸitim daha stabil hale geliyor.
- [ ] Tezde â€œneden staged training?â€ sorusuna net cevap var.

---

# 11. Faz 9 â€” Graph-Aware Policy Upgrade (Efsane Seviye KatkÄ±)

## Hedef

Projeyi gerÃ§ekten sÄ±ra dÄ±ÅŸÄ± hale getirecek ana sÄ±Ã§rama.

## Neden gerekli?

Åu an state bÃ¼yÃ¼k Ã¶lÃ§Ã¼de vektÃ¶r tabanlÄ±. Oysa sistem doÄŸal olarak graf yapÄ±sÄ±nda:

- cihazlar
- edge sunucular
- bulut
- aralarÄ±ndaki mesafe / link quality
- queue iliÅŸkileri
- shared congestion
- mobility

Bu yÃ¼zden GNN tabanlÄ± policy Ã§ok gÃ¼Ã§lÃ¼ bir upgrade olur.

## YapÄ±lacaklar

### 11.1. Graph state tanÄ±mÄ±

- [ ] Node tipleri:
  - [ ] IoT device
  - [ ] Edge server
  - [ ] Cloud node
  - [ ] Current task node (opsiyonel)
- [ ] Node featureâ€™larÄ±:
  - [ ] battery
  - [ ] cpu capacity
  - [ ] queue load
  - [ ] mobility state
  - [ ] semantic priority
  - [ ] deadline pressure
- [ ] Edge featureâ€™larÄ±:
  - [ ] distance
  - [ ] predicted rate
  - [ ] transmission latency
  - [ ] congestion estimate

### 11.2. Graph encoder

- [ ] `src/graph_policy.py` oluÅŸtur.
- [ ] PyTorch Geometric veya DGL ile GNN encoder kur.
- [ ] Ã‡Ä±kÄ±ÅŸta action logits Ã¼ret.

### 11.3. Semantic prior fusion

- [ ] LLM priorâ€™Ä± graph node feature veya policy prior olarak birleÅŸtir.
- [ ] Erken fusion ve geÃ§ fusion olmak Ã¼zere iki strateji test et.

### 11.4. KarÅŸÄ±laÅŸtÄ±rma

- [ ] `MLP-PPO`
- [ ] `GNN-PPO`
- [ ] `GNN-PPO + Semantic Prior`

## Done kriteri

- [ ] Yeni bir makale katkÄ±sÄ± doÄŸuyor.
- [ ] â€œNeden graph?â€ sorusuna deneysel cevap verilebiliyor.

---

# 12. Demo ve GUI YÃ¼kseltmeleri

## Hedef

Seminerde â€œwow effectâ€ yaratmak.

## YapÄ±lacaklar

- [ ] GUIâ€™ye â€œExperiment Modeâ€ ekle:
  - [ ] model seÃ§imi
  - [ ] baseline seÃ§imi
  - [ ] semantic mode aÃ§/kapat
  - [ ] trace mode aÃ§/kapat
- [ ] GUIâ€™de â€œAblation comparison panelâ€ ekle.
- [ ] GUIâ€™ye latency CDF mini-chart ekle.
- [ ] GUIâ€™ye decision overhead paneli ekle.
- [ ] Semantic prior daÄŸÄ±lÄ±mÄ±nÄ± bar chart olarak gÃ¶ster.
- [ ] Reward decomposition paneli ekle:
  - [ ] latency contribution
  - [ ] energy contribution
  - [ ] deadline contribution
  - [ ] semantic contribution

## Done kriteri

- [ ] Sunum sÄ±rasÄ±nda modelin neden ve nasÄ±l karar verdiÄŸi gÃ¶rsel olarak anlatÄ±labiliyor.

---

# 13. Tez ve Makale Ãœretim Paketi

## YazÄ±lacak yeni dokÃ¼manlar

- [ ] `doc/methodology_v2.md`
- [ ] `doc/ablation_plan.md`
- [ ] `doc/evaluation_protocol.md`
- [ ] `doc/trace_mapping_assumptions.md`
- [ ] `doc/threats_to_validity.md`
- [ ] `doc/future_work_graph_rl.md`

## Makale tablolarÄ±

- [ ] Baseline comparison table
- [ ] Ablation table
- [ ] Training strategy comparison table
- [ ] Computational overhead table
- [ ] Dataset / trace mapping table

## Makale ÅŸekilleri

- [ ] overall architecture figure
- [ ] state-action-reward flowchart
- [ ] semantic prior generation diagram
- [ ] training pipeline figure
- [ ] evaluation pipeline figure
- [ ] CDF / boxplot / scatter seti

## Done kriteri

- [ ] Kod + deney + yazÄ±m aynÄ± omurgada birleÅŸiyor.

---

# 14. HaftalÄ±k SÄ±ralÄ± Sprint PlanÄ±

## Sprint 1 â€” Temizlik ve deney omurgasÄ±

- [ ] klasÃ¶r yapÄ±sÄ±
- [ ] config sistemi
- [ ] logging
- [ ] seed / reproducibility
- [ ] evaluator iskeleti

## Sprint 2 â€” Ortam hizalama

- [ ] ortak state builder
- [ ] ortak reward module
- [ ] episode redesign
- [ ] simulator-backed RL env

## Sprint 3 â€” Semantic prior

- [ ] action prior Ã¼retimi
- [ ] confidence entegrasyonu
- [ ] structured output
- [ ] explanation logs

## Sprint 4 â€” Baseline ve ablation

- [ ] baseline engine
- [ ] ablation configs
- [ ] ilk kÄ±yas tablolarÄ±

## Sprint 5 â€” Trace-driven pipeline

- [ ] trace loader
- [ ] synthetic vs trace
- [ ] generalization testleri

## Sprint 6 â€” Two-stage training

- [ ] oracle label Ã¼retimi
- [ ] pretraining
- [ ] PPO fine-tuning

## Sprint 7 â€” GeliÅŸmiÅŸ metrik ve istatistik

- [ ] p95/p99
- [ ] DMR
- [ ] CI ve significance test

## Sprint 8 â€” Graph RL upgrade

- [ ] graph state
- [ ] GNN encoder
- [ ] MLP vs GNN vs GNN+Semantic

## Sprint 9 â€” Demo + yazÄ±m

- [ ] GUI gÃ¼Ã§lendirme
- [ ] seminer slaytlarÄ±
- [ ] makale figÃ¼rleri

---

# 15. Ã‡ok Net Ã–ncelik Listesi (Ã–nce BunlarÄ± Yap)

## P0 â€” Hemen

- [ ] `evaluation.py` yaz
- [ ] `baselines.py` yaz
- [ ] `reward.py` oluÅŸtur
- [ ] `state_builder.py` oluÅŸtur
- [ ] experiment logging ekle

## P1 â€” Ä°lk bÃ¼yÃ¼k kazanÃ§

- [ ] LLM one-hot yerine action prior vektÃ¶rÃ¼ Ã¼ret
- [ ] `PPO_NoSemantics` ve `PPO_WithSemanticPrior` kÄ±yasÄ±nÄ± Ã§Ä±kar
- [ ] deadline miss + p95 latency ekle

## P2 â€” Tez kalitesini sÄ±Ã§ratacak iÅŸ

- [ ] trace-driven mode
- [ ] staged training
- [ ] ablation study

## P3 â€” Makale seviyesini sÄ±Ã§ratacak iÅŸ

- [ ] graph-aware policy
- [ ] statistical testing
- [ ] decision overhead analysis

---

# 16. Riskler ve Dikkat Edilecek Noktalar

- [ ] LLMâ€™yi fazla baskÄ±n yapÄ±p PPOâ€™yu gereksiz hale getirme.
- [ ] Reward shaping ile leakage yaratma.
- [ ] Baselineâ€™larÄ± zayÄ±f bÄ±rakÄ±p kÄ±yasÄ± yapay ÅŸekilde kolaylaÅŸtÄ±rma.
- [ ] Trace mapping varsayÄ±mlarÄ±nÄ± gizleme.
- [ ] GUI Ã§alÄ±ÅŸÄ±yor diye deney disiplinini ikinci plana atma.
- [ ] Mock eÄŸitim ortamÄ±nÄ± gerÃ§ek sistem gibi sunma.

---

# 17. Definition of Done â€” â€œProje Efsane Olduâ€ Kontrol Listesi

AÅŸaÄŸÄ±dakilerin Ã§oÄŸu tamamlandÄ±ysa proje gerÃ§ekten gÃ¼Ã§lÃ¼ hale gelmiÅŸ demektir:

- [ ] AynÄ± deneyler tekrarlanabilir.
- [ ] EÄŸitim ve deÄŸerlendirme ortamÄ± hizalÄ±.
- [ ] LLM semantic prior Ã¼retiyor.
- [ ] En az 8 baseline var.
- [ ] En az 6 ablation var.
- [ ] Synthetic + trace-driven deney var.
- [ ] p95/p99 latency, DMR, fairness, jitter, QoE raporlanÄ±yor.
- [ ] 5 farklÄ± seed ile CI veriliyor.
- [ ] staged training karÅŸÄ±laÅŸtÄ±rmasÄ± var.
- [ ] decision overhead Ã¶lÃ§Ã¼lÃ¼yor.
- [ ] GUIâ€™de explanation ve comparison paneli var.
- [ ] Seminer iÃ§in net mimari ÅŸekli hazÄ±r.
- [ ] Tez iÃ§in methodology ve evaluation protocol yazÄ±lmÄ±ÅŸ.
- [ ] MLP-PPO ile GNN-PPO karÅŸÄ±laÅŸtÄ±rmasÄ± baÅŸlamÄ±ÅŸ veya tamamlanmÄ±ÅŸ.

---

# 18. En Kritik Dosya BazlÄ± MÃ¼dahale PlanÄ±

## `src/rl_env.py`

- [ ] tek-adÄ±mlÄ± episode yapÄ±sÄ±nÄ± kaldÄ±r
- [ ] observationâ€™Ä± semantic prior ile geniÅŸlet
- [ ] reward hesaplamasÄ±nÄ± dÄ±ÅŸ modÃ¼le taÅŸÄ±
- [ ] mock reset akÄ±ÅŸÄ±nÄ± simulator-backed hale getir

## `src/llm_analyzer.py`

- [ ] structured JSON output
- [ ] action prior Ã¼retimi
- [ ] confidence calibration
- [ ] fallback loglama

## `src/simulation_env.py`

- [ ] ortak state/reward entegrasyonu
- [ ] trace-driven task generation
- [ ] karar overhead Ã¶lÃ§Ã¼mÃ¼
- [ ] yeni metriklerin toplanmasÄ±

## `src/train_agent.py`

- [ ] config-based eÄŸitim
- [ ] seed yÃ¶netimi
- [ ] pretraining + PPO pipeline
- [ ] model versioning

## `src/gui.py`

- [ ] experiment mode
- [ ] ablation comparison panel
- [ ] semantic prior visualization
- [ ] reward decomposition
- [ ] decision overhead panel

---

# 19. Son Not

Bu planÄ±n mantÄ±ÄŸÄ± ÅŸu:

Ã–nce sistemi **bilimsel olarak savunulabilir** hale getir,
sonra **deneysel olarak gÃ¼Ã§lÃ¼** hale getir,
sonra **mimari olarak farklÄ±laÅŸtÄ±r**,
en sonda **sunum ve makale etkisini bÃ¼yÃ¼t**.

Yani sÄ±ra:

**stabilite â†’ kÄ±yas â†’ ablation â†’ trace â†’ staged training â†’ graph RL â†’ demo/makale**

Bu sÄ±rayÄ± bozma.




> Trace mapping assumptions dokumani olusturuldu: v2_docs/trace_mapping_assumptions.md
> Domain-shift evaluation icin config + script iskeleti eklendi: configs/trace/domain_shift_evaluation.yaml, experiments/trace/evaluate_domain_shift.py

