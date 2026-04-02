# TODO — Antigravity Upgrade Plan for `task-offloading-study`

Bu dosya, projeyi seminer seviyesinden **tez + makale + güçlü demo** seviyesine çıkarmak için hazırlanmış sıralı geliştirme planıdır.

Amaç sadece birkaç küçük iyileştirme yapmak değil; mevcut sistemi şu çizgiye taşımaktır:

> **LLM-guided, trace-driven, graph-aware, partial task offloading framework with rigorous evaluation, ablation studies, and publication-ready methodology**

---

# 0. Kuzey Yıldızı (North Star)

## Nihai hedef

Aşağıdaki başlığa yaklaşan bir sistem üretmek:

**LLM-Guided Graph Reinforcement Learning for Semantic-Aware Partial Task Offloading in Dynamic Mobile Edge Computing**

## Nihai katkı paketi

- LLM yalnızca açıklama veren modül değil, **action prior / semantic bias generator** olacak.
- PPO yalnızca reward shaping ile çalışan model değil, **gerçek simülasyonla hizalanmış** bir policy olacak.
- Eğitim, doğrudan random mock task’larla değil, **trace-driven ve staged training** ile yapılacak.
- Kıyaslamalar sadece Random/Greedy ile değil, **heuristic + RL + semantic ablation** ailesiyle yapılacak.
- Metrikler yalnızca ortalama latency değil; **deadline miss, p95 latency, energy per success, fairness, jitter, QoE, decision overhead** olacak.
- Sistem mevcut vektör state’ten ileride **graph-aware policy** seviyesine taşınacak.

---

# 1. Mevcut Durum Özeti

## Kod tabanında görülen ana bileşenler

- `src/rl_env.py`
  - 6 aksiyonlu partial offloading yapısı var.
  - Durum vektörü 8 boyutlu.
  - LLM önerisi observation’a one-hot ile giriyor.
  - Reward içinde LLM alignment bonus/penalty var.
- `src/llm_analyzer.py`
  - Rule-based + LLM-based analiz desteği var.
  - `recommended_target`, `priority_score`, `urgency`, `complexity`, `bandwidth_need`, `confidence` üretiyor.
- `src/simulation_env.py`
  - SimPy tabanlı akış var.
  - `NUM_DEVICES = 20`, `NUM_EDGE_SERVERS = 3` benzeri çok cihazlı ortam var.
  - Fairness, jitter, QoE takibi mevcut.
- `src/train_agent.py`
  - PPO eğitimi var.
  - Ancak eğitim ortamı sade/mock ve gerçek simülasyonla tam hizalı görünmüyor.
- `src/gui.py`
  - Güçlü bir demo arayüzü var.

## Şu anki kritik eksikler

- Train environment ile gerçek simulation environment arasında boşluk var.
- LLM katkısı reward shaping ağırlıklı; **prior distribution** olarak kullanılmıyor.
- Baseline seti geniş değil.
- Ablation yapısı sistematik değil.
- Trace-driven deney akışı net değil.
- Reproducibility zayıf.
- Evaluation pipeline ayrı ve tekrarlanabilir değil.
- Karar overhead’i (LLM süresi + PPO inference süresi) net ölçülmüyor.
- Graph-based policy henüz yok.

---

# 2. Uygulama Stratejisi

Bu planı **9 fazda** yürüteceğiz.

Önerilen sıra:

1. Stabilizasyon ve deney tekrarlanabilirliği
2. Ortam hizalama
3. LLM entegrasyonunu güçlendirme
4. Baseline ve ablation paketi
5. Gerçek veri / trace entegrasyonu
6. Gelişmiş metrik ve analiz
7. Two-stage training
8. Graph-aware yükseltme
9. Tez / makale üretim paketi

---

# 3. Faz 1 — Reproducibility ve Kod Temizliği

## Hedef

Projeyi “çalışıyor” seviyesinden “tekrarlanabilir araştırma prototipi” seviyesine çıkarmak.

## Yapılacaklar

### 3.1. Proje klasör yapısını standardize et

- [ ] `src/` içinde aşağıdaki yeni modülleri oluştur:
  - [ ] `src/baselines.py`
  - [ ] `src/evaluation.py`
  - [ ] `src/metrics.py`
  - [ ] `src/config.py`
  - [ ] `src/trace_loader.py`
  - [ ] `src/semantic_prior.py`
  - [ ] `src/pretrain_policy.py`
  - [ ] `src/utils/reproducibility.py`
- [ ] `configs/` klasörü oluştur:
  - [ ] `configs/synthetic/rl_training.yaml`
  - [ ] `configs/synthetic/policy_evaluation.yaml`
  - [ ] `configs/synthetic/ablation.yaml`
  - [ ] `configs/trace/ppo_training.yaml`
- [ ] `results/` klasörü oluştur:
  - [ ] `results/raw/`
  - [ ] `results/processed/`
  - [ ] `results/figures/`
  - [ ] `results/tables/`

### 3.2. Seed ve determinism kontrolü

- [ ] `numpy`, `random`, `torch`, `gymnasium` için tek seed fonksiyonu yaz.
- [ ] Her deneyde seed loglansın.
- [ ] `README.md` içine “How to reproduce” bölümü ekle.

### 3.3. Experiment logging altyapısı

- [ ] Her run sonunda JSON/CSV log üret:
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
- [ ] `evaluation.py` ile bu logları tek tabloda birleştir.

## Done kriteri

- [ ] Aynı config + seed ile aynı sonuç aralığına yakın tekrar alınabiliyor.
- [ ] Tüm deneyler `results/` altında standart biçimde kaydediliyor.

---

# 4. Faz 2 — Train Environment ve Gerçek Simülasyon Hizalaması

## Hedef

`rl_env.py` ile `simulation_env.py` arasındaki farkı azaltmak.

## Kritik sorun

Şu an `rl_env.py` reset sırasında mock task ve mock device üretiyor. Bu pratik ama tez açısından zayıf. Öğrenilen politika gerçek ortam dinamiklerine tam maruz kalmıyor.

## Yapılacaklar

### 4.1. Ortak state builder yaz

- [ ] `src/state_builder.py` oluştur.
- [ ] Hem `rl_env.py` hem `simulation_env.py` aynı state üretim fonksiyonunu kullansın.
- [ ] State içinde şu alanlar yer alsın:
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

- [ ] PPO training için “lightweight simulator mode” oluştur.
- [ ] `OffloadingEnv.reset()` rastgele mock üretmek yerine simülatörden örnek alsın.
- [ ] Task oluşturma mantığı `simulation_env.py` ile uyumlu hale gelsin.

### 4.3. Reward fonksiyonunu tek yerde topla

- [ ] `src/reward.py` oluştur.
- [ ] Reward bileşenlerini fonksiyonlaştır:
  - [ ] latency term
  - [ ] energy term
  - [ ] deadline term
  - [ ] semantic alignment term
  - [ ] battery preservation term
  - [ ] cloud cost term
  - [ ] partial offloading utility term
- [ ] Reward weights config dosyasından gelsin.

### 4.4. Episode tanımını iyileştir

- [ ] Şu an step sonunda `done = True`; bunu geliştir.
- [ ] Çok adımlı episode tasarla:
  - [ ] tek episode = birden fazla görev
  - [ ] veya sliding window task sequence
- [ ] PPO’nun uzun ufuklu davranış öğrenmesi sağlansın.

## Done kriteri

- [ ] Eğitim ve değerlendirme aynı fiziksel mantığı paylaşıyor.
- [ ] Tek adımlık aşırı sade RL ortamından çıkılmış oluyor.

---

# 5. Faz 3 — LLM Entegrasyonunu Gerçek Katkıya Dönüştür

## Hedef

LLM’i sadece reward shaping yapan yardımcı modül olmaktan çıkarıp, karar mekanizmasına **structured prior** sağlayan bileşen haline getirmek.

## Yapılacaklar

### 5.1. `recommended_target` yerine action prior üret

- [ ] `src/semantic_prior.py` oluştur.
- [ ] LLM / rule-based analyzer aşağıdaki dağılımı üretsin:
  - [ ] `P(local)`
  - [ ] `P(edge_25)`
  - [ ] `P(edge_50)`
  - [ ] `P(edge_75)`
  - [ ] `P(edge_100)`
  - [ ] `P(cloud)`
- [ ] Observation içine tek one-hot yerine bu 6 boyutlu prior vektörü girsin.

### 5.2. Confidence kullanımını güçlendir

- [ ] Semantic confidence yalnızca reward scaling’de değil, aşağıdaki yerlerde kullanılsın:
  - [ ] policy prior blending
  - [ ] action masking veya action bias
  - [ ] explanation confidence
- [ ] `confidence calibration` için ayrı analiz yap.

### 5.3. LLM çıktısını parse edilebilir JSON haline getir

- [ ] `llm_analyzer.py` içinde structured output zorunlu hale getir.
- [ ] Parse başarısızsa fallback mekanizması loglansın.
- [ ] `analysis_method` ile birlikte `parse_success`, `fallback_reason` tutulmalı.

### 5.4. Semantic explanation bank

- [ ] Her karar için explanation log üret:
  - [ ] raw semantic input
  - [ ] analyzer output
  - [ ] selected action
  - [ ] reward decomposition
  - [ ] final outcome
- [ ] Bu log ileride tezde case study olarak kullanılacak.

## Done kriteri

- [ ] LLM katkısı yalnızca “bonus/penalty” değil, açıkça state/action prior üreten modül oluyor.
- [ ] Semantic prior ve confidence grafiklerle gösterilebiliyor.

---

# 6. Faz 4 — Baseline Ailesini Genişlet

## Hedef

Kıyaslamayı ciddi hale getirmek.

## Minimum baseline seti

### Zorunlu baseline’lar

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
- [ ] `PPO_WithSemanticPrior`  ← hedef ana model

### Orta seviye ek baseline’lar

- [ ] `DQN` veya `DoubleDQN`
- [ ] `SAC/DDPG` tarzı continuous ratio policy (opsiyonel ama çok değerli)

## Kod görevleri

- [ ] `src/baselines.py` içine bütün baseline’ları tek arayüz altında topla.
- [ ] Hepsi aynı task stream ve aynı seed ile değerlendirilsin.
- [ ] Sonuçlar ortak evaluator ile ölçülsün.

## Done kriteri

- [ ] “Bizim model iyi” cümlesi en az 8–10 baseline’a karşı desteklenebiliyor.

---

# 7. Faz 5 — Sistematik Ablation Study

## Hedef

Katkının nereden geldiğini göstermek.

## Yapılacak ablation’lar

- [ ] `Full Model`
- [ ] `w/o Semantics`
- [ ] `w/o Reward Shaping`
- [ ] `w/o Semantic Prior`
- [ ] `w/o Confidence`
- [ ] `w/o Partial Offloading` (yalnızca Local/Edge/Cloud)
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

- [ ] Her ana tasarım kararının katkısı tablo ve grafikle ayrıştırılabiliyor.

---

# 8. Faz 6 — Gerçek Veri / Trace-Driven Deney Paketi

## Hedef

Synthetic-only görünümden çıkmak.

## Yapılacaklar

### 8.1. Trace loader

- [ ] `src/trace_loader.py` yaz.
- [ ] Google Cluster Trace için:
  - [ ] task arrival pattern
  - [ ] cpu demand
  - [ ] duration / runtime approximation
- [ ] Didi Gaia için:
  - [ ] mobility trajectory
  - [ ] distance-to-edge profile
  - [ ] handover benzeri hareketlilik etkileri

### 8.2. Trace-to-task mapping

- [ ] Task fields ile trace field’ları net eşleştir:
  - [ ] `size_bits`
  - [ ] `cpu_cycles`
  - [ ] `deadline`
  - [ ] `task_type`
- [ ] Eksik alanlar için açık mapping assumptions dosyası yaz:
  - [ ] `doc/trace_mapping_assumptions.md`

### 8.3. İki deney modu oluştur

- [ ] `synthetic mode`
- [ ] `trace-driven mode`

### 8.4. Domain shift testi

- [ ] Synthetic’te eğit, trace’de test et.
- [ ] Trace’de eğit, synthetic’te test et.
- [ ] Generalization tablosu oluştur.

## Done kriteri

- [ ] Proje artık sadece “güzel simülasyon” değil, trace-driven çalışma haline geliyor.

---

# 9. Faz 7 — Gelişmiş Metrik ve İstatistiksel Analiz

## Hedef

Makale seviyesinde ölçüm disiplini kurmak.

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

## İstatistiksel analiz

- [ ] Her deney en az 5 seed ile koşsun.
- [ ] Ortalama ± std ver.
- [ ] 95% güven aralığı ver.
- [ ] Wilcoxon signed-rank testi uygula.
- [ ] Gerekirse Friedman + Nemenyi ekle.

## Görseller

- [ ] latency boxplot
- [ ] p95 latency bar chart
- [ ] energy-success scatter
- [ ] fairness-jitter comparison
- [ ] QoE violin plot
- [ ] CDF of latency

## Done kriteri

- [ ] Sonuç kısmı artık yalnızca tek sayıdan oluşmuyor; istatistiksel olarak savunulabilir hale geliyor.

---

# 10. Faz 8 — Two-Stage Training (En Güçlü Orta Vadeli Katkı)

## Hedef

AgentVNE’deki staged training disiplinini offloading problemine uyarlamak.

## Yapılacaklar

### 10.1. Oracle / heuristic label üretimi

- [ ] Basit maliyet fonksiyonu ile yarı-optimal karar etiketi üret:
  - [ ] latency-aware oracle
  - [ ] energy-aware oracle
  - [ ] weighted objective oracle
- [ ] Bu etiketleri supervised pretraining için dataset haline getir.

### 10.2. Policy pretraining

- [ ] `src/pretrain_policy.py` oluştur.
- [ ] PPO policy ağını önce imitation / supervised şekilde ısıt.
- [ ] Sonra PPO ile fine-tune et.

### 10.3. Karşılaştırma

- [ ] `PPO from scratch`
- [ ] `Pretrained + PPO`
- [ ] convergence speed
- [ ] sample efficiency
- [ ] final performance

## Done kriteri

- [ ] Eğitim daha stabil hale geliyor.
- [ ] Tezde “neden staged training?” sorusuna net cevap var.

---

# 11. Faz 9 — Graph-Aware Policy Upgrade (Efsane Seviye Katkı)

## Hedef

Projeyi gerçekten sıra dışı hale getirecek ana sıçrama.

## Neden gerekli?

Şu an state büyük ölçüde vektör tabanlı. Oysa sistem doğal olarak graf yapısında:

- cihazlar
- edge sunucular
- bulut
- aralarındaki mesafe / link quality
- queue ilişkileri
- shared congestion
- mobility

Bu yüzden GNN tabanlı policy çok güçlü bir upgrade olur.

## Yapılacaklar

### 11.1. Graph state tanımı

- [ ] Node tipleri:
  - [ ] IoT device
  - [ ] Edge server
  - [ ] Cloud node
  - [ ] Current task node (opsiyonel)
- [ ] Node feature’ları:
  - [ ] battery
  - [ ] cpu capacity
  - [ ] queue load
  - [ ] mobility state
  - [ ] semantic priority
  - [ ] deadline pressure
- [ ] Edge feature’ları:
  - [ ] distance
  - [ ] predicted rate
  - [ ] transmission latency
  - [ ] congestion estimate

### 11.2. Graph encoder

- [ ] `src/graph_policy.py` oluştur.
- [ ] PyTorch Geometric veya DGL ile GNN encoder kur.
- [ ] Çıkışta action logits üret.

### 11.3. Semantic prior fusion

- [ ] LLM prior’ı graph node feature veya policy prior olarak birleştir.
- [ ] Erken fusion ve geç fusion olmak üzere iki strateji test et.

### 11.4. Karşılaştırma

- [ ] `MLP-PPO`
- [ ] `GNN-PPO`
- [ ] `GNN-PPO + Semantic Prior`

## Done kriteri

- [ ] Yeni bir makale katkısı doğuyor.
- [ ] “Neden graph?” sorusuna deneysel cevap verilebiliyor.

---

# 12. Demo ve GUI Yükseltmeleri

## Hedef

Seminerde “wow effect” yaratmak.

## Yapılacaklar

- [ ] GUI’ye “Experiment Mode” ekle:
  - [ ] model seçimi
  - [ ] baseline seçimi
  - [ ] semantic mode aç/kapat
  - [ ] trace mode aç/kapat
- [ ] GUI’de “Ablation comparison panel” ekle.
- [ ] GUI’ye latency CDF mini-chart ekle.
- [ ] GUI’ye decision overhead paneli ekle.
- [ ] Semantic prior dağılımını bar chart olarak göster.
- [ ] Reward decomposition paneli ekle:
  - [ ] latency contribution
  - [ ] energy contribution
  - [ ] deadline contribution
  - [ ] semantic contribution

## Done kriteri

- [ ] Sunum sırasında modelin neden ve nasıl karar verdiği görsel olarak anlatılabiliyor.

---

# 13. Tez ve Makale Üretim Paketi

## Yazılacak yeni dokümanlar

- [ ] `doc/methodology_v2.md`
- [ ] `doc/ablation_plan.md`
- [ ] `doc/evaluation_protocol.md`
- [ ] `doc/trace_mapping_assumptions.md`
- [ ] `doc/threats_to_validity.md`
- [ ] `doc/future_work_graph_rl.md`

## Makale tabloları

- [ ] Baseline comparison table
- [ ] Ablation table
- [ ] Training strategy comparison table
- [ ] Computational overhead table
- [ ] Dataset / trace mapping table

## Makale şekilleri

- [ ] overall architecture figure
- [ ] state-action-reward flowchart
- [ ] semantic prior generation diagram
- [ ] training pipeline figure
- [ ] evaluation pipeline figure
- [ ] CDF / boxplot / scatter seti

## Done kriteri

- [ ] Kod + deney + yazım aynı omurgada birleşiyor.

---

# 14. Haftalık Sıralı Sprint Planı

## Sprint 1 — Temizlik ve deney omurgası

- [ ] klasör yapısı
- [ ] config sistemi
- [ ] logging
- [ ] seed / reproducibility
- [ ] evaluator iskeleti

## Sprint 2 — Ortam hizalama

- [ ] ortak state builder
- [ ] ortak reward module
- [ ] episode redesign
- [ ] simulator-backed RL env

## Sprint 3 — Semantic prior

- [ ] action prior üretimi
- [ ] confidence entegrasyonu
- [ ] structured output
- [ ] explanation logs

## Sprint 4 — Baseline ve ablation

- [ ] baseline engine
- [ ] ablation configs
- [ ] ilk kıyas tabloları

## Sprint 5 — Trace-driven pipeline

- [ ] trace loader
- [ ] synthetic vs trace
- [ ] generalization testleri

## Sprint 6 — Two-stage training

- [ ] oracle label üretimi
- [ ] pretraining
- [ ] PPO fine-tuning

## Sprint 7 — Gelişmiş metrik ve istatistik

- [ ] p95/p99
- [ ] DMR
- [ ] CI ve significance test

## Sprint 8 — Graph RL upgrade

- [ ] graph state
- [ ] GNN encoder
- [ ] MLP vs GNN vs GNN+Semantic

## Sprint 9 — Demo + yazım

- [ ] GUI güçlendirme
- [ ] seminer slaytları
- [ ] makale figürleri

---

# 15. Çok Net Öncelik Listesi (Önce Bunları Yap)

## P0 — Hemen

- [ ] `evaluation.py` yaz
- [ ] `baselines.py` yaz
- [ ] `reward.py` oluştur
- [ ] `state_builder.py` oluştur
- [ ] experiment logging ekle

## P1 — İlk büyük kazanç

- [ ] LLM one-hot yerine action prior vektörü üret
- [ ] `PPO_NoSemantics` ve `PPO_WithSemanticPrior` kıyasını çıkar
- [ ] deadline miss + p95 latency ekle

## P2 — Tez kalitesini sıçratacak iş

- [ ] trace-driven mode
- [ ] staged training
- [ ] ablation study

## P3 — Makale seviyesini sıçratacak iş

- [ ] graph-aware policy
- [ ] statistical testing
- [ ] decision overhead analysis

---

# 16. Riskler ve Dikkat Edilecek Noktalar

- [ ] LLM’yi fazla baskın yapıp PPO’yu gereksiz hale getirme.
- [ ] Reward shaping ile leakage yaratma.
- [ ] Baseline’ları zayıf bırakıp kıyası yapay şekilde kolaylaştırma.
- [ ] Trace mapping varsayımlarını gizleme.
- [ ] GUI çalışıyor diye deney disiplinini ikinci plana atma.
- [ ] Mock eğitim ortamını gerçek sistem gibi sunma.

---

# 17. Definition of Done — “Proje Efsane Oldu” Kontrol Listesi

Aşağıdakilerin çoğu tamamlandıysa proje gerçekten güçlü hale gelmiş demektir:

- [ ] Aynı deneyler tekrarlanabilir.
- [ ] Eğitim ve değerlendirme ortamı hizalı.
- [ ] LLM semantic prior üretiyor.
- [ ] En az 8 baseline var.
- [ ] En az 6 ablation var.
- [ ] Synthetic + trace-driven deney var.
- [ ] p95/p99 latency, DMR, fairness, jitter, QoE raporlanıyor.
- [ ] 5 farklı seed ile CI veriliyor.
- [ ] staged training karşılaştırması var.
- [ ] decision overhead ölçülüyor.
- [ ] GUI’de explanation ve comparison paneli var.
- [ ] Seminer için net mimari şekli hazır.
- [ ] Tez için methodology ve evaluation protocol yazılmış.
- [ ] MLP-PPO ile GNN-PPO karşılaştırması başlamış veya tamamlanmış.

---

# 18. En Kritik Dosya Bazlı Müdahale Planı

## `src/rl_env.py`

- [ ] tek-adımlı episode yapısını kaldır
- [ ] observation’ı semantic prior ile genişlet
- [ ] reward hesaplamasını dış modüle taşı
- [ ] mock reset akışını simulator-backed hale getir

## `src/llm_analyzer.py`

- [ ] structured JSON output
- [ ] action prior üretimi
- [ ] confidence calibration
- [ ] fallback loglama

## `src/simulation_env.py`

- [ ] ortak state/reward entegrasyonu
- [ ] trace-driven task generation
- [ ] karar overhead ölçümü
- [ ] yeni metriklerin toplanması

## `src/train_agent.py`

- [ ] config-based eğitim
- [ ] seed yönetimi
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

Bu planın mantığı şu:

Önce sistemi **bilimsel olarak savunulabilir** hale getir,
sonra **deneysel olarak güçlü** hale getir,
sonra **mimari olarak farklılaştır**,
en sonda **sunum ve makale etkisini büyüt**.

Yani sıra:

**stabilite → kıyas → ablation → trace → staged training → graph RL → demo/makale**

Bu sırayı bozma.
