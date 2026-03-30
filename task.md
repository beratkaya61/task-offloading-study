# AgentVNE-Inspired Task Offloading Upgrade Plan

Based on the [TODO_ANTIGRAVITY_TASK_OFFLOADING_UPGRADE.md](file:///c:/Users/BERAT/Desktop/task-offloading-study/TODO_ANTIGRAVITY_TASK_OFFLOADING_UPGRADE.md) and `AgentVNE`, here is the complete step-by-step Execution Plan. We added **Phase 10: LLM Self-Reflection & Experience Replay** to further improve upon AgentVNE's core concept by using the LLM for dynamic context generation.

## Faz 1 — Reproducibility ve Kod Temizliği
- [/] 1.1 `src/baselines.py`, `src/evaluation.py`, `src/metrics.py`, `src/config.py`, `src/trace_loader.py`, `src/semantic_prior.py`, `src/pretrain_policy.py`, `src/utils/reproducibility.py` dosyalarını (modüllerini) oluştur.
- [ ] 1.2 `configs/` (`train_ppo.yaml`,vb.) ve `results/` klasör yapılarını hazırlama.
- [ ] 1.3 `numpy`, `random`, `torch`, `gymnasium` için tek seed fonksiyonunu yaz.
- [ ] 1.4 JSON/CSV loglama altyapısını kur (run_id, timestamp, metrikler).
- [ ] 1.5 Faz 1'i test et ve Git Commit at (`git commit -m "Faz 1: Reproducibility and Code Hygiene"`).

## Faz 2 — Train Environment ve Gerçek Simülasyon HizalamasıTODO_ANTIGRAVITY_TASK_OFFLOADING_UPGRADE
- [ ] 2.1 Ortak `src/state_builder.py` yazımı.
- [ ] 2.2 `rl_env.py` içinde mock yerine simülatör tabanlı reset işlemleri.
- [ ] 2.3 Reward bileşenlerinin `src/reward.py` altına taşınması.
- [ ] 2.4 Çok adımlı episode tasarımı.
- [ ] 2.5 Faz 2'yi test et ve Git Commit at.

## Faz 3 — LLM Entegrasyonunu Gerçek Katkıya Dönüştür
- [ ] 3.1 Gözlemler (observation) için 6 boyutlu action prior vektörünün LLM ile üretimi.
- [ ] 3.2 Confidence skorunun reward/policy etkisine entegre edilmesi.
- [ ] 3.3 LLM çıktısını parse edilebilir JSON hale getirme ve loglama.
- [ ] 3.4 Faz 3 testi ve Git Commiti.

## Faz 4 — Baseline Ailesini Genişlet
- [ ] 4.1 Tüm baselineların (`LocalOnly`, `EdgeOnly`, `Random`, `Greedy` vb.) tam kodlanması.
- [ ] 4.2 Baselines sonuçlarının `evaluation.py` altında ortak ölçümü.
- [ ] 4.3 Faz 4 testleri ve Git Commiti.

## Faz 5 — Sistematik Ablation Study
- [ ] 5.1 Ablation configlerinin `w/o Semantics`, `w/o Reward Shaping` gibi hazırlanması.
- [ ] 5.2 Toplu ablation deney loglarının alınması.
- [ ] 5.3 Faz 5 testleri ve Git Commiti.

## Faz 6 — Gerçek Veri / Trace-Driven Deney Paketi
- [ ] 6.1 `trace_loader.py` yazımı (örneğin Google/Didi dataset benzeri synthetic veriler veya gerçek data ile).
- [ ] 6.2 Trace to task mapping mantığı.
- [ ] 6.3 Domain shift (Synthetic vs. Trace) fonksiyonlarının kurulumu.
- [ ] 6.4 Faz 6 testleri ve Git Commiti.

## Faz 7 — Two-Stage Training (AgentVNE Concept)
- [ ] 7.1 Oracle / heuristic label üretimi.
- [ ] 7.2 PPO ağını imitation / supervised mode ile önceden eğitme (`pretrain_policy.py`).
- [ ] 7.3 Fine-tune PPO vs PPO from scratch sonuç karşılaştırması.
- [ ] 7.4 Faz 7 testleri ve Git Commiti.

## Faz 8 — Graph-Aware Policy Upgrade (AgentVNE Concept)
- [ ] 8.1 Graph state node ve edge feature'larının tanımı (IoT Device, Edge, vb.).
- [ ] 8.2 PyTorch Geo. / DGL tabanlı GNN logic (`graph_policy.py`).
- [ ] 8.3 Semantic prior fusion entegrasyonu.
- [ ] 8.4 Faz 8 testleri ve Git Commiti.

## Faz 9 — Gelişmiş Metrik ve İstatistiksel Analiz / GUI
- [ ] 9.1 Gelişmiş metrik ölçümlerinin (Jitter, Fairness, p95) GUI 'ye bağlanması.
- [ ] 9.2 GUI'de Result / Ablation / Reward Decomposition panellerinin aktivasyonu.
- [ ] 9.3 Faz 9 testleri ve Git Commiti.

## Faz 10 (Ekstra) — LLM Self-Reflection & Experience Replay
- [ ] 10.1 Ajan hatalı ya da düşük utility kararlar aldığında LLM'den post-hoc analiz (Self-reflection) isteme.
- [ ] 10.2 Elde edilen açıklamaların (explanation logs) bir Experience Buffer'da (veya Semantic Bank'da) biriktirilerek bir sonraki `state` prompt'larında RAG benzeri dinamik few-shot örnekleri olarak sağlanması.
- [ ] 10.3 Final test ve Git Commiti.
