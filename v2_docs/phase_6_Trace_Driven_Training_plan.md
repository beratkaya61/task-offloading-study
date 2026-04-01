# Faz 6: Trace-driven Training - Güncel Plan ve Altyapı

**Tarih:** 1 April 2026  
**Durum:** Altyapı hazır, doğrulanmış artefaktlarla yeniden koşturulacak  
**Hedef:** %68-77 başarı oranı (Faz 5: %62.4 -> +%5-15 iyileştirme)

---

## Özet
- Faz 5 bulguları: Reward shaping kritik, partial offloading yüksek etki; semantik özellikler nominal; battery/mobility yeniden ayarlanmalı.
- Faz 6'da trace-driven training, gerçekçi kanal/topoloji ve otomatik metrik logging devrede.
- Önceki koşunun checkpoint ve metrics artefaktları repo snapshot'ında olmadığı için Faz 6 kapanışı yeniden doğrulanacak.

---

## Çıktılar
- Final report: `phase_reports/Phase_6_Report.md`
- Checkpoint: `models/ppo_v3_trace/ppo_v3_trace_best.zip`
- Metrik CSV: `results/raw/phase6_trace_training/trace_training_metrics.csv`
- Karşılaştırma tablosu: Faz 5 vs Faz 6

---

## Orchestrator (`experiments/run_trace_training.py`)
- SB3 PPO ile eğitim; `TraceOffloadingEnv` trace episode'larını besliyor.
- Fiziksel topoloji: `WirelessChannel`, `EdgeServer`, `CloudServer`, `IoTDevice` ile daha gerçekçi datarate/mesafe akışı.
- `TraceMetricsCallback` her epizotta başarı, gecikme ve enerji ölçümlerini CSV'ye yazıyor.
- Validasyon ve ablation adımı, Faz 5'in reward shaping ve partial offloading bulgularını trace üzerinde yeniden test ediyor.
- Çıktı yolları artık track edilen dizinlerde: CSV `results/raw/phase6_trace_training/trace_training_metrics.csv`, checkpoint `models/ppo_v3_trace/ppo_v3_trace_best.zip`, rapor `phase_reports/Phase_6_Report.md`.

---

## Koşum
```bash
cd d:\task-offloading-study
python experiments/run_trace_training.py
```

---

## Başarı Kriteri
- Validation success rate >= %68
- Eğitim kararlı olmalı; NaN/Inf oluşmamalı
- Reward shaping ve partial offloading trace tarafında da anlamlı fark üretmeli
- `phase_reports/Phase_6_Report.md` gerçek artefaktlarla güncellenmeli

---

## Faz 7 İçin Devreden İçgörüler
- Battery/mobility ağırlıkları yeni metriklerle yeniden kalibre edilecek.
- Semantik katkının düşük görünmesinin trace dağılımından mı geldiği Faz 6 metrikleriyle ayrıştırılacak.
- Trace tabanlı sonuçlar staged training (Faz 7) ve GNN (Faz 8) için temel sağlayacak.
