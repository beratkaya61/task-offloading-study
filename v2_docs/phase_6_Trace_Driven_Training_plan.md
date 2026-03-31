# Faz 6: Trace-driven Training - Güncel Plan ve Altyapı

**Tarih:** 31 Mart 2026  
**Durum:** ✅ Altyapı + metrik logging tamamlandı, training koşulacak  
**Hedef:** %68-77 başarı oranı (Faz 5: %62.4 → +%5-15 iyileştirme)

---

## Özet
- Faz 5 bulguları: Reward shaping kritik, partial offloading yüksek etki; semantik özellikler nominal; battery/mobility yeniden ayarlanmalı.
- Faz 6’da trace-driven training + gerçekçi kanal/topoloji + otomatik metrik logging devrede.

---

## Çıktılar
- Phase_6_Report.md (final)
- Checkpoint: `models/ppo_v3_trace_best.zip`
- Metrik CSV: `logs/phase6_trace_training/trace_training_metrics.csv` (success, delay, energy)
- Karşılaştırma tablosu: Faz 5 vs Faz 6

---

## Orchestrator (experiments/run_trace_training.py)
- SB3 PPO ile eğitim; `TraceOffloadingEnv` trace episodlarını besliyor.
- Fiziksel topoloji: `WirelessChannel`, `EdgeServer`, `CloudServer`, `IoTDevice` → datarate/mesafe gerçekçi.
- `TraceMetricsCallback` her epizotta başarı, gecikme, enerji’yi CSV’ye yazıyor.
- Validasyon & ablation: delay/energy topluyor, Faz 5 kritik senaryoları (reward_shaping/partial_offloading) trace üzerinde tekrar doğruluyor.
- Log yolları: CSV `logs/phase6_trace_training/trace_training_metrics.csv`, checkpoint `models/ppo_v3_trace_best.zip`.

---

## Koşum
```bash
cd d:\task-offloading-study
python experiments/run_trace_training.py
# Beklenen süre: 15-30 dk (donanıma bağlı)
```

---

## Başarı Kriteri (Faz 6)
- Validation success rate ≥ %68
- Early stopping tetiklenmeden önce kararlı öğrenme (NaN/Inf yok)
- Reward shaping/partial offloading trace’de de kritik olarak doğrulanır
- Phase_6_Report.md güncellenir + Faz 5 karşılaştırması eklenir

---

## Faz 7 İçin Devreden İçgörüler
- Battery/mobility ağırlıkları yeni metriklerle yeniden kalibre edilecek.
- Semantik katkının düşük görünmesi iz kaynaklı mı anlamak için Faz 6 metrikleri kullanılacak.
- Trace tabanlı sonuçlar staged training (Faz 7) ve GNN (Faz 8) için temel sağlayacak.
