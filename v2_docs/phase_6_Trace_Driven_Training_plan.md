# Faz 6: Trace-Driven Training Plan

**Tarih:** 2 April 2026  
**Durum:** Baslangic asamasi / belge-kod hizalamasi yapildi  
**Hedef:** Sentetik Faz 5 bulgularini trace-driven ortamda yeniden sinamak ve dogrulamak

---

## Ozet

Faz 5 sonunda sentetik taraf donduruldu ve en kararlı bulgu olarak `mobility_features` etkisi öne çıktı. Buna rağmen sentetik dünyadaki sonuçların genellenebilirliği henüz doğrulanmış değil. Faz 6'nın amacı, aynı karar mekanizmasını trace tabanlı görev akışları üzerinde yeniden eğitmek ve sentetik bulguların trace ortamında korunup korunmadığını test etmektir.

Bu dosya, Faz 6'ya "tamamlanmış bir paket" gibi değil, mevcut repo durumuna göre gerçek başlangıç planı olarak bakar.

---

## Mevcut Kod Durumu

Hazır olan parçalar:
- `experiments/trace/train_ppo.py`
- `configs/trace/ppo_training.yaml`
- `src/core/trace_processor.py`
- `data/traces/` altındaki episode JSON çıktıları

Henüz eksik veya yarım olan parçalar:
- Gerçek trace kaynağını yükleyen ayrı loader yok
- `Success Bonus` entegrasyonu yok
- Partial offloading için switching overhead modeli yok
- Synthetic vs trace domain-shift değerlendirme akışı yok
- Faz 6 kapanışı için doğrulanmış checkpoint ve metrics CSV repo içinde yok

---

## Faz 5'ten Devreden Bulgular

- Reward shaping katkısı tamamen kapanmış bir konu değil; trace tarafında yeniden test edilmeli.
- Partial offloading sentetik tarafta önemli, ancak etkisi algoritmaya göre değişebiliyor.
- Mobility, Faz 5'in en güçlü ve en stabil bileşeni olarak öne çıktı.
- Edge enerji modeli yeni kalibre edildi; trace yüklerinde nasıl davrandığı henüz bilinmiyor.

---

## Faz 6 Hedefleri

### 1. Gerçek trace yükleme akışını netleştirmek
- `src/core/trace_loader.py` implement edildi.
- Trace okuma ile trace-to-task dönüştürme sorumlulukları ayrılacak.
- Mapping varsayımları belgeye bağlanacak.

### 2. Trace reward mantığını tamamlamak
- `Success Bonus` eklenecek.
- Partial offloading için switching overhead modeli tanımlanacak.

### 3. Trace eğitim ve değerlendirme akışını doğrulamak
- PPO trace-driven ortamda yeniden eğitilecek.
- Metrics CSV ve checkpoint track edilen dizinlerde üretilecek.
- Validation tarafında sentetik Faz 5 bulgularının trace karşılığı ölçülecek.

### 4. Domain-shift analizini eklemek
- Synthetic train -> trace test
- Trace train -> synthetic test
- Karşılaştırmalı tablo / kısa yorum

---

## Beklenen Artefaktlar

- Rapor: `phase_reports/Phase_6_Report.md`
- Checkpoint: `models/ppo/trace_training/ppo_v3_trace_best.zip`
- Metrics CSV: `results/raw/trace_training/trace_training_metrics.csv`
- Domain-shift karşılaştırması: Faz 6 raporu içinde veya ek tablo olarak

---

## Basari Kriteri

- Trace eğitim akışı hata vermeden tamamlanmalı.
- Validation metrics repo içinde tekrar üretilebilir olmalı.
- Faz 5'ten gelen en az bir ana bulgu trace tarafında doğrulanmalı veya açıkça çürütülmeli.
- Faz 6 raporu gerçek artefaktlarla güncellenmiş olmalı.

---

## Uygulama Sirasi

1. `task.md` ve Faz 6 belgelerini mevcut gerçekliğe hizala
2. Trace mapping varsayımlarını netleştir
3. Reward içine `Success Bonus` ekle
4. Partial offloading switching overhead modelini ekle
5. Trace PPO eğitimini koştur
6. Domain-shift değerlendirmesini ekle
7. `phase_reports/Phase_6_Report.md` dosyasını gerçek sonuçlarla kapat
