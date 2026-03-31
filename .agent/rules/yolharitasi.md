# 🗺️ Agent Kod Yol Haritası & Kuralları

## 📌 KRİTİK KURAL: GIT COMMIT

### ⚠️ ASLA OTOMATIK COMMIT ATMA!
```
❌ YAPMA: git commit -m "..." 
✅ YAP: Kullanıcıya rapor ver, kullanıcı manuel commit atsın
```

**Neden?**
- Kullanıcı commit history'yi kontrol etmek istiyor
- Her commit anlamlı ve revertilebilir olmalı
- Çalışma akışı: Plan → Kod → Test → Kullanıcı Onayı → Commit

**Prosedür:**
1. Değişiklikleri yap (`git add -A` yapma!)
2. Değişiklikleri özetle (rapor yaz)
3. Kullanıcıya sor: "Değişiklikleri commit etmek istersen şunu çalıştır:"
4. Komutu ver: `git add -A && git commit -m "..."`
5. Kullanıcı manuel olarak çalıştırsın

---

## 🎯 Faz Yönetimi Kuralları

### Her Faz Sonunda:
- [ ] TODO_ANTIGRAVITY_TASK_OFFLOADING_UPGRADE.md kontrol et (plan uygun mu?)
- [ ] task.md kontrol et (adımlar tamamlandı mı?)
- [ ] Phase_X_Report.md detailed şekilde yaz
- [ ] Tüm output'ları results/ klasöründe sakla
- [ ] Git status kontrol et (tüm dosyalar staged mi?)
- [ ] Kullanıcıya commit komutu sun (manuel atması için)

### Her Kod Değişikliğinde:
- [ ] Tek bir sorumluluğu yapan, minimal kod
- [ ] Test et (çalışıyor mı? Hata var mı?)
- [ ] Docstring/comments ekle (neden yaptık?)
- [ ] Backward compatibility kontrol et

---

## 📊 Faz Kontrolü Matrisi

| Faz | Durum | Başarı Hedefi | Report Dosyası |
|-----|-------|---------------|-----------------|
| Faz 1 | ✅ Tamamlandı | Reproducibility + Config setup | Phase_1_Report.md |
| Faz 2 | ✅ Tamamlandı | Train/Sim hizalaması | Phase_2_Report.md |
| Faz 3 | ✅ Tamamlandı | LLM semantic integration | Phase_3_Report.md |
| Faz 4 | ✅ Tamamlandı | %62.67 PPO_v2 başarısı | Phase_4_Report.md |
| Faz 5 | 🟡 Hazırlanıyor | Ablation study → %65-70 hedefi | Phase_5_Report.md (YAPILACAK) |
| Faz 6 | ⬜ Planlı | Trace-driven → %68-77 hedefi | Phase_6_Report.md (YAPILACAK) |
| Faz 7 | ⬜ Planlı | Two-stage training → %73-85 hedefi | Phase_7_Report.md (YAPILACAK) |
| Faz 8 | ⬜ Planlı | Graph NN → %78-95 hedefi | Phase_8_Report.md (YAPILACAK) |
| Faz 9 | ⬜ Planlı | Advanced metrics + GUI | Phase_9_Report.md (YAPILACAK) |
| Faz 10 | ⬜ Planlı | AgentVNE comparison + Paper ready | Phase_10_Report.md (YAPILACAK) |

---

## 🔍 Kod Kalitesi Standartları

### Python Kodunda:
- Docstring: GoogleStyle veya NumPy style
- Type hints: `def function(x: np.ndarray) -> float:`
- Error handling: Try/except ile detaylı logging
- Comments: Kompleks logic'i açıkla

### Config Dosyalarında:
- YAML formatı (JSON değil)
- Comments ile açıklama
- Default values her zaman var
- Validation logic yaz

### Markdown Dokumentasyon:
- Title: # ile başla
- Sections: ## ile strukturelandır
- Tables: Markdown table kullan
- Code blocks: ``` ile işaretle (language belirt)
- Links: Diğer dosyalara link ver

---

## 📂 Dosya Organizasyonu

```
task-offloading-study/
├── .agent/rules/
│   └── yolharitasi.md         ← Bu dosya
├── configs/
│   ├── train_ppo.yaml         ✅ Var
│   ├── eval_default.yaml      ✅ Var
│   ├── baselines.yaml         ✅ Var
│   └── ablation.yaml          ✅ Var
├── docs/
│   ├── baseline_literature_analysis.md    ✅ Yeni
│   ├── phase_5_ablation_plan.md           ✅ Var
│   ├── implementation_plan.md             ✅ Var
│   └── agentvne_comparison.md             ⚠️ YAPILACAK
├── phase_reports/
│   ├── Phase_1_Report.md      ✅ Var
│   ├── Phase_2_Report.md      ✅ Var
│   ├── Phase_3_Report.md      ✅ Var
│   ├── Phase_4_Report.md      ✅ Tamamlandı
│   └── Phase_5_Report.md      ⚠️ YAPILACAK (Faz 5 sonunda)
├── results/
│   ├── raw/
│   │   ├── master_experiments.csv      ✅ Var
│   │   └── ablation_experiments.csv    ⚠️ YAPILACAK (Faz 5)
│   ├── tables/
│   │   ├── summary.md                  ✅ Var
│   │   └── ablation_comparison.md      ⚠️ YAPILACAK (Faz 5)
│   └── figures/
│       └── ablation_impact.png         ⚠️ YAPILACAK (Faz 5)
└── src/
    ├── agents/
    │   ├── baselines.py        ✅ Var (7 baseline)
    │   ├── llm_analyzer.py     ✅ Var (TinyLlama)
    │   └── semantic_prior.py   ✅ Var
    ├── core/
    │   ├── evaluation.py       ✅ Hazırlandı (SB3 fix)
    │   ├── metrics.py          ⚠️ YAPILACAK (Faz 5+)
    │   └── reward.py           ✅ Var
    └── env/
        ├── rl_env.py           ✅ Var (11 boyutlu state)
        └── state_builder.py    ✅ Var

```

---

## 🚀 Faz 5 Operasyon Planı

### Hedef: Ablation Study (Hangi Bileşen Ne Kadar Katkı Sağlıyor?)

**Yapılacaklar (Sıralı):**

1. **Environment'ı Modifiye Et**
   - Dosya: `src/env/rl_env.py`
   - Eklenecek: Ablation flags (`disable_semantics`, `disable_reward_shaping`, etc.)
   - Test: Her ablation mode'da test et

2. **Ablation Runner Scripti Yaz**
   - Dosya: `experiments/run_ablation_study.py` (YENİ)
   - Koş: 9 ablation × 10 episode = 90 run
   - Kaydet: `results/raw/ablation_experiments.csv`

3. **Metrikleri Genişlet**
   - Dosya: `src/core/metrics.py` (YENİ)
   - Ekle: avg_latency, p95_latency, deadline_miss, energy, fairness, jitter, qoe

4. **Karşılaştırma Raporu Yaz**
   - Dosya: `results/tables/ablation_comparison.md` (YAPILACAK)
   - Tablo: 9 ablation × 9 metrik
   - Chart: matplotlib ile impact graph'ları

5. **Faz 5 Raporu Yaz**
   - Dosya: `phase_reports/Phase_5_Report.md` (YAPILACAK)
   - İçerik: Detaylı bulgular, istatistiksel testler, sonuçlar

---

## 📋 Kontrol Listesi (Her Faz Sonunda)

### Pre-Commit Kontrolleri:
- [ ] Bütün testler geçti mi? (`run_baselines.py`, `run_ablation_study.py`)
- [ ] Bütün output'lar `results/` altında mı?
- [ ] Bütün docstring'ler yazılı mı?
- [ ] README güncel mi?
- [ ] Phase_X_Report.md tamamlandı mı?

### Git Kontrolleri:
- [ ] `git status` → Tüm değişiklikler görülüyor mü?
- [ ] `git diff` → Değişiklikler mantıklı mı?
- [ ] Commit message açık mı? (Başında "Faz X: ...")
- [ ] Author info doğru mu? (Name, Email)

### Teknik Kontrolleri:
- [ ] Kod hataları yok mu? (flake8, pylint okuyabilirim)
- [ ] Backward compatibility OK mi? (Eski scripts çalışıyor mu?)
- [ ] Performance regression yok mu? (Yavaşlamadı mı?)

---

## 🔗 Bağlantılar & Referanslar

### Yol Haritası Belgeleri:
- [TODO_ANTIGRAVITY_TASK_OFFLOADING_UPGRADE.md](../TODO_ANTIGRAVITY_TASK_OFFLOADING_UPGRADE.md)
- [task.md](../task.md)
- [Phase Reports](../phase_reports/)

### Baseline Analizi:
- [baseline_literature_analysis.md](../docs/baseline_literature_analysis.md) ← YENI

### Faz Planları:
- [phase_5_ablation_plan.md](../docs/phase_5_ablation_plan.md)

### Konfigürasyon:
- [configs/ablation.yaml](../../configs/ablation.yaml)

---

## ⚠️ Yaygın Hatalar (Kaçılması Gereken)

| Hata | Çözüm |
|------|-------|
| Otomatik commit atmak | ❌ YAPMA! Kullanıcı manuel atsın |
| TODO/task.md'i güncellememek | ✅ Her faz sonunda güncelle |
| Phase_Report yazılmaması | ✅ Detaylı rapor her faz sonunda |
| Eksik docstring | ✅ Her function'a docstring yaz |
| Git history karmaşası | ✅ Atomic commit'ler (bir işi yapar) |
| Başarı hedeflerini kaçırmak | ✅ Başta hedef belirle, sonunda raporla |

---

## 📞 İletişim Kuralları

### Kullanıcı ile Konuşma:
- Kararlar net sunmam
- Her faz sonunda özet rapor
- Herhangi hata varsa hemen rapor et
- Sorular varsa değil cevapları vereceğim, seçenekleri sunacağım

### Commit Mesajları:
```
Format:
[Faz X] Kısa açıklama (imperative form)

Örnek:
✅ [Faz 4] Fix PPO_v2 batch dimension in evaluation.py
✅ [Faz 5] Add ablation study configuration and runner script
✅ [Faz 6] Implement trace-driven training data loader
```

---

## 🎓 Son Notlar

Bu yol haritası değişebilir! Eğer:
- Yazılım mimarisi değişirse → Yeni version yaz
- Faz planında değişiklik olursa → TODO_ANTIGRAVITY güncelle
- Standartlar değişirse → Bu dosyayı güncelle

**Kısaca:** Esneklik ve netlik birlikte!

---

## 🏷️ v1_docs / v2_docs Dökümantasyon Kuralı

- **v1_docs:** AgentVNE makalesi ÖNCESİ yapılan tüm geliştirme, test ve dokümantasyonlar burada tutulur. (İlk yol haritası, eski mimari, eski metrikler)
- **v2_docs:** AgentVNE makalesi incelendikten SONRA, yeni yol haritası ve yeni mimari ile yapılan tüm geliştirme, test ve dokümantasyonlar burada tutulur. (Güncel task.md, TODO_ANTIGRAVITY_TASK_OFFLOADING_UPGRADE.md, yeni fazlar, yeni metrikler)
- **Kural:**
  - v1'de eksik/geliştirilecek yerler v2'de odaklanılarak düzeltilir.
  - Dökümantasyon, rapor ve kod yorumlarında bu ayrım açıkça belirtilir.
  - v2_docs, projenin asıl referans ve yayınlanacak sürümüdür.
- **Amaç:** Gelişim sürecinin şeffaf ve izlenebilir olması, literatürle uyumlu ilerlenmesi.
