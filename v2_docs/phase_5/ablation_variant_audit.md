# Phase 5 Ablation Variant Audit

Bu dokuman, Faz 5 ablation study kapsaminda hangi varyantlarin incelendigini, bu varyantlarin hangi bilesenleri kapatmayi hedefledigini ve mevcut kod/rapor okumasinda dikkat edilmesi gereken metodolojik notlari toplar.

## Neden Bu Audit Gerekli

Faz 5'te full modelden belirli bilesenler cikarilarak performans etkisi olculdu. Ancak bazi varyantlarda success rate'in Full Model ile birebir ayni gorunmesi ilk bakista supheli duruyor.

Bu suphe haklidir. Cunku mevcut implementasyonda bazi ablation flag'leri tam izole edici sekilde uygulanmamis olabilir. Bu nedenle Faz 5 sonuclari su sekilde okunmalidir:

- `partial_offloading` ve `mobility_features` sonuclari daha guclu sinyal verir.
- `semantics`, `semantic_prior`, `confidence`, `queue_awareness` gibi bilesenlerde sifira yakin fark, bilesenin etkisiz oldugunu kesin olarak gostermez.
- Bazi varyantlar ayni aksiyon davranisini korudugu icin success/latency/energy metrikleri ayni kalabilir.
- Bazi flag'ler state, reward ve environment dynamics tarafinda ayni anda kapanmadigi icin tam faktoriyel ablation sayilmayabilir.

## Incelenen Varyantlar

Kanonik config:

- `configs/synthetic/ablation.yaml`

Deney scripti:

- `experiments/synthetic/run_ablation_study.py`

Ham CSV ciktilari:

- `results/raw/synthetic/ablation/synthetic_ablation_ppo_multi_seed_evaluation.csv`
- `results/raw/synthetic/ablation/synthetic_ablation_ppo_multi_seed_retraining.csv`
- `results/raw/synthetic/ablation/synthetic_ablation_dqn_multi_seed_evaluation.csv`
- `results/raw/synthetic/ablation/synthetic_ablation_dqn_multi_seed_retraining.csv`
- `results/raw/synthetic/ablation/synthetic_ablation_a2c_multi_seed_evaluation.csv`
- `results/raw/synthetic/ablation/synthetic_ablation_a2c_multi_seed_retraining.csv`

Uretilen figure'lar:

- `results/figures/synthetic/ablation/`

Kanonik ozet rapor:

- `v2_docs/phase_5/offloading_experiment_report.md`
- `phase_reports/Phase_5_Report.md`

## Varyant Listesi

| Variant | Hedeflenen kapatma | Config seviyesindeki anlam |
|---|---|---|
| `full_model` | Hicbir bilesen kapali degil | Tum semantic, physical ve partial offloading bilesenleri acik |
| `w_o_semantics` | LLM semantic analysis kapali | `semantics=false`, `semantic_prior=false`, `confidence_weighting=false` |
| `w_o_reward_shaping` | Reward shaping kapali | Semantic/structured reward yerine daha basit delay-energy ceza fonksiyonu |
| `w_o_semantic_prior` | Semantic action prior kapali | Policy input tarafinda semantic prior etkisini kapatma hedefi |
| `w_o_confidence` | Confidence weighting kapali | LLM guven skorunun etkisini kapatma hedefi |
| `w_o_partial_offloading` | Partial offloading kapali | Aksiyon uzayi Local/Edge/Cloud davranisina indirgenir |
| `w_o_battery_awareness` | Battery awareness kapali | Battery drain/state etkisini kapatma hedefi |
| `w_o_queue_awareness` | Queue awareness kapali | Edge/cloud queue bilgisinin etkisini kapatma hedefi |
| `w_o_mobility_features` | Mobility/distance features kapali | Cihaz hareketi/link-quality etkisini kapatma hedefi |

## Kod Seviyesinde Dikkat Edilecek Noktalar

### 1. `w_o_semantics` tam semantik kapatma olmayabilir

`OffloadingEnv._get_obs()` icinde `disable_semantics=true` oldugunda state'in semantic prior bolumu sifirlaniyor:

```text
full_state[6:12] = 0.0
```

Ancak `calculate_reward(...)` fonksiyonu halen `task.semantic_analysis` uzerinden `llm_rec`, `confidence` ve `priority_score` okuyabiliyor. Bu nedenle `w_o_semantics`, mevcut haliyle:

```text
semantic input/prior kapali, fakat reward tarafinda semantic_analysis tamamen koparilmis olmayabilir
```

seklinde okunmalidir.

Bu nedenle `w_o_semantics` sonucunun Full Model'e yakin cikmasi, semantik bilginin tamamen etkisiz oldugunu kanitlamaz.

### 2. `w_o_semantic_prior` flag'i state builder tarafinda ayri uygulanmamis olabilir

`state_builder.py` icinde prior uretimi su kosula baglidir:

```text
if not disable_semantics:
    generate_action_prior(...)
```

Burada `disable_semantic_prior` bayragi ayrica kontrol edilmemektedir. Bu nedenle `w_o_semantic_prior`, `semantics=true` kaldigi surece state tarafinda Full Model ile ayni prior'i uretmis olabilir.

Bu, `w_o_semantic_prior` sonucunun Full Model ile birebir ayni cikmasinin ana aciklamalarindan biridir.

### 3. `w_o_confidence` flag'i semantic prior fonksiyonuna aktarilmiyor

`generate_action_prior(...)` fonksiyonu semantic analysis icindeki `confidence` degerini kullanir. Ancak `disable_confidence_weighting` flag'i bu fonksiyona gecirilmedigi icin confidence etkisi tam olarak kapatilmamis olabilir.

Bu nedenle `w_o_confidence` varyanti da tam izole edilmis bir confidence ablation olarak okunmamalidir.

### 4. `w_o_queue_awareness` state/dynamics tarafinda tam ayrismiyor olabilir

`disable_queue_awareness` flag'i config seviyesinde tanimli olsa da `rl_env.py` icinde queue/load gecikme dinamiklerinin bu flag ile tamamen kapatildigi gorunmemektedir.

Bu nedenle `w_o_queue_awareness` sonucunun Full Model'e yakin cikmasi beklenebilir.

### 5. `w_o_partial_offloading` ve `w_o_mobility_features` daha guvenilir sinyal verir

Bu iki varyant kodda daha dogrudan davranis degistirir:

- `disable_partial_offloading` aksiyon secimini Local/Edge/Cloud ailesine indirger.
- `disable_mobility_features` link-quality cezasini ve cihaz konum guncellemesini etkiler.

Bu nedenle Faz 5'te en guvenilir yorum:

```text
partial_offloading ve mobility_features bilesenleri performans acisindan kritik gorunmektedir.
```

seklinde kurulmalidir.

## Neden Semantics ve Reward Shaping Etkisi Zayif Gorunuyor

Bunun birden fazla nedeni olabilir:

1. Policy zaten `action=3` etrafinda dominant davranisa cokmustur. Aksiyon dagilimi degismeyince success/latency/energy ayni kalabilir.
2. Evaluation-only ablation, egitilmis policy'nin test anindaki hassasiyetini olcer; bilesenin egitim sirasindaki katkisini tam olcmez.
3. `w_o_semantics`, `w_o_semantic_prior` ve `w_o_confidence` flag'leri state/reward hattinda tam izole edilmemis olabilir.
4. Reward shaping kapandiginda success ayni kalabilir, fakat `metric_avg_reward` ciddi degisir. Bu, reward fonksiyonunun degistigini ama policy aksiyon davranisinin ayni kaldigini gosterir.
5. Mevcut ortamda fiziksel sinyaller, semantic sinyale gore daha baskin kalmis olabilir.

## Faz 5 Icin Guvenli Akademik Yorum

Faz 5 sonuclari su sekilde yazilmalidir:

> Ablation study sonucunda partial offloading ve mobility features bilesenleri en kararli performans etkisini gostermistir. Semantic prior, confidence weighting ve reward shaping varyantlarinda success rate farklari sinirli kalmistir; ancak mevcut implementasyonda bu bilesenlerin tam izole edilmemis olma ihtimali nedeniyle bu sonuc "semantik bilesenler etkisizdir" seklinde degil, "mevcut protokolde semantik katkisi yeterince ayrismamistir" seklinde yorumlanmalidir.

## Journal Icin Tavsiye Edilen Duzeltme

Journal taslaginda ablation bolumu su ifadeden kacinmalidir:

```text
Semantic bileşenler katkı sağlamadı.
```

Bunun yerine su ifade kullanilmalidir:

```text
Semantic ve reward-shaping varyantlari mevcut Faz 5 protokolunde success rate uzerinde sinirli ayrisma gostermistir. Bu durum, bilesenlerin etkisiz oldugunu kesin olarak gostermemekte; daha temiz izole edilmis ablation, reward decomposition ve trace-driven yeniden test gerektirmektedir.
```

## Sonraki Teknik Duzeltme Onerisi

Faz 9 veya journal-oncesi tekrar icin ablation flag'leri su sekilde netlestirilmelidir:

1. `disable_semantics`: semantic_analysis tamamen bos/neutral hale getirilmeli.
2. `disable_semantic_prior`: state prior bolumu uniform veya zero yapilmali, ancak semantic reward istenirse acik kalabilmeli.
3. `disable_confidence_weighting`: `confidence=1.0` veya sabit neutral degerle override edilmeli.
4. `disable_reward_shaping`: reward sadece physical delay-energy-deadline bilesenlerine indirgenmeli.
5. `disable_queue_awareness`: state ve delay dynamics tarafindaki queue etkisi birlikte kapatilmali.
6. `disable_battery_awareness`: state, reward ve battery drain etkileri ayrik olarak kontrol edilmeli.

Bu duzeltmelerden sonra Faz 5 ablation study daha guclu bir journal tablosuna donusturulebilir.
