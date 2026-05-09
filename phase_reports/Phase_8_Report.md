Bkz. ortak kavram sozlugu: v2_docs/project_concepts_glossary.md

# Phase 8 Report: Graph-Aware Policy Upgrade

**Durum:** Faz 8 teknik olarak tamamlandi  
**Baslangic tarihi:** 5 May 2026  
**Guncel tamamlanan adim:** 8.4 Final policy comparison

---

## 2026-05-09 Real-Data Kapsam Notu

Bu rapordaki Faz 8 sonuclari graph-aware policy mimarisinin, semantic prior fusion hattinin ve MLP-vs-Graph karsilastirma evaluator'unun calistigini gosterir.
Ancak bu rapordaki final sayilar henuz raw real-data backbone uzerinde yeniden kosulmus sonuclar degildir.

Dogru etiket:

```text
synthetic/simulation-stage graph comparison
```

Bu nedenle Faz 8'in bilimsel cikti dili su sekilde sinirlandirilmalidir:

- Graph-aware temsil ve evaluator koprusu teknik olarak dogrulandi.
- Semantic prior supervised graph warm-start tarafinda faydali gorundu.
- Final env karsilastirmasinda semantic prior katkisi henuz net ustunluge donusmedi.
- Bu bulgular raw real-data validated sonuc olarak sunulmayacak.

Nihai tez/makale iddiasi icin Faz 6R kapsaminda hazirlanacak real-data omurgasi uzerinde bu karsilastirma yeniden kosulmalidir.

---

## Faz 8 Hedefi

Faz 8'in hedefi, mevcut 12 boyutlu vektor state ile calisan MLP tabanli policy hattini graph-aware bir karar mekanizmasina tasimaktir.

Faz 8 oncesi en guclu neural baseline:

- semantic prior destekli ve staged-training ile isitilmis `Pretrained MLP-PPO`

Faz 8 boyunca ana teknik soru:

> Graph-aware policy, device-task-edge-cloud iliskilerini acikca temsil ederek karar kalitesini ve action diversity'yi iyilestirebilir mi?

---

## 8.1 Graph State Node and Edge Features

### Yapilanlar

Yeni kod:

- `src/env/graph_state_builder.py`

Yeni test:

- `tests/phase_8/test_graph_state_builder.py`

Yeni gorsellestirme:

- `src/visualization/graph_state_visualizer.py`
- `experiments/phase_8/visualize_graph_state.py`
- `results/phase_8/figures/sample_graph_state.png`
- `results/phase_8/figures/sample_graph_state_detailed.png`

Yeni destek dokumani:

- `v2_docs/phase_8/phase_8_explaination_of_studies.md`

Bu adimda `GraphState` dataclass'i ve `build_graph_state(...)` fonksiyonu eklendi.
Builder framework-neutral tasarlandi; yani PyTorch Geometric kurulu olmasa bile graph observation uretimi ve testleri calisiyor.

### GraphState Cikti Alanlari

`GraphState` su alanlari tasir:

- `node_features`
- `edge_index`
- `edge_features`
- `global_features`
- `action_prior`
- `action_mask`
- `node_type_ids`
- `edge_type_ids`
- `node_id_map`
- `metadata`
- `vector_state_reference`

Bu sozlesme Faz 8.2'de GNN policy forward path'e, Faz 8.3'te semantic prior fusion'a ve Faz 9'da gelismis analizlere acik olacak sekilde kuruldu.

### Ilk Topology

Ilk graph topology:

- 1 device node
- 1 task node
- N edge server node
- 1 cloud node

Directed edge aileleri:

- device-edge
- task-edge
- device-cloud
- task-device
- task-cloud

### Test Sonucu

Calistirilan komut:

```powershell
python -m unittest tests.phase_8.test_graph_state_builder
```

Sonuc:

```text
Ran 3 tests in 1.688s
OK
```

Testlerin kapsami:

- node feature shape
- edge index shape
- edge feature shape
- global feature shape
- action prior normalization
- action mask behavior
- semantic ablation behavior
- NaN/Inf kontrolu
- metadata ve trace context tasinmasi
- graph visualization PNG yazimi

### Graph Visualization Kontrolu

Calistirilan komut:

```powershell
python experiments\phase_8\visualize_graph_state.py
```

Sonuc:

```text
[OK] GraphState visualization written to results\phase_8\figures\sample_graph_state.png
```

Ek olarak edge label'lari acik detayli gorsel de uretildi:

```powershell
python experiments\phase_8\visualize_graph_state.py --show-edge-labels --output results\phase_8\figures\sample_graph_state_detailed.png
```

Bu gorsel, GNN policy'ye gecmeden once graph topology'nin insan tarafindan okunabilir hale gelmesini saglar.

### Repo Hijyen Notu

`.gitignore` icindeki `ENV/` kurali Windows ortaminda `src/env/` altindaki yeni dosyalari da ignore edebildigi icin guvenli bir istisna eklendi:

- `!src/env/`
- `!src/env/*.py`

Bu sayede `src/env` altindaki Python kaynak kodlari commit adaylari arasinda gorunur kalir.

## 8.2 GNN Policy Implementation

### Yapilanlar

Yeni kod:

- `src/agents/graph_policy.py`

Yeni test:

- `tests/phase_8/test_graph_policy.py`

Bu adimda PyTorch-only ilk graph-aware policy forward path kuruldu.
PyTorch Geometric mevcut ortamda kurulu olmayabilecegi icin ilk model PyG'e bagimli degil.

### Model Ciktisi

`GraphPolicyNetwork`, `GraphState` girdisinden su ciktilari uretir:

- `logits`
- `masked_logits`
- `action_probabilities`
- `graph_embedding`

Bu sayede model henuz egitilmemis olsa bile su temel yol dogrulandi:

```text
GraphState -> message passing -> graph embedding -> 6 action logits
```

### Teknik Not

Ilk modelde:

- node feature encoder
- edge feature encoder
- PyTorch-only message passing
- mean graph pooling
- global feature encoder
- action mask
- optional late semantic prior logit fusion

bulunur.

Semantic prior fusion yardimci fonksiyonu eklendi, ancak Faz 8.3 altinda bunun deneysel etkisi ayrica ele alinacak.

### Test Sonucu

Calistirilan komut:

```powershell
python -m unittest tests.phase_8.test_graph_state_builder tests.phase_8.test_graph_policy
```

Sonuc:

```text
Ran 7 tests in 1.836s
OK
```

Testlerin kapsami:

- graph policy logits shape
- masked logits shape
- action probability normalization
- graph embedding shape
- action mask'in partial aksiyonlari kapatmasi
- deterministic predict davranisi
- `GraphState` tensor donusumu
- visualization smoke testinin devam etmesi

---

## 8.2 Sonunda 8.3'e Devredilenler

Siradaki adim:

- semantic prior fusion stratejilerini net ayirmak
- `none`, `late`, ileride `early/node-feature fusion` varyantlarini karsilastirmak
- teacher-label dataset ile supervised graph warm-start icin hazirlik yapmak

Faz 8.2 sonunda kabul edilen sonuc:

> Bir offloading karar ani artik graph olarak temsil edilebiliyor ve bu graph, ilk PyTorch-only graph-aware policy tarafindan 6 aksiyon logit'ine donusturulebiliyor.

## 8.3 Semantic Prior Fusion

### Yapilanlar

Yeni kod:

- `src/training/pretrain_graph_policy.py`
- `experiments/phase_8/run_graph_supervised_pretraining.py`
- `experiments/phase_8/run_graph_fusion_comparison.py`
- `configs/phase_8/graph_supervised_pretraining.yaml`

Guncellenen kod:

- `src/agents/graph_policy.py`
- `tests/phase_8/test_graph_policy.py`

Bu adimda semantic prior fusion deneysel olarak ayrilabilir hale getirildi.
Onemli duzeltme: `semantic_prior_fusion="none"` secildiginde semantic prior artik policy head'e gizli sekilde girmiyor.
Boylece `none` varyanti gercek kontrol grubu olarak kullanilabiliyor.

Desteklenen fusion modlari:

- `none`: graph policy semantic prior kullanmadan karar skoru uretir.
- `input`: semantic prior policy head girisine eklenir.
- `late`: graph policy once kendi skorlarini uretir, semantic prior son logit seviyesinde yumusak bias olarak eklenir.
- `input_late`: semantic prior hem giriste hem son logit seviyesinde kullanilir.

### Supervised Graph Warm-Start

Graph policy icin Faz 7 teacher disiplinine uyumlu supervised warm-start hatti eklendi.
Bu hat, synthetic environment icinde graph state uretir ve `teacher_contextual_reward_aligned` oracle kararlarini hedef etiket olarak kullanir.

Akis:

```text
OffloadingEnv decision state -> GraphState -> GraphPolicyNetwork -> teacher action imitation
```

Bu asama RL fine-tuning degildir.
Amaci, graph-aware policy'nin teacher kararlarini taklit edip edemedigini ve semantic prior fusion'in action diversity uzerindeki ilk etkisini olcmektir.

### Smoke Fusion Comparison

Calistirilan komut:

```powershell
python experiments\phase_8\run_graph_fusion_comparison.py --fusions none late
```

Kanonik cikarimlar:

- Faz 8 aciklama ve not dokumani: `v2_docs/phase_8/phase_8_explaination_of_studies.md`
- Graph warm-start config: `configs/phase_8/graph_supervised_pretraining.yaml`
- Graph warm-start scriptleri: `src/training/pretrain_graph_policy.py`, `experiments/phase_8/run_graph_supervised_pretraining.py`, `experiments/phase_8/run_graph_fusion_comparison.py`
- Graph checkpoint'leri calistirildiginda kanonik olarak `models/phase_8/` altina yazilir

Rapor hijyen karari:

- Per-run/per-fusion aciklama raporlari Faz 8 klasorunde tutulmayacak.
- Fusion karsilastirmasi ve kapanis hikayesi ana aciklama dosyasinda izlenecek: `v2_docs/phase_8/phase_8_explaination_of_studies.md`.
- Tekil debug kosulari icin ayrica markdown raporu uretilmeyecek; normal Faz 8 takibi ana aciklama dokumani uzerinden yapilacak.
- Varsayilan akista Faz 8 icin per-run debug CSV klasoru zorunlu tutulmayacak; yalnizca acikca `--write_csv` istendiginde `results/phase_8/metrics/` altina olcum yazilacak.
- Ham CSV ciktisi yalnizca ozellikle `--write_csv` verilirse debug amacli uretilir; normal Faz 8 takibi tek markdown raporu uzerinden yapilir.

Ilk smoke protokolu:

| Fusion | Samples | Best Val Acc | Test Acc | Test Prediction Diversity |
|---|---:|---:|---:|---:|
| `none` | 432 | 72.92% | 63.54% | 0.0000 |
| `late` | 432 | 72.92% | 66.67% | 0.3661 |

Ilk yorum:

- `none` fusion test accuracy acisindan makul basladi, ancak prediction diversity sifira yakin oldugu icin tek aksiyon yigilmasi riski tasiyor.
- `late` fusion test accuracy'yi artirdi ve prediction diversity'yi anlamli bicimde yukseltti.
- Bu sonuc, semantic prior'in graph policy icinde son karar skoruna kontrollu bias olarak eklenmesinin Faz 7'den kalan action diversity problemini azaltabilecegine dair ilk sinyaldir.
- Bu tablo final bilimsel sonuc degildir; sadece hat dogrulama ve ilk sinyal icin smoke/diagnostic protokoldur.

### Profesyonel Fusion Deney Protokolu

Kullanim amaci:

- Smoke testten gelen sinyali bilimsel olarak daha savunulabilir hale getirmek.
- Tek seed / 4 epoch sonucuna dayanarak iddia kurmamak.
- Fusion etkisini 5 seed uzerinden mean/std/95% CI ile raporlamak.

Yeni full config:

- `configs/phase_8/graph_supervised_pretraining.yaml`

Full protokol:

- 60 episode
- 50 step
- 30 epoch
- minimum 12 epoch before early stopping
- early stopping patience 8
- 5 seed: `42, 43, 44, 45, 46`
- tek konsolide markdown raporu

Calistirilacak komut:

```powershell
python experiments\phase_8\run_graph_fusion_comparison.py --config configs\phase_8\graph_supervised_pretraining.yaml --fusions none late --seeds 42 43 44 45 46
```

Rapor hijyeni:

- Per-seed veya per-fusion aciklama raporlari uretilmez.
- Sonuclar ana aciklama dokumaninda yorumlanir: `v2_docs/phase_8/phase_8_explaination_of_studies.md`

### Test Sonucu

Calistirilan komut:

```powershell
python -m unittest tests.phase_8.test_graph_state_builder tests.phase_8.test_graph_policy
```

Sonuc:

```text
Ran 9 tests in 0.621s
OK
```

Ek smoke komutlari:

```powershell
python experiments\phase_8\run_graph_supervised_pretraining.py --fusion none
python experiments\phase_8\run_graph_supervised_pretraining.py --fusion late
python experiments\phase_8\run_graph_fusion_comparison.py --fusions none late
```

Not:

- Komutlar basariyla tamamlandi.
- Ortam, Hugging Face cache klasoru icin yazma uyarisi verdi: `C:\Users\BERAT\.cache\huggingface\hub`.
- Bu uyari Faz 8.3 ciktilarini engellemedi; ileride gerekirse `TRANSFORMERS_CACHE` proje icindeki yazilabilir cache dizinine alinabilir.

## 8.4 Final Policy Comparison

### Yapilanlar

Yeni kod:

- `src/agents/graph_policy_evaluator.py`
- `experiments/phase_8/run_graph_policy_comparison.py`
- `tests/phase_8/test_phase8_evaluation_bridge.py`

Guncellenen kod:

- `src/core/evaluation.py`
- `src/training/pretrain_graph_policy.py`
- `configs/phase_8/graph_supervised_pretraining.yaml`

Bu adimda graph policy'nin mevcut vector-state evaluator mantigina dogrudan baglanmasi saglandi.
`GraphPolicyEnvAdapter`, `OffloadingEnv` icindeki canli `device/task/edge/cloud` durumundan tekrar `GraphState` kurup graph policy'ye aktarir.
Boylece `MLP-PPO`, `Pretrained MLP-PPO`, `GraphPolicy none` ve `GraphPolicy late` ayni rollout mantigi ve ayni evaluator altinda yan yana karsilastirilabilir hale geldi.

### Final Karsilastirma Protokolu

Calistirilan komut:

```powershell
python experiments\phase_8\run_graph_policy_comparison.py --seeds 42 43 44 --eval_episodes 10 --report phase_reports\Phase_8_policy_comparison.md
```

Not:

- Ortak karsilastirma seed kumesi `42, 43, 44` olarak secildi.
- Bunun nedeni, Faz 7'den gelen kanonik PPO artefaktlarinin ortak ve hazir karsilastirma kumesinin bu seed'lerde bulunmasidir.
- Semantic fusion'in supervised graph warm-start tarafindaki profesyonel protokolu ayrica `42, 43, 44, 45, 46` seed'leriyle calistirildi ve asagida ayri yorumlandi.

### Test Sonucu

Calistirilan komut:

```powershell
python -m unittest tests.phase_8.test_graph_state_builder tests.phase_8.test_graph_policy tests.phase_8.test_phase8_evaluation_bridge
```

Sonuc:

```text
Ran 11 tests
OK
```

### Final Environment Sonucu

| Model | Seeds | Success Mean | Success 95% CI | P95 Latency Mean | Avg Energy Mean | QoE Mean | Dominant Action |
|---|---:|---:|---:|---:|---:|---:|---|
| `MLP-PPO` | 3 | 66.53% | +/- 0.47% | 3.617 | 0.1321 | 48.45 | `edge_75` |
| `Pretrained MLP-PPO` | 3 | 75.40% | +/- 0.60% | 2.892 | 0.0702 | 60.94 | `edge_75` |
| `GraphPolicy none` | 3 | 75.00% | +/- 3.77% | 2.529 | 0.0705 | 62.36 | `edge_75` |
| `GraphPolicy late` | 3 | 74.87% | +/- 2.38% | 2.671 | 0.0694 | 61.51 | `edge_75` |

### 8.3'ten Gelen 5-Seed Profesyonel Fusion Sonucu

Semantic prior'in graph policy icindeki katkisini sadece final environment rollout'u ile degil, once teacher-label supervised graph warm-start tarafinda da ayri okuduk.
Bu nedenle `none` vs `late` icin profesyonel 5-seed protokol ayrica calistirildi.

Full protokol:

- 60 episode
- 50 step
- 30 epoch
- minimum 12 epoch before early stopping
- 5 seed: `42, 43, 44, 45, 46`

Sonuc:

| Fusion | Seeds | Best Val Acc Mean | Test Acc Mean | Test Acc 95% CI | Diversity Mean | Diversity 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| `none` | 5 | 75.38% | 74.04% | +/- 1.57% | 0.4609 | +/- 0.0673 |
| `late` | 5 | 84.13% | 82.76% | +/- 2.15% | 0.4380 | +/- 0.0308 |

Bu tablo, semantic prior'in supervised graph warm-start asamasinda ogrenmeyi belirgin bicimde iyilestirdigini gostermektedir.
Ancak final environment evaluation'da ayni ustunluk `GraphPolicy late > GraphPolicy none` seklinde korunmamistir.

### Dort Ana Sorunun Net Cevabi

1. `MLP-PPO + semantic prior` iyi mi?

Evet, ama Faz 8 sonunda en guclu yapi degil.
`MLP-PPO`, semantic prior ile anlamli bir baseline saglasa da ortalama `66.53%` success ile hem graph varyantlarinin hem de `Pretrained MLP-PPO`nun gerisinde kaldi.

2. `Graph policy + semantic prior` daha iyi mi?

Klasik `MLP-PPO`ya gore evet; `Pretrained MLP-PPO`ya gore net olarak hayir diyemiyoruz.
`GraphPolicy late`, `MLP-PPO`yu belirgin bicimde gecerken, `Pretrained MLP-PPO` ile neredeyse ayni banda geldi ama onu gecemedi.

3. `Graph policy`, semantic prior olmadan ne yapiyor?

Beklenenden daha guclu performans veriyor.
`GraphPolicy none`, final env karsilastirmasinda graph varyantlari icinde en iyi ortalama success (`75.00%`), en iyi latency (`2.529 s`) ve en iyi QoE (`62.36`) degerini verdi.
Bu da graph yapisinin tek basina ciddi bir karar sinyali tasidigini gosteriyor.

4. `Semantic prior`, gercekten katki sagliyor mu?

Evet, fakat katkisi asamaya bagli.
Supervised graph warm-start tarafinda semantic prior acik fayda sagliyor (`late > none`).
Ancak end-to-end environment karsilastirmasinda bu fayda henuz net ve tutarli bir ustunluge donusmus degil (`GraphPolicy late`, `GraphPolicy none`u ortalama olarak gecemedi).
Bu nedenle Faz 8 sonu icin en dogru cikarim: semantic prior, graph policy icin guclu bir ogrenme rehberi; fakat nihai env performansina katkisi icin ek RL fine-tuning veya farkli fusion tasarimi gerekebilir.

### Faz 8'in Nihai Cikarimi

- Graph-aware temsil, klasik `MLP-PPO`ya gore acik kazanc sagladi.
- `Pretrained MLP-PPO`, Faz 8 sonunda hala en guclu vector-state baseline olarak kaldi.
- `GraphPolicy none`, graph yapisinin semantic prior olmadan bile cok guclu olabildigini gosterdi.
- `GraphPolicy late`, supervised warm-start tarafinda semantic prior faydasini dogruladi; fakat bu fayda final env karsilastirmasinda ayni netlikle korunmadi.
- Tum modellerde dominant aksiyonun `edge_75` kalmasi, Faz 7'den devreden `Edge %75` attractor probleminin tamamen kirilmadigini gosterdi.

Bu nedenle Faz 8'in bilimsel cikarimi su sekilde yazilabilir:

> Graph-aware policy, partial task offloading probleminde vector-state MLP tabanli baseline'a gore daha guclu bir temsil sunmakta ve semantic prior olmadan bile yuksek karar kalitesi uretebilmektedir. Semantic prior ise graph policy icin supervised ogrenme asamasinda faydali bir rehberdir; ancak end-to-end environment performansinda katkisinin daha ileri RL uyarlamasi olmadan otomatik olarak garanti oldugu henuz gosterilememistir.

## Faz 9'a Devredilenler

Siradaki adim:

- Real-data recovery kapisini acmak ve Faz 6R kapsaminda hazirlanacak veri omurgasi uzerinde bu graph karsilastirmasini yeniden kosmak
- Real-data mode icin sessiz synthetic fallback'i kapatmak ve manifest tabanli veri envanteri tutmak
- Faz 9 kapsaminda daha zengin metrikler (`p99`, fairness, jitter, battery depletion, decision overhead) ile Faz 8 sonucunu derinlestirmek
- Graph policy icin RL fine-tuning veya alternatif fusion stratejileriyle `semantic prior` katkisinin env seviyesinde guclenip guclenmedigini test etmek
- `Edge %75` attractor problemini graph tarafinda da azaltmaya yonelik yeni egitim veya reward tasarimlari denemek

Zorunlu kapanis notu:

> Faz 8'in teknik kapanis kapisi olan `MLP-PPO / Pretrained MLP-PPO / GraphPolicy none / GraphPolicy late` karsilastirmasi synthetic/simulation-stage kosulda tamamlanmistir.
> Real-data validated kapanis icin ayni karsilastirma Faz 6R kapsaminda hazirlanacak veri omurgasi uzerinde tekrar kosulacaktir.

Faz 8 sonunda kullanilacak hikaye formatli kapanis sablonu:

- `v2_docs/phase_8/phase_8_explaination_of_studies.md`


