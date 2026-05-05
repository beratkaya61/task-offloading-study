Bkz. ortak kavram sozlugu: v2_docs/project_concepts_glossary.md

# Faz 8.1 Graph State Builder Contract

**Tarih:** 5 May 2026  
**Durum:** 8.1 ilk implementasyon tamamlandi ve unit test ile dogrulandi  
**Kod:** `src/env/graph_state_builder.py`  
**Test:** `tests/test_graph_state_builder.py`

---

## 1. Bu Adimda Ne Yaptik

Faz 8'in ilk kod adimi olarak `GraphState` sozlesmesini ve `build_graph_state(...)` fonksiyonunu ekledik.

Bu adimda henuz GNN policy egitmedik.
Once karar anini graph olarak temsil eden temel veri yapisini kurduk.

Bu bilincli bir secimdir:

> Graph-aware policy yazmadan once, policy'nin okuyacagi graph observation'in sekli sabit ve test edilebilir olmalidir.

---

## 2. Neden Bastan Genis Dusunduk

Graph state builder'i sadece bugunku minimum ihtiyac icin yazmadik.
Faz 8.2, Faz 8.3, Faz 9 ve ilerideki tez/makale analizleri icin genisleyebilir bir sozlesme olarak tasarladik.

Bu nedenle girdi uzayi su alanlara acik:

- `device`
- `task`
- `edge_servers`
- `cloud_server`
- `channel`
- `previous_action`
- `current_step`
- `ablation_flags`
- `semantic_analysis`
- `semantic_prior`
- `trace_context`
- `normalization_config`
- `topology_config`
- `vector_state_reference`

Bu alanlar sayesinde ileride su bilgileri graph'a eklenebilir:

- onceki aksiyon ve switching overhead riski
- trace-driven veya synthetic deney kaynagi
- semantic confidence ve action prior
- ablation modlari
- farkli normalization araliklari
- daha zengin graph topology varyantlari

---

## 3. Cikti Sozlesmesi

`build_graph_state(...)` su alanlari tasiyan bir `GraphState` dondurur:

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

Bu cikti PyTorch Geometric'e bagimli degildir.
Bu sayede PyG kurulu olmasa bile graph-state testleri calisir.
Faz 8.2'de istersek bu sozlesme PyTorch tensor veya PyG `Data` formatina cevrilebilir.

---

## 4. Ilk Graph Topology

Ilk surumde karar ani su node'larla temsil edilir:

- 1 `device` node
- 1 `task` node
- N adet `edge` node
- 1 `cloud` node

Ilk edge ailesi:

- `device -> edge`
- `edge -> device`
- `task -> edge`
- `edge -> task`
- `device -> cloud`
- `cloud -> device`
- `task -> device`
- `device -> task`
- `task -> cloud`
- `cloud -> task`

Yani graph directed edge'ler tasir.
Bu, ileride GNN message passing yaparken bilginin iki yonde akabilmesini saglar.

---

## 5. Node Feature Mantigi

Node feature schema `NODE_FEATURE_SCHEMA` icinde tutulur.
Ilk surumde su fikirleri temsil eder:

- node tipi: device/task/edge/cloud
- device bataryasi
- edge/cloud kapasitesi
- load ve queue bilgisi
- remaining energy
- mobility speed
- task size ve CPU ihtiyaci
- deadline tightness
- semantic priority
- semantic confidence
- cloud latency

Bu feature seti bugun icin yeterli, ama schema listesi uzerinden genisletilebilir.

---

## 6. Edge Feature Mantigi

Edge feature schema `EDGE_FEATURE_SCHEMA` icinde tutulur.
Ilk surumde su fikirleri temsil eder:

- edge tipi: device-edge, device-cloud, task-device, task-edge, task-cloud
- distance
- datarate
- tx latency
- queue
- compute latency
- link quality
- offload ratio hint

Bu sayede graph sadece node'lari degil, node'lar arasindaki baglantinin kalitesini de tasir.

---

## 7. Global Feature Mantigi

`global_features` tum graph icin ortak baglam bilgisidir.
Ilk surumde su alanlari tasir:

- current step
- previous action
- edge server sayisi
- ortalama edge load
- maksimum edge queue
- en yakin edge mesafesi
- task deadline tightness
- semantic priority
- semantic confidence
- trace mode flag

Bu alanlar, graph node/edge yapisinin disinda kalan ama karar icin faydali global sinyalleri tasir.

---

## 8. Action Prior ve Action Mask

`action_prior`, Faz 3'ten beri kullandigimiz 6 boyutlu semantic prior'i graph sozlesmesine tasir:

- local
- edge_25
- edge_50
- edge_75
- edge_100
- cloud

`action_mask`, hangi aksiyonlarin aktif oldugunu gosterir.
Ornegin `disable_partial_offloading=True` oldugunda partial aksiyonlar kapatilir:

```text
[1, 0, 0, 0, 1, 1]
```

Bu alan Faz 8.3 semantic prior fusion ve ileride action masking icin kritik olacak.

---

## 9. Test Sonucu

Calistirilan test:

```powershell
python -m unittest tests.test_graph_state_builder
```

Sonuc:

```text
Ran 2 tests in 0.005s
OK
```

Testlerin kontrol ettigi ana noktalar:

- node feature shape
- edge index shape
- edge feature shape
- global feature shape
- action prior shape ve normalize olmasi
- action mask davranisi
- semantic ablation davranisi
- NaN/Inf olmamasi
- metadata ve trace context tasinmasi

---

## 10. Faz 8.2'ye Devredilenler

Bu adimdan sonra siradaki teknik adim:

1. `src/agents/graph_policy.py` icinde graph-aware policy forward path kurmak.
2. `GraphState` ciktilarini PyTorch tensor formatina cevirmek.
3. Ilk modelde 6 action logits uretmek.
4. Semantic prior'i once late-fusion olarak policy logits'e ekleme denemesi yapmak.

Faz 8.1 sonunda elimizde artik su temel var:

> Bir offloading karar ani, sadece 12 boyutlu vektor degil, device-task-edge-cloud iliskilerini tasiyan test edilmis bir graph observation olarak temsil edilebiliyor.
