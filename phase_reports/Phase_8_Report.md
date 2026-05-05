Bkz. ortak kavram sozlugu: v2_docs/project_concepts_glossary.md

# Phase 8 Report: Graph-Aware Policy Upgrade

**Durum:** Devam ediyor  
**Baslangic tarihi:** 5 May 2026  
**Guncel tamamlanan adim:** 8.1 Graph state node and edge features

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

- `tests/test_graph_state_builder.py`

Yeni destek dokumani:

- `v2_docs/phase_8/graph_state_builder_contract.md`

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
python -m unittest tests.test_graph_state_builder
```

Sonuc:

```text
Ran 2 tests in 0.012s
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

### Repo Hijyen Notu

`.gitignore` icindeki `ENV/` kurali Windows ortaminda `src/env/` altindaki yeni dosyalari da ignore edebildigi icin dar bir istisna eklendi:

- `!src/env/`
- `src/env/*`
- `!src/env/graph_state_builder.py`

Bu sayede sadece yeni `graph_state_builder.py` dosyasi commit adaylari arasinda gorunur hale geldi.

---

## 8.2'ye Devredilenler

Siradaki adim:

- `src/agents/graph_policy.py` icinde graph-aware policy forward path kurmak
- `GraphState` ciktilarini PyTorch tensor formatina cevirmek
- 6 action logits ureten ilk graph policy modelini test etmek
- semantic prior'i once late-fusion stratejisiyle policy tarafina baglamak

Faz 8.1 sonunda kabul edilen sonuc:

> Bir offloading karar ani artik sadece 12 boyutlu vektor olarak degil, device-task-edge-cloud iliskilerini tasiyan test edilmis bir graph observation olarak temsil edilebiliyor.
