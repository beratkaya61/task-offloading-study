Bkz. ortak kavram sozlugu: v2_docs/project_concepts_glossary.md

# Faz 8 Plan: Graph-Aware Policy Upgrade

**Tarih:** 5 May 2026  
**Durum:** Baslangic plani  
**Onceki faz baglami:** Faz 7, `teacher_contextual_reward_aligned` kanonik teacher secimi ve `Pretrained + PPO = 75.20%` vs `Scratch PPO = 63.00%` sonucu ile kapatildi.

---

## Okuma Notu

Bu dosya Faz 8'in teknik uygulama planidir.
Faz 8'i ilk kez okuyorsan once `v2_docs/phase_8/phase_8_explaination_of_studies.md` dosyasini oku.
O dosya, graph-aware policy calismasini "neden yapiyoruz, ne elde edecegiz, bize ne katacak?" sorulari uzerinden daha sade sekilde anlatir.

---

## 1. Faz 8'e Neden Simdi Geciyoruz

`task.md` icindeki guncel fiili akisa gore Faz 7 tamamlandi ve siradaki asama `Faz 8 - Graph-Aware Policy Upgrade` olarak tanimlandi.

TODO dosyasindaki eski numaralandirmada gelismis metrik paketi Faz 7 gibi gorunse de, guncel yol haritasinda bu paket `Faz 9 - Advanced Metrics, Statistical Analysis and GUI` altinda ele alinacak. Bu nedenle su anki dogru siralama:

1. Faz 8: Graph-aware policy upgrade
2. Faz 9: Gelismis metrik, istatistiksel analiz ve GUI genisletmeleri
3. Faz 10: LLM self-reflection ve final dokumantasyon

Faz 7'den Faz 8'e devreden ana teknik risk:

- Kanonik pretrained policy `Full Cloud` collapse riskini kirdi.
- Ancak karar yapisi hala agirlikli olarak `Edge %75` etrafinda toplaniyor.
- Faz 8 boyunca graph-aware state ve policy'nin action diversity / decision structure kalitesini iyilestirip iyilestirmedigi izlenecek.

---

## 2. Mevcut Kod Durumu

Faz 8 baslangicinda mevcut sistem:

- `src/env/state_builder.py` 12 boyutlu vektor state uretiyor.
  - 6 fiziksel feature
  - 6 semantic prior feature
- `src/env/rl_env.py` SB3 uyumlu `Box(12,)` observation ile MLP tabanli PPO/DQN/A2C akisini destekliyor.
- `src/training/train_agent.py` MLP policy training pipeline'ini config tabanli calistiriyor.
- `src/core/evaluation.py` common evaluation pipeline ile success, p95 latency, avg energy, QoE ve action dagilimini logluyor.
- `requirements.txt` icinde Faz 8 hazirligi olarak `torch-geometric==2.5.0` bulunuyor.

Faz 8 oncesinde kullandigimiz policy ailesi acik ayrimla soyledir:

- heuristic / rule-based policy'ler: `LocalOnly`, `EdgeOnly`, `CloudOnly`, `RandomPolicy`, `GreedyLatency`, `GreedyEnergy`, `GeneticAlgorithm`
- MLP tabanli RL policy'ler: `PPO`, `DQN`, `A2C`
- semantic prior ve staged-training ile guclendirilmis MLP-PPO kolu: `PPO_from_scratch` ve `PPO_pretrained_finetuned`

Graph-aware policy, bu gruplar icinde ozellikle MLP tabanli neural RL policy hattina alternatif/guclendirici bir mimari olarak ele alinacak. Yani asil karsilastirma, `12D vector state -> MLP-PPO` ile `graph state -> GNN/graph-aware policy` arasinda kurulacak.

Yerel ortam notu:

- `requirements.txt` icinde `torch-geometric==2.5.0` yazili olsa da mevcut venv'de import kontrolu basarisiz oldu.
- Kod implementasyonuna gecmeden once PyG kurulumu dogrulanmali veya ilk graph-state testleri PyTorch-only fallback ile baslatilmali.

---

## 3. Faz 8'in Ana Hedefi

Mevcut vektor tabanli MLP policy hattini, sistemin dogal topolojisini temsil eden graph-aware bir karar mekanizmasina tasimak.

Burada "mevcut MLP policy hatti" derken ozellikle su mekanizmalar kastedilir:

- Faz 4-6 boyunca kullanilan `PPO`, `DQN`, `A2C`
- Faz 7 sonunda kanonik hale gelen semantic prior destekli `Pretrained + PPO`

Graph-aware policy, heuristic baseline'lari tamamen ortadan kaldirmak icin degil, bu ogrenebilen neural policy ailesinin state temsilini guclendirmek icin tasarlanir.

Graph yapisi su varliklari temsil edecek:

- IoT device node
- Edge server node'lari
- Cloud node
- Current task node

Policy ciktisi mevcut 6 aksiyonla uyumlu kalacak:

- `local`
- `edge_25`
- `edge_50`
- `edge_75`
- `edge_100`
- `cloud`

Bu uyumluluk onemli, cunku Faz 5-7 karsilastirmalari ve staged-training artefaktlari ayni action space ile okunmaya devam edecek.

---

## 4. Faz 8 Uygulama Sirasi

### 8.1 Graph State Sozlesmesi

Yeni dosya onerisi:

- `src/env/graph_state_builder.py`

Uretecegi ana alanlar:

- `node_features`
- `edge_index`
- `edge_features`
- `global_features`
- `action_prior`
- `node_type_ids`
- `metadata`

Ilk hedef, training'e gecmeden once ayni simulator state'inden deterministik ve test edilebilir graph temsili uretmek.

Durum guncellemesi:

- `src/env/graph_state_builder.py` eklendi.
- Framework-neutral `GraphState` dataclass'i kuruldu.
- `node_features`, `edge_index`, `edge_features`, `global_features`, `action_prior`, `action_mask`, `metadata` ve `vector_state_reference` alanlariyla genisletilebilir cikti sozlesmesi olusturuldu.
- Unit test: `tests/phase_8/test_graph_state_builder.py`
- Graph topology gorsellestirme: `src/visualization/graph_state_visualizer.py` ve `experiments/phase_8/visualize_graph_state.py`
- Ornek cikti: `results/phase_8/figures/sample_graph_state.png`
- Ayrintili sozlesme ve test notu ana aciklama dokumanina tasindi: `v2_docs/phase_8/phase_8_explaination_of_studies.md`

Node feature taslagi:

| Node tipi | Feature adaylari |
|---|---|
| device | battery, mobility speed, distance-to-nearest-edge, current task pressure |
| edge | normalized load, queue length, remaining energy, cpu capacity, distance to device |
| cloud | cloud queue, cloud load, normalized cloud latency, capacity proxy |
| task | size, cpu cycles, deadline tightness, semantic priority |

Edge feature taslagi:

| Edge | Feature adaylari |
|---|---|
| device-edge | distance, datarate, tx latency, link quality |
| device-cloud | fixed cloud latency, tx latency, cloud congestion proxy |
| task-device | local execution cost proxy |
| task-edge/task-cloud | offload feasibility proxy |

### 8.2 GNN Policy Implementation

Yeni dosyalar:

- `src/agents/graph_policy.py`
- ileride gerekirse `src/env/graph_rl_env.py`

Amac:

- Mevcut `OffloadingEnv` ve MLP-PPO deneylerini bozmadan graph-aware policy forward path kurmak.
- `GraphState` ciktilarini PyTorch tensor formatina cevirmek.
- Graph policy deneylerini ayri entrypoint/test hattinda calistirmak.
- SB3'e dogrudan graph observation vermek yerine ilk prototipi PyTorch-only custom policy olarak baslatmak.

Ilk mimari:

- graph encoder: PyTorch-only message passing fallback
- graph pooling: global mean / attention pooling
- fusion: pooled graph embedding + semantic prior
- output: 6 action logits

Ilk hedef PPO'dan once su iki problemi cozmek:

1. Graph encoder forward pass deterministik calisiyor mu?
2. Ayni state icin action logits ve valid action mask dogru uretiliyor mu?

Durum guncellemesi:

- `src/agents/graph_policy.py` eklendi.
- PyTorch-only `GraphPolicyNetwork` kuruldu; PyTorch Geometric zorunlu degil.
- `GraphState -> tensor` donusumu icin `graph_state_to_tensors(...)` eklendi.
- Action mask uygulamasi `apply_action_mask(...)` ile test edildi.
- Ilk semantic prior late-fusion yardimcisi `fuse_semantic_prior_logits(...)` eklendi; asil 8.3 deneysel fusion karsilastirmasi ayri adim olarak kalacak.
- Unit test: `tests/phase_8/test_graph_policy.py`
- Combined test: `python -m unittest tests.phase_8.test_graph_state_builder tests.phase_8.test_graph_policy`
- Sonuc: 7 test OK

### 8.3 Semantic Prior Fusion

Amac:

- Semantic prior'in graph policy kararina etkisini ayri ve olculebilir hale getirmek.
- `none`, `late fusion` ve ileride `early/node-feature fusion` varyantlarini karsilastirmak.
- Semantic prior'in action diversity uzerindeki etkisini Faz 7'den gelen `Edge %75` agirlikli davranis siniriyle birlikte okumak.

Durum:

- `fuse_semantic_prior_logits(...)` yardimcisi 8.2 kapsaminda eklendi.
- 8.3 tamamlandi; fusion modlari `none`, `input`, `late`, `input_late` olarak ayrildi.
- `semantic_prior_fusion="none"` secildiginde prior artik policy head'e gizli sekilde girmiyor; bu varyant gercek kontrol grubu oldu.
- Supervised graph warm-start hatti `src/training/pretrain_graph_policy.py` ile eklendi.
- Ilk smoke karsilastirmasi `none` ve `late` fusion icin calistirildi.
- Sonuc: `late` fusion, `none` fusion'a gore test accuracy `66.67%` vs `63.54%` ve prediction diversity `0.3661` vs `0.0000` uretmistir.
- Fusion protokolu ve kapanis hikayesi ana aciklama dokumanina tasindi: `v2_docs/phase_8/phase_8_explaination_of_studies.md`
- Bu sonuc final bilimsel sonuc degil, sadece smoke/diagnostic bulgudur.
- Profesyonel Faz 8 fusion protokolu `configs/phase_8/graph_supervised_pretraining.yaml` ile tanimlandi: 60 episode, 50 step, 30 epoch, minimum 12 epoch, patience 8 ve 5 seed.
- `experiments/phase_8/run_graph_fusion_comparison.py` artik varsayilan olarak 5 seed uzerinden mean/std/95% CI ozet raporu uretir.
- Rapor kalabaligini onlemek icin per-fusion aciklama dosyalari uretilmez; fusion anlatimi ana aciklama dokumaninda tutulur: `v2_docs/phase_8/phase_8_explaination_of_studies.md`.

### 8.4 Training Strategy

Faz 8'de asiri risk almamak icin iki asamali ilerleme:

1. Supervised graph policy warm-start
   - Faz 7 oracle label dataset'i kullanilir.
   - `selected_action_id` hedef olarak kalir.
   - Bu, GNN'in graph temsiliyle teacher kararlarini taklit edip edemedigini hizli olcer.
2. RL fine-tuning
   - Baslangicta kisa synthetic budget.
   - Sonra MLP-PPO vs GNN-policy vs GNN+semantic prior karsilastirmasi.

Bu siralama Faz 7'de kurulan staged-training disipliniyle uyumludur.

### 8.5 Evaluation

Zorunlu not:

> Faz 8, MLP-PPO ile graph-aware policy ayni evaluator uzerinde karsilastirilmadan kapanmis sayilmayacak.

Faz 8 sonunda minimum karsilastirma:

- `MLP-PPO` kanonik baseline
- `Pretrained + PPO` Faz 7 kanonik checkpoint
- `GNN policy`
- `GNN policy + semantic prior fusion`

Guncel implementasyon notu:

- Graph evaluator adapter'i `src/agents/graph_policy_evaluator.py` icinde kuruldu.
- Faz 8.4 karsilastirma entrypoint'i `experiments/phase_8/run_graph_policy_comparison.py` olarak ayrildi.
- Bu script, mevcut `evaluate_policy(...)` mantigini koruyup graph policy icin sadece observation-to-graph koprusunu ekler.

Izlenecek ana metrikler:

- success rate
- p95 latency
- avg energy
- QoE
- action diversity
- dominant action
- Edge %75 attractor oraninin dusup dusmedigi

Gelismis istatistiksel paket Faz 9'a birakilacak; ancak Faz 8 icinde graph policy'nin davranissal sinyali mutlaka raporlanacak.

Faz 8 icinde fusion icin minimum bilimsel protokol:

```powershell
python experiments\phase_8\run_graph_fusion_comparison.py --config configs\phase_8\graph_supervised_pretraining.yaml --fusions none late --seeds 42 43 44 45 46
```

Bu komutun uretmesi gereken ana dosya:

- `v2_docs/phase_8/phase_8_explaination_of_studies.md`

Rapor/sonuc hijyeni:

- Varsayilan akista Faz 8 icin per-run CSV klasoru zorunlu tutulmayacak; yalnizca acikca `--write_csv` istendiginde `results/phase_8/metrics/` altina olcum yazilacak.
- Ham CSV sadece debug icin ozellikle `--write_csv` verilirse uretilir.
- Faz 8 takibi ve not alma tek konsolide markdown raporu uzerinden yapilacak.

### 8.6 Faz 8 Kapanis Hikayesi

Faz 8 sonunda sadece teknik tablo verilmeyecek.
Kapanis raporu su hikaye akisi ile yazilacak:

1. Problem: Faz 7 sonunda `Pretrained MLP-PPO` basariyi artirdi, fakat kararlar hala `Edge %75` etrafinda yogunlasabiliyordu.
2. Hipotez: Offloading problemi dogal olarak graph yapisinda oldugu icin device-task-edge-cloud iliskilerini acikca goren graph-aware policy daha context-sensitive karar verebilir.
3. Mimari adim: Vektor state korundu, ek olarak `GraphState` ile cihaz, task, edge server ve cloud node'lari olusturuldu.
4. Policy adimi: PyTorch-only `GraphPolicyNetwork` ile graph state'ten 6 offloading aksiyonu icin logit uretildi.
5. Semantic adim: LLM/semantic prior'in graph policy icine `none`, `input`, `late`, `input_late` yollarla nasil katilacagi ayrildi.
6. Ara bulgu: Ilk supervised smoke testte `late` fusion, `none` fusion'a gore accuracy ve action diversity tarafinda daha iyi sinyal verdi.
7. Zorunlu karsilastirma: `MLP-PPO`, `Pretrained MLP-PPO`, `GraphPolicy none`, `GraphPolicy late` ayni task stream/evaluator altinda karsilastirilacak.
8. Sonuc yorumu: Graph-aware policy sadece daha yuksek skor aldi mi, yoksa Faz 7'den kalan `Edge %75` davranissal yigilmasini da azaltti mi?

Bu hikaye formatinin detayli doldurulacak sablonu:

- `v2_docs/phase_8/phase_8_explaination_of_studies.md`

Durum guncellemesi:

- Bu zorunlu karsilastirma Faz 8 sonunda tamamlandi.
- Final env karsilastirmasi: `MLP-PPO=66.53%`, `Pretrained MLP-PPO=75.40%`, `GraphPolicy none=75.00%`, `GraphPolicy late=74.87%`.
- 5-seed supervised fusion protokolunde `late` fusion, `none`a gore teacher-imitation accuracy tarafinda acik avantaj gosterdi.
- Buna ragmen final env seviyesinde `GraphPolicy late`, `GraphPolicy none`u ortalama olarak gecemedi; dolayisiyla semantic prior'in katkisi Faz 8 sonunda "ogrenme asamasinda faydali, final env davranisinda henuz kesin degil" seklinde okunur.

---

## 5. Test ve Kapanis Kriterleri

Faz 8 test katmanlari:

1. Unit-level graph state testleri
   - node count
   - edge count
   - feature shape
   - finite value kontrolu
   - semantic prior alignment
2. Policy forward testleri
   - action logits shape `(batch, 6)`
   - NaN/Inf yok
   - deterministic seed kontrolu
3. Smoke training
   - kisa supervised run
   - kisa RL/evaluation run
4. Kapanis raporu
   - `phase_reports/Phase_8_Report.md`
   - destekleyici dokumanlar `v2_docs/phase_8/` altinda tutulacak

Faz 8 tamamlanmis sayilmasi icin:

- graph state builder kodlanmis ve test edilmis olacak
- en az bir GNN/graph-aware policy forward path calisacak
- semantic prior fusion icin en az bir varyant calisacak
- MLP-PPO ile graph-aware policy ayni evaluator mantiginda karsilastirilacak
- action diversity / decision structure yorumu Faz 7 devriyle baglanacak

---

## 6. Ilk Uygulama Karari

Ilk kod adimi:

1. `src/env/graph_state_builder.py` eklenecek.
2. Deterministik graph state shape testleri yazilacak.
3. PyG kurulumu dogrulanacak; kurulum sorunluysa PyTorch-only fallback ile graph batch temsili kurulacak.

Bu karar, Faz 8'in riskini dusurur cunku once veri temsili ve test edilebilirlik sabitlenir; policy/training karmasikligi ikinci adima birakilir.


