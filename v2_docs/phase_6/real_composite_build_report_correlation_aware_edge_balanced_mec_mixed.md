Bkz. ortak kavram sozlugu: v2_docs/project_concepts_glossary.md

# Real Composite Build Report (correlation_aware)

Bu rapor, lokal gercek veri kaynaklarindan olusturulan kompozit MEC task kayitlarini ve episode splitlerini ozetler.

## Build Summary

- fusion mode: `correlation_aware`
- decision balance: `edge_balanced`
- service class mode: `mec_mixed`
- total task record: `5000`
- train episode: `80`
- val episode: `10`
- test episode: `10`
- device count: `10`
- server count: `30`

## Source Composition

- `arrival_time`, workload identity ve difficulty ranking: Alibaba `batch_task.csv`
- `location`, `device_id`: Glasgow MEC dataset
- `server context`: Alibaba `machine_usage.csv` + `machine_meta.csv`
- `execution_time_s`: UCI MEC execution-time dataset
- `Google Cluster Trace` ve `Didi Gaia`: secondary validation / cross-check havuzunda tutulur

## Fusion Policy

- Bu build `correlation-aware fusion` olarak okunur; Alibaba arrival/difficulty rank, Glasgow mobility rank, server-load rank ve UCI execution-time rank birlikte eslenir.
- `edge_balanced` karar dengesi, payload bandini genisletip deadline slack'ini daraltarak cloud-dominant oracle davranisini azaltmayi hedefler.
- `mec_mixed` service-class kalibrasyonu local-sensing, balanced-partial, urgent-edge ve cloud-batch task aileleri uretir.

## Calibrated Proxy Fields

- `cpu_cycles`: Alibaba difficulty ranking + UCI MEC execution-time olcegi ile MEC kapasitesine kalibre edilmis proxy
- `data_size`: Alibaba `plan_mem` ranking'i ile MEC payload bandina map edilmis proxy
- `deadline`: task-specific best-case lower bound uzerinden kurulan compute-proportional proxy
- `priority`: deadline tightness / urgency proxy

## Calibration Summary

- median cpu_cycles: `1183071232`
- median deadline window: `0.590 s`
- median reference best-case delay: `0.710 s`
- state normalization metadata: `D:/task-offloading-study/data/real_composite_trace_correlation_aware_edge_balanced_mec_mixed/records/calibration_metadata.json`

## Priority Distribution

- priority `0`: `1658` task
- priority `1`: `451` task
- priority `2`: `342` task
- priority `3`: `2549` task

## Outputs

- records csv: `D:/task-offloading-study/data/real_composite_trace_correlation_aware_edge_balanced_mec_mixed/records/composite_task_records.csv`
- calibration metadata: `D:/task-offloading-study/data/real_composite_trace_correlation_aware_edge_balanced_mec_mixed/records/calibration_metadata.json`
- train split: `D:/task-offloading-study/data/real_composite_trace_correlation_aware_edge_balanced_mec_mixed/splits/train_episodes.json`
- val split: `D:/task-offloading-study/data/real_composite_trace_correlation_aware_edge_balanced_mec_mixed/splits/val_episodes.json`
- test split: `D:/task-offloading-study/data/real_composite_trace_correlation_aware_edge_balanced_mec_mixed/splits/test_episodes.json`
