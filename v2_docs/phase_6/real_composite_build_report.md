Bkz. ortak kavram sozlugu: v2_docs/project_concepts_glossary.md

# Real Composite Build Report

Bu rapor, Faz 6R.4 kapsaminda lokal gercek veri kaynaklarindan olusturulan guncel kompozit MEC task kayitlarini ve episode splitlerini ozetler.

## Build Summary

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
- `Google Cluster Trace` ve `Didi Gaia`: bu buildde cekirdek split icin zorunlu degil; secondary validation / cross-check havuzunda tutuluyor

## Calibrated Proxy Fields

- `cpu_cycles`: Alibaba difficulty ranking + UCI MEC execution-time olcegi ile MEC kapasitesine kalibre edilmis proxy
- `data_size`: Alibaba `plan_mem` ranking'i ile MEC payload bandina map edilmis proxy
- `deadline`: task-specific best-case lower bound uzerinden kurulan compute-proportional proxy
- `priority`: deadline tightness / urgency proxy

## Calibration Summary

- median cpu_cycles: `1604443695`
- median deadline window: `0.692 s`
- median reference best-case delay: `0.601 s`
- state normalization metadata: `D:/task-offloading-study/data/real_composite_trace/records/calibration_metadata.json`

## Priority Distribution

- priority `0`: `613` task
- priority `1`: `1431` task
- priority `2`: `1290` task
- priority `3`: `1666` task

## Outputs

- records csv: `D:/task-offloading-study/data/real_composite_trace/records/composite_task_records.csv`
- calibration metadata: `D:/task-offloading-study/data/real_composite_trace/records/calibration_metadata.json`
- train split: `D:/task-offloading-study/data/real_composite_trace/splits/train_episodes.json`
- val split: `D:/task-offloading-study/data/real_composite_trace/splits/val_episodes.json`
- test split: `D:/task-offloading-study/data/real_composite_trace/splits/test_episodes.json`
