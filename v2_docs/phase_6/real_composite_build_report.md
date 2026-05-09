Bkz. ortak kavram sozlugu: v2_docs/project_concepts_glossary.md

# Real Composite Build Report

Bu rapor, Faz 6R.4 kapsaminda lokal gercek veri kaynaklarindan olusturulan ilk kompozit MEC task kayitlarini ve episode splitlerini ozetler.

## Build Summary

- total task record: `5000`
- train episode: `80`
- val episode: `10`
- test episode: `10`
- device count: `10`
- server count: `2`

## Source Composition

- `arrival_time`, workload identity ve raw duration: Alibaba `batch_task.csv`
- `location`, `device_id`, `server context`: Glasgow MEC dataset
- `execution_time_s`: UCI MEC execution-time dataset
- `Google Cluster Trace` ve `Didi Gaia`: bu ilk buildde cekirdek split icin zorunlu degil; secondary validation / cross-check havuzunda tutuluyor

## Proxy Fields

- `deadline`: UCI execution time tabanli design proxy
- `data_size`: Alibaba `plan_mem` tabanli proxy
- `cpu_cycles`: Alibaba `plan_cpu` tabanli proxy
- `priority`: execution-time quantile tabanli proxy

## Priority Distribution

- priority `0`: `1284` task
- priority `1`: `1224` task
- priority `2`: `1227` task
- priority `3`: `1265` task

## Outputs

- records csv: `D:/task-offloading-study/data/real_composite_trace/records/composite_task_records.csv`
- train split: `D:/task-offloading-study/data/real_composite_trace/splits/train_episodes.json`
- val split: `D:/task-offloading-study/data/real_composite_trace/splits/val_episodes.json`
- test split: `D:/task-offloading-study/data/real_composite_trace/splits/test_episodes.json`
