Bkz. ortak kavram sozlugu: v2_docs/project_concepts_glossary.md

# Real Data Inventory Report

Bu rapor, Faz 6R icin lokal olarak mevcut gercek veri kaynaklarinin dosya, satir sayisi ve kolon yapisini ozetler.

## alibaba_cluster_trace_2018

**Files**

- `batch_task.csv`
- `machine_meta.csv`
- `machine_usage.csv`

**Tables**

- `batch_task`: row_count_skipped_large_file
  columns: `task_name, instance_num, job_name, task_type, status, start_time, end_time, plan_cpu, plan_mem`
- `machine_meta`: 17591 rows
  columns: `machine_id, time_stamp, failure_domain_1, failure_domain_2, cpu_num, mem_size, status`
- `machine_usage`: row_count_skipped_large_file
  columns: `machine_id, time_stamp, cpu_util_percent, mem_util_percent, mem_gps, mpki, net_in, net_out, disk_io_percent`

## didi_gaia

**Files**

- `trajectories_20161116_0800_0805.csv`
- `trajectories_20161116_0800_0805_without_fill.csv`

**Tables**

- `trajectories_20161116_0800_0805`: 107998 rows
  columns: `Unnamed: 0, vehicle_id, time, longitude, latitude`
- `trajectories_20161116_0800_0805_without_fill`: 15942 rows
  columns: `Unnamed: 0, vehicle_id, time, longitude, latitude`

## glasgow_mec

**Files**

- `consecutiveTimeWDPaper3.csv`
- `RandomBasedDataset.csv`

**Tables**

- `consecutiveTimeWDPaper3`: 274 rows
  columns: `Unnamed: 0, X, X.1, V1, V2, V3, lat, long, machine_name, serverId, time (s), time (h), cpu_utilization, mem_utilization, sum`
- `RandomBasedDataset`: 203 rows
  columns: `Unnamed: 0, X.2, X, X.1, V1, V2, V3, lat, long, TotalDelay, MachineName, time`

## google_cluster_trace

**Files**

- `job_events_part_00000.csv`
- `machine_attributes.csv`
- `machine_events.csv`
- `task_events_part_00000.csv`
- `task_usage_part_00000.csv`

**Tables**

- `job_events_part_00000`: 10703 rows
  columns: `col_0, col_1, col_2, col_3, col_4, col_5, col_6, col_7`
- `machine_attributes`: row_count_skipped_large_file
  columns: `col_0, col_1, col_2, col_3, col_4`
- `machine_events`: 37779 rows
  columns: `col_0, col_1, col_2, col_3, col_4, col_5`
- `task_events_part_00000`: 450145 rows
  columns: `col_0, col_1, col_2, col_3, col_4, col_5, col_6, col_7, col_8, col_9, col_10, col_11, col_12`
- `task_usage_part_00000`: row_count_skipped_large_file
  columns: `col_0, col_1, col_2, col_3, col_4, col_5, col_6, col_7, col_8, col_9, col_10, col_11, col_12, col_13, col_14, col_15, col_16, col_17, col_18, col_19`

## uci_mec_execution_times

**Files**

- `MacBookPro1.csv`
- `MacBookPro2.csv`
- `RasberryPi.csv`
- `VM.csv`

**Tables**

- `MacBookPro1`: 1000 rows
  columns: `Time, Execution Time`
- `MacBookPro2`: 1000 rows
  columns: `Time, Execution Time`
- `RasberryPi`: 1000 rows
  columns: `Time, Execution Time`
- `VM`: 1000 rows
  columns: `Time, Execution Time`

