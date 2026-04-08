Bkz. ortak kavram sozlugu: v2_docs/project_concepts_glossary.md

# Trace Hold-Out Test Report

Bu rapor, Faz 6 kapanisindan once ayni trace checkpoint'in `train`, `val` ve `test` splitlerinde nasil davrandigini birlikte ozetler.
Amac, validation sonucunun test splitini ne kadar temsil ettigini ve belirgin bir overfitting izi olup olmadigini gormektir.

## Split Karsilastirmasi

| Split | Episode Count | Mean Success | Std | Min | Max | Mean Delay | Mean Energy |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 80 | 99.38% | 1.08 | 96.00% | 100.00% | 0.3245 s | 0.0128 |
| val | 10 | 99.00% | 1.00 | 98.00% | 100.00% | 0.3453 s | 0.0207 |
| test | 10 | 99.60% | 0.80 | 98.00% | 100.00% | 0.3317 s | 0.0128 |

## Gap Analizi

- Train - Val success gap: `+0.38` puan
- Val - Test success gap: `-0.60` puan
- Train - Test success gap: `-0.22` puan

## Yorum

Validation ve test sonuclari birbirine cok yakin. Bu, Faz 6 kapanisinda belirgin bir split-mismatch veya validation yanilgisi olmadigini destekler.
Train ve test arasindaki fark da sinirli; belirgin bir overfitting izi gorulmuyor.
