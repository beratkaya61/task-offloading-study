Bkz. ortak kavram sozlugu: v2_docs/project_concepts_glossary.md

# Real Composite Feasibility Audit

Bu rapor, guncel `real_composite_trace` benchmark'inin MEC task offloading problemi icin fiziksel olarak tutarli olup olmadigini denetler.
Bu surum artik iki ayri alt-sinir okur:
- builder referans datarate varsayimi
- env-faithful wireless/topology varsayimi

## Ana Hukum

- builder-referans alt sinir feasibility: `55.12%`
- env-faithful alt sinir feasibility: `97.40%`
- deadline penceresi cloud alt sinirindan daha kisa olan task orani: `50.38%`
- state tarafinda `cpu_norm` saturasyon orani: `0.00%`
- state tarafinda `size_norm` saturasyon orani: `4.20%`

Ilk okuma:
Env-faithful audit, benchmark'in fazla yumusaklasmis olabilecegini dusunduruyor.

## Kompozit Kayit Ozetleri

- task sayisi: `5000`
- deadline pencere medyani: `0.692 s`
- cpu_cycles medyani: `1604443695`
- size_bits medyani: `5988352`
- reference best-case delay medyani: `0.686 s`
- env-faithful best-case delay medyani: `0.440 s`

## Ham Kaynaklarin Karakteri

- Glasgow consecutive satir sayisi: `274`
- Glasgow random satir sayisi: `203`
- Glasgow consecutive icindeki serverId kumesi: `[3]`
- Glasgow machine_name sayisi: `1`
- Glasgow median cpu_utilization: `34.00`
- Glasgow median mem_utilization: `92.00`

Alibaba ilk 200k satir icinden gecerli alt-kume gozlemi:
- plan_cpu `100.0` gorulme sayisi: `95353`
- plan_cpu `50.0` gorulme sayisi: `65276`
- plan_cpu `10.0` gorulme sayisi: `5714`
- plan_cpu `400.0` gorulme sayisi: `5643`
- plan_cpu `30.0` gorulme sayisi: `5444`
- plan_cpu `5.0` gorulme sayisi: `5341`
- plan_cpu `75.0` gorulme sayisi: `1844`
- plan_cpu `200.0` gorulme sayisi: `677`
- plan_cpu `300.0` gorulme sayisi: `171`
- plan_cpu `700.0` gorulme sayisi: `52`
- Alibaba duration medyani: `10.00 s`
- Alibaba duration p95: `375.00 s`

UCI MEC execution-time kaynaklari:
- `MacBookPro1` median: `0.1074 s`, p95: `0.1179 s`, max: `0.2264 s`
- `MacBookPro2` median: `0.3070 s`, p95: `0.4875 s`, max: `1.6072 s`
- `RasberryPi` median: `1.0175 s`, p95: `1.1664 s`, max: `3.1679 s`
- `VM` median: `0.4118 s`, p95: `0.4478 s`, max: `20.7894 s`

## Priority ve Env-Feasibility

| Priority | Impossible Share | Feasible Share |
|---|---:|---:|
| 0 | 0.00% | 100.00% |
| 1 | 0.00% | 100.00% |
| 2 | 0.00% | 100.00% |
| 3 | 7.80% | 92.20% |

## Kalibrasyon Senaryolari

### CPU Divisor Senaryolari

| cpu_cycles boleni | Feasible Lower-Bound | CPU Norm Saturation | Median Cloud Delay |
|---:|---:|---:|---:|
| 1 | 55.12% | 0.00% | 0.686 s |
| 2 | 98.40% | 0.00% | 0.531 s |
| 5 | 100.00% | 0.00% | 0.420 s |
| 10 | 100.00% | 0.00% | 0.378 s |
| 20 | 100.00% | 0.00% | 0.359 s |

### Deadline Multiplier Senaryolari

| deadline carpani | Env-Feasible Lower-Bound | Median Deadline |
|---:|---:|---:|
| 0.8 | 80.40% | 0.554 s |
| 1.0 | 97.40% | 0.692 s |
| 1.2 | 99.92% | 0.831 s |
| 1.5 | 100.00% | 1.039 s |
| 2.0 | 100.00% | 1.385 s |

## Profesyonel Sonuc

Guncel `real_composite_trace` benchmark'i, onceki kalibrasyonsuz surume gore belirgin bicimde toparlanmistir.
Ancak builder-referans feasibility ile env-faithful feasibility birlikte okunmadan benchmark tam saglam ilan edilmemelidir.
Bundan sonraki yorumlarda env-faithful audit ana referans kabul edilmelidir.
