Bkz. ortak kavram sozlugu: v2_docs/project_concepts_glossary.md

# Trace Domain-Shift Report

Bu rapor, Faz 6 kapsaminda domain shift kavramini olcmek icin uretilir.
Domain shift, bir veri dagiliminda egitilen modelin farkli bir veri dagiliminda ne kadar bozuldugunu gosterir.
- synthetic train -> trace test
- trace train -> synthetic test

## Son Durum

| Train Domain | Test Domain | Model | Success Rate | P95 Latency | Avg Energy | Dominant Action | Status |
|---|---|---|---:|---:|---:|---:|---|
| synthetic | trace | PPO | 99.60% | 0.5618 | 0.0255 | 3 | completed |
| trace | synthetic | PPO | 53.20% | 4.6883 | 0.0101 | 4 | completed |

