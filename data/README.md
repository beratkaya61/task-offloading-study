Bkz. ortak kavram sozlugu: v2_docs/project_concepts_glossary.md

# Data Directory Layout

Bu klasorun sade anlami su:

- `raw_real_datasets/`
  - indirilen ham gercek veri kaynaklari
  - Glasgow, UCI, Alibaba, Google, Didi gibi datasetler burada durur

- `synthetic_trace/`
  - eski Faz 6 `synthetic_didi / trace-inspired` episode splitleri
  - bunlar real-data validated sonuc degildir

- `real_composite_trace/records/`
  - gercek veri kaynaklarindan birlestirilerek uretilen tablosal kayitlar

- `real_composite_trace/splits/`
  - Faz 6R icin uretilen train/val/test episode JSON splitleri

Kisa okuma:
- ham dataset mi ariyorsun -> `raw_real_datasets/`
- eski sentetik trace split mi ariyorsun -> `synthetic_trace/`
- gercek veri birlestirme sonrasi cikti mi ariyorsun -> `real_composite_trace/`
