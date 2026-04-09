Bkz. ortak kavram sozlugu: v2_docs/project_concepts_glossary.md

# Trace Mapping Assumptions

Amac: Faz 6 trace-driven egitim ve domain-shift analizinde kullanilan trace verisini, proje icindeki task alanlarina nasil cevirdigimizi acik ve tekrar kullanilabilir bicimde belgelemek.

Bu dokuman iki isi birden yapar:
- tez ve rapor tarafinda hangi alanin neye donusturuldugunu aciklar
- kod tarafinda TraceLoader ve TraceProcessor icindeki varsayimlari merkezi bir yerde toplar

## 1. Neden Mapping Assumptions Gerekiyor?

Trace veri setleri, offloading ortamimizin ihtiyac duydugu tum alanlari birebir vermez. Bu nedenle bazi alanlari dogrudan okuyoruz, bazilarini ise mantikli varsayimlarla turetiyoruz.

Ornek:
- bir trace dosyasinda cpu_cycles vardir ama task_type yoktur
- deadline mutlak zaman gibi gelebilir; bizim env ise goreli deadline kullanir
- mobilite konumu vardir ama semantic oncelik yoktur

Bu ceviriler kayit altina alinmazsa Faz 6 sonuclari metodolojik olarak zayif gorunur.

## 2. Kodda Bu Mapping Nerede Yapiliyor?

- src/core/trace_loader.py: diskten ham trace CSV veya kaydedilmis episode JSON dosyalarini yukler
- src/core/trace_processor.py: trace kayitlarini episode ve task yapisina cevirir
- src/env/rl_env.py: trace task objesini env icindeki gorev formatina donusturur

Kisaca:
- Loader = oku
- Processor = hazirla ve bol
- Env = egitimde kullanilacak son task nesnesine cevir

## 3. Alan Eslesme Tablosu

| Trace Alani | Hedef Alan | Donusum veya Varsayim |
|---|---|---|
| task_id | task_id | Dogrudan korunur |
| device_id | device_id | Dogrudan korunur |
| arrival_time | arrival_time | Dogrudan korunur |
| deadline | task.deadline | max(0.1, deadline - arrival_time) ile goreli deadline'a cevrilir |
| data_size (KB) | task.size_bits | data_size * 8 * 1024 |
| cpu_cycles | task.cpu_cycles | Dogrudan korunur |
| priority (0-3) | priority_score ve semantic oncelik | Dusuk -> local, orta -> cloud, yuksek -> edge egilimi olarak yorumlanir |
| location_x, location_y | location | (x, y) tuple olarak saklanir |
| task_type yoksa | task.task_type | Faz 6 mevcut surumde enum icinden rastgele secilir |
| semantic etiket yoksa | semantic_analysis | priority tabanli basit heuristic ile turetilir |

## 4. Bugun Kullandigimiz Temel Varsayimlar

### 4.1 Data size -> size_bits
Trace tarafinda data_size KB cinsinden geliyor. Env tarafi ise transmission ve enerji hesabini bit cinsinden yaptigi icin su donusum kullaniliyor:
size_bits = data_size * 8 * 1024

### 4.2 Absolute deadline -> relative deadline
Trace kaydindaki deadline degeri dogrudan gorev bitis zamani gibi yorumlaniyor. Env ise bir gorevin o andan itibaren ne kadar suresi kaldigiyla calisiyor. Bu nedenle su ceviri kullaniliyor:
relative_deadline = max(0.1, deadline - arrival_time)

Buradaki 0.1 tabani, sifir veya negatif deadline degerlerinin env'i bozmasini engelleyen guvenlik katmanidir.

### 4.3 Priority -> semantic recommendation
Trace veri seti semantic analyzer cikisi vermedigi icin Faz 6'da gecici bir heuristic kullaniyoruz:
- priority >= 3 -> edge
- priority == 2 -> cloud
- digerleri -> local

Bu, nihai semantic model degil; trace-driven egitim sirasinda semantic kanalin tamamen bos kalmamasini saglayan pratik bir koprudur.

### 4.4 Missing task_type
Trace tarafinda gorev tipi birebir yoksa TaskType enum icinden rastgele seciliyor. Bu bugun icin kabul edilebilir bir placeholder, ama Faz 6 kapanisinda sinir olarak not edilmelidir.

## 5. Bu Varsayimlarin Sonuclara Etkisi Nedir?

- Deadline mapping, success rate ve reward'u dogrudan etkiler.
- Priority -> semantics cevirisi, gercek LLM cikisi degil; trace icinden turetilmis proxy sinyaldir.
- Task type rastgele geldigi icin task turune bagli ince semantic ayrismalar Faz 6'da zayif gorunebilir.

## 6. Faz 6 Icin Simdilik Kabul Ettigimiz Sinirlar

- trace veri seti, env icin gereken tum semantic alanlari dogrudan saglamiyor
- semantic recommendation su an proxy ve heuristic nitelikte
- task_type alani gercek trace semantiginden gelmiyor
- bu nedenle Faz 6, gercek trace + kismi semantic reconstruction olarak okunmalidir

## 7. Faz 6 Kapanisinda Ne Guncellenmeli?

Trace egitimi ve domain-shift sonucu tamamlandiginda bu dosyaya sunlar eklenmeli:
- hangi raw trace dosyalari kullanildi
- nihai mapping tablosu degisti mi
- heuristic semantic reconstruction yeterli bulundu mu
- ek normalizasyon veya filtering adimlari uygulandi mi

## 8. Kisa Ozet

Faz 6'da trace-driven deney yaparken ham veriyi dogrudan env'e vermiyoruz.
1. TraceLoader diskteki trace veya episode split dosyasini okur
2. TraceProcessor bunlari episode ve task yapisina hazirlar
3. rl_env.py her trace task'i env icindeki gorev formatina cevirir
4. Bu ceviriler sirasinda data_size, deadline ve priority gibi alanlar belirli varsayimlarla map edilir
