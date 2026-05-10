Bkz. ortak kavram sozlugu: v2_docs/project_concepts_glossary.md

# Faz 5 Aciklamasi: Sistematik Ablation Study Calismasini Sifirdan Anlamak

## Bu Dokuman Neden Var

Bu dokuman Faz 5'in once `synthetic` kolunu, sonra da onun `real_data` koluna neden genisletildigini aciklar. Yani burada hem Faz 5'in sentetik mantigi hem de gercek veri tarafina neden ayni deney ailesinin tasindigi anlatilir.

Faz 1'den Faz 4'e kadar sistemimize bircok yeni ozellik ekledik: LLM'in anlamsal cikarimlari (semantics), odul sekillendirme (reward shaping), kismen gorev gonderme (partial offloading), cihaz hareketliligi (mobility), batarya farkindaligi vb. Sistem calisti ve belli bir basari elde etti. 

Fakat bilimsel bir calismada "Sistemimiz iyi calisiyor" demek yetmez. Akademik hakemler size su soruyu sorar: **"Bu kadar cok parca eklemissin, basari gercekten hangisinden geliyor? Belki de ekledigin parcalarin yarisi aslinda hicbir ise yaramiyor?"**

Iste Faz 5, bu zorlu ve elestirel soruya bilimsel ve sayisal bir cevap vermek icin kurgulanmistir.

---

## Veri Rejimi Notu

Faz 5 icin iki ayri rapor hatti vardir:
- `synthetic`: `v2_docs/phase_5/synthetic_phase_5_report.md`
- `real_data`: `v2_docs/phase_5/real_data_phase_5_report.md`

Bu aciklama dosyasi iki kolun ortak hikayesini anlatir; sayisal sonuclar ise kendi veri rejimi raporunda tutulur.

## 1. Faz 5'e Gelmeden Once Sistem Ne Durumdaydi?

Sistemimiz bir "Yapay Zeka Ajanina" (RL Agent) sahipti. Bu ajan, onune gelen gorevleri (task) cihazda mi cozecegine, edge sunucusuna mi gonderecegine (tamami veya bir kismi) yoksa buluta mi (cloud) gonderecegine karar veriyordu. 

Ajan bu karari verirken bir suru ipucu kullaniyordu:
- LLM'den gelen oncelik tavsiyeleri (Semantics)
- Uzaklik ve baglanti kalitesi (Mobility)
- Ekstra odul mekanizmalari (Reward Shaping)
- 6 farkli karar secenegi (Partial Offloading)

Ama elimizde bu ipuclarinin **gercekten** ne kadar degerli olduguna dair net bir kanit yoktu. Sadece "hepsini kullaninca fena sonuc almiyoruz" diyebiliyorduk.

---

## 2. Ablation Study (Bilesen Analizi) Ne Demek?

Ablation (Ablasyon), tipta "bir dokuyu veya parcayi cikarip almak" anlamina gelir. Makine ogrenmesinde de benzer bir anlami vardir: **Bir modelin nasil calistigini anlamak icin parcalarindan birini sokup cikaririz ve sistemin ne kadar coktugune bakariz.**

Bunu bir araba motorunu test etmeye benzetebiliriz:
- Araba 200 km/h hiz yapiyor. (Bu bizim `Full Model`'imiz, yani her seyi acik model)
- Acaba turbonun katkisi ne? Turboyu sokuyoruz (`w/o semantics` varyanti). Hiz 150 km/h'e duserse, anlariz ki turbo gercekten cok onemliymis.
- Acaba klimanin hiza etkisi var mi? Klimayi sokuyoruz. Hiz hala 200 km/h ise anlariz ki hiz acisindan klima gereksizmis.

Iste Faz 5'te biz ajanimizin beynindeki bu "parcalari" tek tek kapatip, sistemi o sekilde test ettik:
- **`w/o_semantics` (Anlamsal tavsiye yok):** Ajan LLM'in tavsiyelerini hic gormesin dedik.
- **`w/o_reward_shaping` (Ekstra odul yok):** Ajan sadece gorev basarili mi degil mi ona baksin, ince odullendirmeleri kaldirdik.
- **`w/o_partial_offloading` (Kismen gonderme yok):** Ajan gorevi ya %0 (cihaz) ya da %100 (sunucu) gondersin, aradaki %25, %50 gibi kesirli gonderimleri kapattik.
- **`w/o_mobility_features` (Hareketlilik yok):** Ajan cihazlarin hareket ettigini ve uzakliklarin degistigini gormesin.

Bu sayede her bir bilesenin tek basina ne kadar deger yarattigini bilimsel olarak olctuk.

---

## 3. Neden Multi-Seed (Coklu Tohum) Egitimi Yaptik?

Yapay zeka egitimleri bazen sansa baglidir. Ajan bazen egitime cok sansli bir noktadan baslar ve cok iyi sonuc alir. Bazen de ayni kod olmasina ragmen kotu bir baslangic yapar ve basarisiz olur. 

Eger modelimizi sadece bir kere egitip "Bak %80 basarili oldu" dersek, bu tesaduf olabilir. Bilimsel bir tezde buna itiraz edilir.

Bu yuzden Faz 5'te **Multi-Seed Retraining** (Farkli baslangic noktalariyla tekrar egitim) uyguladik. Yani her bir varyanti (tam model, turbosuz model, klimasiz model vb.) defalarca bastan egittik ve ortalamasini aldik. 
Bu sayede sonuclarimiz `76.17% +- 10.63` gibi gercek istatistiksel sapmalariyla birlikte raporlanabilir hale geldi. Biz buna **Scientific Seal** (Bilimsel Muhur) diyoruz.

---

## 4. Karsilastigimiz En Buyuk Sorun: Cloud-Collapse (Buluta Cokme)

Faz 5'i yaparken cok tehlikeli bir anomali (bozukluk) fark ettik. Ajanimiz butun gorevleri tembellik yapip surekli Buluta (Cloud - action 5) yollamaya baslamisti. Buna literaturde "Collapse" (Cokme) denir. Model zekice karar vermek yerine tek bir kolay yolu secmis ve hep onu kullaniyordu.

Neden boyle yapiyordu?
Cunku sentetik simulatorde bulutun "gecikme suresi" cok ideal ve masrafsiz ayarlanmisti. Edge (uc sunucular) ise dolu ve sorunluydu. Ajan "neden ugrasayim ki, her seyi buluta atarim" dedi.

**Nasil Cozduk?**
Faz 5 icinde simulatordeki gercekligi artirdik (Anomaly Giderimi ve Kalibrasyon):
- Edge sunucularina "Enerji Butcesi" ekledik.
- Cihaz bataryasi ve edge yuklerini sisteme daha sert tanittik.
- Bulutun getirecegi agir masraflari ve gecikmeleri daha gercekci modelledik.

**Sonuc:** Cloud-collapse kirildi! Son durumda ajanlar agirlikli olarak `action=3` (Agirlikli Edge gonderimi) kararina odaklanmaya basladilar. Bu da sistemimizin gercekten yerel sunuculari zekice kullandigini ispatlamis oldu.

---

## 5. Hangi Bilesen Ne Kadar Ise Yariyormus? (Buyuk Sonuclar)

Faz 5 bittiginde elimizde net bir sonuc tablosu olustu:

1. **En Kritik Bilesen: `Mobility Features` (Hareketlilik Durumu)**
   Ajanin karar verirken cihazlarin sunuculara olan mesafesini bilmesi *hayati* onem tasiyormus. Bu bilgiyi sakladigimizda basari orani bir anda %9'dan fazla dustu!

2. **Gerektiginde Kurtarici: `Partial Offloading` (Kismen Gonderme)**
   Gorevleri parcalayip gonderme (ornek: %50 cihaza, %50 edge'e) secenegini elimizden aldigimizda sistem bocaliyor. Sistem daha kati hale geldigi icin karar esnekligi azaliyor.

3. **Gercek Dunyayi Bekleyen Bilesenler: `Semantics` ve `Reward Shaping`**
   Sentetik ortamda (bilgisayar tarafindan uretilen sanal ve duzenli tasklarda) LLM'in "bu gorev karmasiktir, onceliklidir" demesinin veya ince odul ayarlamalarinin basariya devasa bir etkisi *olmadigini* gorduk.
   
   Bu husran mi? Hayir, mukemmel bir bilimsel bulgu! 
   Sentetik ortam zaten fazla steril oldugu icin zekaya pek ihtiyac duyulmuyor. Tipki duz ve bos bir otobanda giderken akilli navigasyona ihtiyac duymamaniz gibi. 
   Bu sonuc bize sunu soyledi: **"LLM'in gercek gucunu gormek istiyorsan, siradan uydurma verilerle degil, gercek ve karmasik dunyadan alinmis kargasali verilerle test etmelisin!"**

---

## 6. Sira Neden Faz 6'ya Geldi? (Gercek Veri Testi)

Faz 5'i "Sentetik / Simulasyon" ortaminda basariyla dondurduk (Sealed).
Elimizde su an:
- Nasil olctugunu bilen,
- Defalarca kendini test etmis,
- Matematiksel olarak saglam bir yapi var.

Ama Faz 5'te gorduk ki yapay (sentetik) veriler artik bizim modelimizin potansiyelini gormeye yetmiyor. 

Iste Faz 6 tam olarak bunun icin basliyor: **Trace-Driven Pipeline (Gercek Veri Izleri)**.
Artik sistemi Alibaba sunucularinin gercek yogunluk verileriyle, Roma'daki taksilerin gercek hareket rotalariyla zorlayacagiz. Faz 5'te kurdugumuz bu mukemmel test mekanizmasi (Ablation study), Faz 6'da gercek veriler uzerinde kosulacak ve "Gercek dunya verilerinde de gercekten basarili miyiz?" sorusunu yanitlayacak. (Bunun karsiligi artik Faz 5'in `real_data` koludur; yani sentetikte kosulan ayni deney ailesinin gercek veri omurgasinda yeniden kosulmasidir).

## Ozetle
Faz 5, projenin "kendi kendini sorguladigi", icindeki her bir parcanin hakkini verip vermedigini yuzlestigi ve gercek dunyaya (Faz 6) cikmadan onceki en sert antrenmanidir.
