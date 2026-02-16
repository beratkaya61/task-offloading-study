# Mobile IoT Senaryosu Hakkında

## Neden Cihazlar Hareket Ediyor?

Şu anki simülasyonda **Mobile IoT (Hareketli IoT)** senaryosu kullanılıyor. Bu senaryo şu tip cihazları temsil eder:

### Örnekler:
- 🚗 **Akıllı Arabalar (Connected Vehicles)**: Otonom araçlar, araç içi eğlence sistemleri
- 🚁 **Dronlar (UAVs)**: Teslimat dronları, gözetleme insansız hava araçları
- 📱 **Mobil Cihazlar**: Akıllı telefonlar, giyilebilir sağlık monitörleri
- 🚴 **IoT Wearables**: Konum takipli fitness cihazları

### Kullanılan Veri Seti:
- **Didi Gaia Mobility Dataset**: Gerçek araç GPS verilerinden oluşturulan hareket patternleri
- Bu sayede gerçekçi hareket simülasyonu yapılıyor (hızlanma, yavaşlama, dönüşler)

## Pil Durumu Değişiyor mu?

**Evet!** Pil dinamik olarak azalıyor:

1. **Görev Gönderme (Transmission)**: Her görev Edge/Cloud'a iletilirken enerji harcanıyor
   - Enerji = İletim Gücü × İletim Süresi
   
2. **Mesafe Etkisi**: Cihaz uzak bir sunucuya bağlanırsa daha fazla enerji harcanıyor

3. **Pil < 20%**: Cihaz kırmızı uyarı veriyor
4. **Pil = 0%**: Cihaz durur (artık görev gönderemez)

## Sabit IoT Cihazlarına Geçiş

Eğer **sabit IoT cihazları** (sensörler, akıllı ev cihazları) simüle etmek isterseniz:

1. `simulation_env.py` dosyasında `update_mobility()` fonksiyonunu devre dışı bırakabiliriz
2. Cihaz simgesini 🌡️ (termometre), 💡 (lamba), 🔌 (priz) gibi simgelerle değiştirebiliriz
3. Mobility dataset yerine sabit konumlar kullanabiliriz

**Hangi senaryoyu tercih edersiniz?**
- Option 1: Mevcut **Mobile IoT** (daha dinamik, gerçek dünya veri setleriyle)
- Option 2: **Statik IoT** (sensörler, akıllı ev cihazları - sabit konumlar)
- Option 3: **Karma** (hem sabit hem mobil cihazlar aynı anda)
