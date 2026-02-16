# Simülasyon Ortamı Tasarımı: Dinamik Modeller ve Veri Setleri

Bu doküman, oluşturulacak Python/SimPy tabanlı simülasyonun teknik detaylarını, kullanılacak gerçek veri setlerini ve **statik olmayan (dinamik)** maliyet modellerinin matematiksel altyapısını içerir.

## 1. Kullanılacak Gerçek Veri Setleri (Real-World Datasets)

Simülasyonun gerçekçi olması için, rastgele sayı üretmek yerine aşağıdaki gerçek iz (trace) verilerini kullanacağız.

### A. İş Yükü (Workload) Verisi: **Google Cluster Trace (2019)**
*   **Nedir?** Google veri merkezlerinde çalışan binlerce görevin (task) CPU, RAM kullanımı ve sürelerini içerir.
*   **Nasıl Kullanacağız?**
    *   `Task Arrival Time`: Görevlerin simülasyona giriş anı.
    *   `Task Size (MI)`: Görevin büyüklüğü (Milyon Instruction).
    *   `CPU/RAM Request`: Görevin kaynak gereksinimi.
    *   Bu veriyi "Task Generator" modülümüz okuyacak ve sisteme enjekte edecek.

### B. Kullanıcı Hareketliliği (Mobility) Verisi: **Didi Gaia (Ride-Hailing Dataset)**
*   **Nedir?** GPS tabanlı araç rotalarını içerir.
*   **Nasıl Kullanacağız?**
    *   IoT cihazlarının (örneğin akıllı araçlar veya dronlar) $t$ anındaki $(x, y)$ koordinatlarını belirlemek için.
    *   Bu sayede Edge sunucusuna olan mesafe ($d$) sürekli değişecek, bu da iletişim gecikmesini dinamik hale getirecek.

---

## 2. Dinamik Maliyet Modelleri (Mathematical Models)

Kullanıcı "statik maliyet istemiyorum" dediği için, tüm maliyetler o anki duruma (mesafe, yoğunluk, işlemci frekansı) göre hesaplanacaktır.

### A. İletişim Modeli (Communication Model)
Kablosuz kanal üzerinden veri aktarım hızı $R_{i,j}$ (Device $i$ -> Edge $j$) Shannon Formülü ile dinamik hesaplanır:

$$ R_{i,j}(t) = B \cdot \log_2 \left( 1 + \frac{P_i \cdot h_{i,j}(t)}{\sigma^2 + I_{inter}(t)} \right) $$

*   $B$: Kanal bant genişliği (Hz).
*   $P_i$: Cihazın iletim gücü (Transmission Power).
*   $h_{i,j}(t) = d_{i,j}(t)^{-\alpha}$: Yol kaybı (Path Loss). Cihaz hareket ettikçe $d$ (mesafe) değişir, bu da hızı etkiler.
*   $\sigma^2$: Gürültü gücü.
*   $I_{inter}$: Girişim (Interference). Ortamdaki diğer cihaz sayısı arttıkça gürültü artar.

**İletişim Gecikmesi ($T_{comm}$):**
$$ T_{comm} = \frac{D_{task}}{R_{i,j}(t)} $$
($D_{task}$: Görev verisinin boyutu)

### B. Hesaplama Modeli (Computation Model)
Edge sunucularında işlem süresi, sunucunun o anki yüküne ve işlemci frekansına bağlıdır.

**Dinamik Frekans Ölçekleme (DVFS):**
Sunucu yoğunluğuna göre işlemci frekansı $f$ ($f_{min} \le f \le f_{max}$) ayarlanır.

1.  **İşlem Süresi ($T_{comp}$):**
    $$ T_{comp} = \frac{C_{task}}{f_{edge}(t)} $$
    ($C_{task}$: Görevin toplam CPU döngü sayısı - Cycles)

2.  **Kuyruk Bekleme Süresi ($T_{queue}$):**
    Edge sunucusu $M/M/1$ kuyruk yapısında modellenecektir. Sunucu doluysa görev bekler.
    $$ T_{wait} = \text{CurrentQueueLength} \times \text{AverageTaskTime} $$

### C. Enerji Tüketimi (Energy Consumption)
Enerji, hem iletim hem de işlem sırasında harcanır.

1.  **İletişim Enerjisi:**
    $$ E_{comm} = P_i \cdot T_{comm} $$

2.  **Hesaplama Enerjisi (Edge ve Device için):**
    CPU'nun güç tüketimi kapasitif özelliklere bağlıdır ($\kappa$ sabiti ile):
    $$ E_{comp} = \kappa \cdot f^3 \cdot T_{comp} $$
    *(Not: Frekansın küpü ile artar, bu yüzden DVFS enerji tasarrufu için kritiktir.)*

3.  **Pil Modeli (Battery Consumption):**
    Her cihazın bir pil kapasitesi $B_{max}$ vardır. Her işlemde pil azalır:
    $$ B(t+1) = B(t) - (E_{comm} + E_{comp}) $$
    *Eğer $B(t) < B_{threshold}$ ise cihaz offloading yapamaz veya sadece düşük güçlü moda geçer.*

### D. Ek Metrik Modelleri (Additional Metrics)

1.  **Fairness (Jain's Fairness Index):**
    Sistemin kaynakları kullanıcılara ne kadar adil dağıttığını ölçer.
    $$ J = \frac{(\sum_{i=1}^N x_i)^2}{N \cdot \sum_{i=1}^N x_i^2} $$
    ($x_i$: Kullanıcı i'nin aldığı verim/throughput)

2.  **Güvenlik Maliyeti (Security Overhead):**
    Eğer görev "Yüksek Güvenlik" gerektiriyorsa (LLM belirler), ek bir şifreleme süresi $T_{sec}$ eklenir.

---

## 3. Simülasyon Mimarisi (Python / SimPy)

Nesne Yönelimli (OOP) bir yaklaşım kullanacağız.

```python
class IoTDevice:
    def __init__(self, id, mobility_trace, battery_capacity):
        self.location = mobility_trace.get_initial_pos()
        self.battery = battery_capacity
    
    def move(self, time):
        # Didi Gaia verisinden yeni konumu al
        self.location = self.mobility_trace.get_pos(time)

class EdgeServer:
    def __init__(self, capacity, max_freq):
        self.cpu_frequency = 0  # DVFS ile değişecek
        self.queue = simpy.Resource() # Kuyruk yönetimi
        self.current_load = 0
        
    def process_task(self, task):
        # Yüke göre frekans belirle ve işlem süresini hesapla
        ...

class CommunicationChannel:
    def calculate_datarate(self, device, server):
        # Shannon formülü ve o anki mesafe ile hız hesapla
        ...
```

## 4. Sonraki Adım
Bu tasarımı onaylarsanız, **Python kodlamasına** geçip `simulation_env.py` dosyasını oluşturarak temel sınıfları (Device, Edge, Channel) yazmaya başlayacağım.
