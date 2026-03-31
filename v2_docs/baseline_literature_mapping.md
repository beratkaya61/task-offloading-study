# Baseline Literature Mapping: reinforcement Learning Ajanlarının Seçimi ve Gerekçelendirmesi

Gelişmiş IoT "Task Offloading" problematiğinde, oluşturduğumuz sistemin (Semantik Analiz Odaklı AgentVNE türevi) başarısını değerlendirmek adına literatürde endüstri ve araştırma standardı haline gelmiş çeşitli baseline'lar seçilmiştir. Heuristic (kural-tabanlı) algoritmaların dışında, problem doğasının bir Gereksinimi olarak Derin Pekiştirmeli Öğrenme (Deep Reinforcement Learning - DRL) tabanlı üç ana ajan seçilmiştir.

Bu belgede, test paketimizde yer alan ajanların ve eklenmeyenlerin literatür tabanlı seçilme/elenme nedenlerini detaylandırıyoruz.

## 1. Modellerin Seçim Gerekçeleri

### Proximal Policy Optimization (PPO) - Ana Model
* **Neden Seçildi?:** OpenAI tarafından varsayılan model olarak kabul edilen PPO, off-policy varyansına (A2C'deki soruna) ve DDPG/SAC'deki hiperparametre kırılganlıklarına karşı dayanıklı olan kararlı algoritmaların başında gelmektedir. PPO, politika (policy) güncellemelerini kısıtlayarak (clipping) güvenli bölgelerde öğrenme gerçekleştirir. Offloading literatüründe, birden fazla objective'in optimizasyon işlemlerinde en başarılı olan algoritmalardan birisidir.
* **Temsil Ettiği Durum:** Sistemimizde **"Son Teknoloji Actor-Critic"** yapısıdır.

### Deep Q-Network (DQN) - Temel RL Modeli
* **Neden Seçildi?:** DQN, value-based (değer-odaklı) temel derim öğrenme modellerindendir ve literature göre benchmark testlerinin vazgeçilmez bir parçasıdır. Karar probleminin discrete (ayrık) olması -bizim sistemimizde 6 ayrı aksiyon (Edge1, Edge2, Cloud vs.)- DQN'i mükemmel bir baseline (kontrol) haline dönüştürür.
* **Temsil Ettiği Durum:** Value-based (Q-learning) DRL'in alt limitlerini test eder. Karşılaştırmalı analizde PPO'nun (Policy Optimization) farklılığını göstermek adına en önemli mihenk taşıdır.

### Advantage Actor-Critic (A2C) - Actor-Critic Modeli
* **Neden Seçildi?:** DQN sadece değeri ölçer, PPO ise oldukça komplekstir. İkisinin ortasında köprü görevi gören A2C (Advantage Actor Critic), hem değeri hem de seçilecek eylemi aynı anda öğrenir, ancak PPO kadar yüksek stabilite metotlarına sahip olmadığı için eğitim veriminin ölçülmesinde kullanılır.
* **Temsil Ettiği Durum:** Politika tabanlı modellerdeki Advantage etkisinin offloading problemlerine düz tepkisi ölçülür. PPO'nun "clipping" yapısının ne kadar fayda sağladığını görmek için A2C mükemmel bir ablasyon çalışması aracıdır.

---

## 2. Dışarıda Bırakılan Modeller 

### Soft Actor-Critic (SAC) ve DDPG
* **Neden Seçilmediler?:** SAC ve Deep Deterministic Policy Gradient (DDPG), sürekli eylem (continuous action space) uzayları için tasarlanmış, özellikle robotik manipülasyon problematiği için muazzam modellerdir. Bizim problemimizdeki task offloading ağacımız ise belirli "Kısımlara/Oranlara" (%25, %50 Edge vs.) veya direkt düğümlere yönlendirmeyi kapsadığı için Ayrık (Discrete) uzaylıdır. Literatürde Gumbel-Softmax gibi yaklaşımlarla SAC ayrık hale dönüştürülebilse de, bu durum problemin doğasını değiştirmekte olup doğrudan DQNs/A2C/PPO ile daha sağlıklı ve bilimsel bir kıyaslama elde edilmektedir.

---

## Sonuç
Bu araştırmada tasarlanan **PPO_v2** modelinin %62'nin üzerinde yakaladığı performans, ancak **DQN** ve **A2C** standartlarında da çalıştırılarak tam doğrulığa ulaşacaktır. Deneylerin bir sonraki fazında bu modellerin eşzamanlı, aynı random veri seti üstünde yarışması ve Jitter, Enerji, P95 metrikleriyle raporlanması tamamlanacaktır.
