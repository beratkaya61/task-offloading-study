import random
import numpy as np
import math

class BasePolicy:
    """Tüm baselinelar için temel sınıf."""
    def predict(self, obs):
        raise NotImplementedError

class LocalOnlyPolicy(BasePolicy):
    """Her şeyi cihazda işleyen (Action 0) baseline."""
    def predict(self, obs, deterministic=True):
        return 0, None

class EdgeOnlyPolicy(BasePolicy):
    """Her şeyi Edge sunucusuna (Action 4) gönderen baseline."""
    def predict(self, obs, deterministic=True):
        return 4, None

class CloudOnlyPolicy(BasePolicy):
    """Her şeyi Cloud sunucusuna (Action 5) gönderen baseline."""
    def predict(self, obs, deterministic=True):
        return 5, None

class RandomPolicy(BasePolicy):
    """Tamamen rastgele karar veren (Action 0-5) baseline."""
    def predict(self, obs, deterministic=True):
        return random.randint(0, 5), None

class GreedyLatencyPolicy(BasePolicy):
    """O anki en düşük gecikmeyi (Latency) seçen sezgisel baseline."""
    def predict(self, obs, deterministic=True):
        # State: [snr_norm, size_norm, cpu_norm, batt_norm, load_norm, ...]
        snr = obs[0]
        size = obs[1]
        cpu = obs[2]
        load = obs[4]
        
        # Basit gecikme tahmini (rl_env.py split logic'e benzer)
        # 0: Local
        local_lat = cpu * 10.0 # Normalize edilmiş cpu_cycles
        
        # 4: Edge (Yaklaşık)
        datarate = snr * 50e6
        edge_tx_lat = (size * 10e6) / max(1e-5, datarate)
        edge_comp_lat = (cpu * 1e10) / 2e9
        edge_lat = edge_tx_lat + edge_comp_lat + (load * 0.1) # Load etkisi
        
        # 5: Cloud
        cloud_lat = edge_tx_lat + 0.1 + ((cpu * 1e10) / 5e9)
        
        latencies = [local_lat, edge_lat, cloud_lat]
        best_idx = latencies.index(min(latencies))
        
        mapping = {0: 0, 1: 4, 2: 5}
        return mapping[best_idx], None

class GeneticAlgorithmPolicy(BasePolicy):
    """
    Eksik 1: Genetic Algorithm (GA) tabanlı Offloading Optimizer.
    Basit bir popülasyon üzerinden en iyi aksiyonu (Gen) seçer.
    """
    def __init__(self, population_size=10, generations=5):
        self.pop_size = population_size
        self.generations = generations

    def _fitness(self, action, obs):
        # Basit bir reward/fitness tahmini (Greedy'den daha kapsamlı)
        # Bu kısım rl_env.py'deki reward mantığına benzer olmalı
        # Gecikme ve Enerji dengesini gözetir
        snr = obs[0]
        size = obs[1]
        cpu = obs[2]
        batt = obs[3]
        
        # Basit simülasyon (yaklaşık)
        datarate = max(1e-5, snr * 50e6)
        tx_lat = (size * 10e6) / datarate
        
        if action == 0: # Local
            lat = (cpu * 1e10) / 1e9
            en = 1e-28 * (1e9**2) * (cpu * 1e10)
        elif action == 5: # Cloud
            lat = tx_lat + 0.1 + ((cpu * 1e10) / 5e9)
            en = 0.5 * tx_lat
        else: # Edge (Partial or Full)
            ratios = {1: 0.25, 2: 0.5, 3: 0.75, 4: 1.0}
            r = ratios[action]
            local_part = ((1-r) * cpu * 1e10) / 1e9
            edge_part = (r * cpu * 1e10) / 2e9
            lat = max(local_part, (r * size * 10e6 / datarate) + edge_part)
            en = (1-r) * (1e-28 * (1e9**2) * (cpu * 1e10)) + (0.5 * r * size * 10e6 / datarate)

        # Fitness: Düşük latency ve enerji (batarya koruma)
        score = -(lat * 10.0 + en * 2.0)
        if batt < 0.3 and action != 0: score -= 20.0 # Batarya koruma cezası
        return score

    def predict(self, obs, deterministic=True):
        # Ensure obs is a numpy array for processing
        if not isinstance(obs, np.ndarray):
            obs = np.array(obs)
            
        population = [random.randint(0, 5) for _ in range(self.pop_size)]
        
        for _ in range(self.generations):
            # Fitness hesapla - action tamsayı olmalı
            fitness_scores = [self._fitness(int(act), obs) for act in population]
            
            # Seçim (En iyi 2 ebeveyn)
            parents_indices = np.argsort(fitness_scores)[-2:]
            parents = [population[i] for i in parents_indices]
            
            # Crossover & Mutation (Yeni nesil)
            new_pop = list(parents)
            while len(new_pop) < self.pop_size:
                child = random.choice(parents)
                if random.random() < 0.2: # Mutasyon
                    child = random.randint(0, 5)
                new_pop.append(child)
            population = new_pop
            
        final_scores = [self._fitness(int(p), obs) for p in population]
        best_p = population[np.argmax(final_scores)]
        return int(best_p), None

# Not: DQN ve A2C baselineları için Stable Baselines3 kütüphanesi kullanılacaktır.
# Bu sınıflar evaluation.py içerisinde SB3 üzerinden yüklenecektir.
