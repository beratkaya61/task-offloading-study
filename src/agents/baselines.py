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

    def predict_with_env(self, obs, env, deterministic=True):
        estimator = DeadlineAwareGreedyPolicy(energy_weight=0.0, queue_weight=0.0, slack_weight=0.0)
        candidates = []
        for action in getattr(env, "valid_actions", range(6)):
            estimate = estimator._estimate_action(env, int(action))
            candidates.append((estimate["delay"], int(action)))
        _, best_action = min(candidates, key=lambda item: item)
        return best_action, None

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

    def predict_with_env(self, obs, env, deterministic=True):
        valid_actions = list(getattr(env, "valid_actions", range(6)))
        population = [random.choice(valid_actions) for _ in range(self.pop_size)]

        for _ in range(self.generations):
            fitness_scores = [self._env_fitness(int(action), env) for action in population]
            parents_indices = np.argsort(fitness_scores)[-2:]
            parents = [population[i] for i in parents_indices]
            new_pop = list(parents)
            while len(new_pop) < self.pop_size:
                child = random.choice(parents)
                if random.random() < 0.2:
                    child = random.choice(valid_actions)
                new_pop.append(child)
            population = new_pop

        final_scores = [self._env_fitness(int(action), env) for action in population]
        return int(population[int(np.argmax(final_scores))]), None

    def _env_fitness(self, action, env):
        estimator = DeadlineAwareGreedyPolicy()
        estimate = estimator._estimate_action(env, action)
        task = env.current_task
        deadline = max(0.1, float(getattr(task, "deadline", 1.0)))
        miss = max(0.0, estimate["delay"] - deadline)
        feasible_bonus = 100.0 if miss <= 1e-12 else 0.0
        slack_bonus = 8.0 * max(0.0, (deadline - estimate["delay"]) / deadline)
        return (
            feasible_bonus
            + slack_bonus
            - 30.0 * miss
            - 5.0 * estimate["delay"]
            - 1.5 * estimate["energy"]
            - 0.2 * estimate["queue_delay"]
        )


class DeadlineAwareGreedyPolicy(BasePolicy):
    """
    Context-aware heuristic for real trace sanity checks.

    It reads the current env task, link, queue and deadline state, estimates every
    valid action with the same physical equations as OffloadingEnv.step(), and
    picks the lowest-cost feasible action. If no action is feasible, it chooses
    the smallest deadline miss. This is not an oracle label for training; it is a
    strong non-learning benchmark-validity baseline.
    """

    def __init__(self, energy_weight=0.02, queue_weight=0.35, slack_weight=0.05):
        self.energy_weight = float(energy_weight)
        self.queue_weight = float(queue_weight)
        self.slack_weight = float(slack_weight)

    def predict(self, obs, deterministic=True):
        return GreedyLatencyPolicy().predict(obs, deterministic=deterministic)

    def predict_with_env(self, obs, env, deterministic=True):
        task = getattr(env, "current_task", None)
        device = getattr(env, "current_device", None)
        if task is None or device is None:
            return self.predict(obs, deterministic=deterministic)

        candidates = []
        for action in getattr(env, "valid_actions", range(6)):
            estimate = self._estimate_action(env, int(action))
            deadline = float(getattr(task, "deadline", 1.0))
            miss = max(0.0, estimate["delay"] - deadline)
            feasible = miss <= 1e-12
            slack = max(0.0, deadline - estimate["delay"])
            if feasible:
                score = (
                    estimate["delay"]
                    + self.energy_weight * estimate["energy"]
                    + self.queue_weight * estimate["queue_delay"]
                    - self.slack_weight * min(slack, deadline)
                )
            else:
                score = 1_000.0 + (100.0 * miss) + estimate["delay"]
            candidates.append((score, estimate["delay"], int(action)))

        _, _, best_action = min(candidates, key=lambda item: item)
        return best_action, None

    def _estimate_action(self, env, action):
        task = env.current_task
        device = env.current_device
        closest_edge = self._closest_edge(env, device)

        if closest_edge is not None and getattr(env, "channel", None) is not None:
            datarate, snr = env.channel.calculate_datarate(device, closest_edge)
            link_quality_factor = min(1.0, snr / 20.0)
        else:
            datarate = 10e6
            link_quality_factor = 0.5

        datarate = max(float(datarate), 1e-6)
        size_bits = float(getattr(task, "size_bits", 0.0))
        cpu_cycles = float(getattr(task, "cpu_cycles", 0.0))
        transmission_time_full = size_bits / datarate
        tx_energy_pred_full = 0.5 * transmission_time_full
        local_comp_energy_pred_full = 1e-28 * (1e9 ** 2) * cpu_cycles

        if action == 0:
            return {
                "delay": cpu_cycles / 1e9,
                "energy": local_comp_energy_pred_full,
                "queue_delay": 0.0,
            }

        if action == 5:
            cloud = getattr(env, "cloud_server", None)
            cloud_queue = float(getattr(cloud, "queue_length", 0.0)) if cloud is not None else 0.0
            cloud_load = float(getattr(cloud, "current_load", 0.0)) if cloud is not None else 0.0
            cloud_congestion_delay = 0.02 * cloud_queue + 0.03 * cloud_load
            return {
                "delay": transmission_time_full + float(getattr(env, "cloud_fixed_latency", 0.1)) + (cpu_cycles / 5e9) + cloud_congestion_delay,
                "energy": tx_energy_pred_full,
                "queue_delay": cloud_congestion_delay,
            }

        ratios = {1: 0.25, 2: 0.5, 3: 0.75, 4: 1.0}
        ratio = ratios.get(action, 1.0)
        local_part_lat = ((1.0 - ratio) * cpu_cycles) / 1e9
        local_part_en = (1.0 - ratio) * local_comp_energy_pred_full
        edge_tx_lat = (ratio * size_bits) / datarate
        edge_comp_lat = (ratio * cpu_cycles) / 2e9
        edge_queue_delay = 0.0
        if closest_edge is not None:
            edge_queue_delay = 0.015 * float(getattr(closest_edge, "queue_length", 0.0)) + 0.02 * float(getattr(closest_edge, "current_load", 0.0))

        size_factor = min(1.0, size_bits / 10e6)
        coordination_factor = 1.0 if action in (1, 2, 3) else 0.35
        mobility_penalty = (1.0 - link_quality_factor) * 0.03
        transition_penalty = 0.015 if getattr(env, "previous_action", None) is not None and env.previous_action != action else 0.0
        overhead = coordination_factor * (0.01 + 0.02 * size_factor + mobility_penalty + transition_penalty)

        return {
            "delay": max(local_part_lat, edge_tx_lat + edge_comp_lat + edge_queue_delay) + overhead,
            "energy": local_part_en + (0.5 * edge_tx_lat),
            "queue_delay": edge_queue_delay,
        }

    @staticmethod
    def _closest_edge(env, device):
        edge_servers = getattr(env, "edge_servers", [])
        if not edge_servers:
            return None
        return min(
            edge_servers,
            key=lambda edge: math.dist(getattr(device, "location", (0, 0)), getattr(edge, "location", (0, 0))),
        )

# Not: DQN ve A2C baselineları için Stable Baselines3 kütüphanesi kullanılacaktır.
# Bu sınıflar evaluation.py içerisinde SB3 üzerinden yüklenecektir.
