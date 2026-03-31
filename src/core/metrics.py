"""
Advanced Metrics for Task Offloading Evaluation.
Faz 5 ablation study ve ilerideki fazlar için ihtiyaç duyulan metrikleri hesaplar.
"""

import numpy as np
from typing import List, Dict, Tuple

class OffloadingMetrics:
    """
    Compute advanced metrics for task offloading evaluation.
    """
    
    def __init__(self):
        self.latencies = []
        self.energies = []
        self.deadlines = []
        self.deadline_misses = []
        self.device_fairness_scores = []
        self.qoe_scores = []
        self.decisions = []
        self.rewards = []
    
    def reset(self):
        """Reset all metrics."""
        self.latencies = []
        self.energies = []
        self.deadlines = []
        self.deadline_misses = []
        self.device_fairness_scores = []
        self.qoe_scores = []
        self.decisions = []
        self.rewards = []
    
    def add_task_result(self, latency, energy, deadline, device_id=None):
        """Record a single task result."""
        self.latencies.append(latency)
        self.energies.append(energy)
        self.deadlines.append(deadline)
        
        # Deadline miss - görevin zamanında bitip bitmediğini kontrol et
        deadline_miss = 1 if latency > deadline else 0
        self.deadline_misses.append(deadline_miss)
    
    def compute_avg_latency(self) -> float:
        """Average task latency (ms)."""
        if len(self.latencies) == 0:
            return 0.0
        return np.mean(self.latencies) * 1000  # Milisaniyeye dönüştür
    
    def compute_p95_latency(self) -> float:
        """95th percentile latency (ms)."""
        if len(self.latencies) == 0:
            return 0.0
        return np.percentile(self.latencies, 95) * 1000  # Milisaniyeye dönüştür
    
    def compute_p99_latency(self) -> float:
        """99th percentile latency (ms)."""
        if len(self.latencies) == 0:
            return 0.0
        return np.percentile(self.latencies, 99) * 1000  # Milisaniyeye dönüştür
    
    def compute_deadline_miss_ratio(self) -> float:
        """Deadline miss ratio - zamanında tamamlanamayan görevlerin yüzdesi."""
        if len(self.deadline_misses) == 0:
            return 0.0
        return np.mean(self.deadline_misses)
    
    def compute_avg_energy(self) -> float:
        """Average energy consumption - görev başına ortalama enerji tüketimi (Joules)."""
        if len(self.energies) == 0:
            return 0.0
        return np.mean(self.energies)
    
    def compute_jitter(self) -> float:
        """Jitter - latency'nin standart sapmasi (ms)."""
        if len(self.latencies) < 2:
            return 0.0
        return np.std(self.latencies) * 1000  # Milisaniyeye dönüştür
    
    def compute_fairness_score(self, device_loads: Dict[int, float]) -> float:
        """
        Jain Fairness Index - cihazlar arasında load dengesini ölçer.
        Ölçek: 0 (haksız) to 1 (mükemmel adil)
        
        Formula: (sum x_i)^2 / (n * sum x_i^2)
        """
        if not device_loads or len(device_loads) == 0:
            return 1.0
        
        values = list(device_loads.values())
        if len(values) == 0 or sum(values) == 0:
            return 1.0
        
        n = len(values)
        sum_values = sum(values)
        sum_squares = sum(x**2 for x in values)
        
        if sum_squares == 0:
            return 1.0
        
        jain_index = (sum_values ** 2) / (n * sum_squares)
        return min(jain_index, 1.0)  # [0, 1] aralığında kısıtla
    
    def compute_qoe(self, success_rate: float, avg_latency_ms: float, 
                   jitter_ms: float, deadline_miss_ratio: float) -> float:
        """
        Quality of Experience (QoE) - kullanıcı deneyim kalitesi puanı.
        
        Formula:
        QoE = w1 * success + w2 * (1 - latency_penalty) + w3 * (1 - jitter_penalty) + w4 * (1 - miss_penalty)
        
        Ağırlıklar:
        - Success rate: %40
        - Latency responsiveness: %30
        - Consistency (low jitter): %15
        - Deadline respect: %15
        """
        # Latency normalizasyonu (max kabul edilebilir = 1000ms)
        latency_penalty = min(avg_latency_ms / 1000.0, 1.0)
        
        # Jitter normalizasyonu (yüksek jitter = 100ms max)
        jitter_penalty = min(jitter_ms / 100.0, 1.0)
        
        # Deadline miss cezası
        miss_penalty = deadline_miss_ratio
        
        # Ağırlıklar
        w_success = 0.40
        w_latency = 0.30
        w_jitter = 0.15
        w_miss = 0.15
        
        qoe = (w_success * success_rate + 
               w_latency * (1.0 - latency_penalty) +
               w_jitter * (1.0 - jitter_penalty) +
               w_miss * (1.0 - miss_penalty))
        
        return max(0.0, min(qoe, 1.0))  # [0, 1] aralığında kısıtla
    
    def compute_all_metrics(self, success_rate: float, device_loads: Dict[int, float] = None) -> Dict:
        """
        Compute all metrics at once - tüm metrikleri aynı anda hesapla.
        
        Args:
            success_rate: Task success rate - görev başarı oranı (0-1)
            device_loads: Dict mapping device_id -> cumulative_load
        
        Returns:
            Dictionary of all metrics - tüm metriklerin sözlüğü
        """
        avg_lat = self.compute_avg_latency()
        p95_lat = self.compute_p95_latency()
        p99_lat = self.compute_p99_latency()
        deadline_miss = self.compute_deadline_miss_ratio()
        avg_energy = self.compute_avg_energy()
        jitter = self.compute_jitter()
        fairness = self.compute_fairness_score(device_loads or {})
        qoe = self.compute_qoe(success_rate, avg_lat, jitter, deadline_miss)
        
        return {
            'metric_success_rate': success_rate,
            'metric_avg_latency_ms': avg_lat,
            'metric_p95_latency_ms': p95_lat,
            'metric_p99_latency_ms': p99_lat,
            'metric_deadline_miss_ratio': deadline_miss,
            'metric_avg_energy_j': avg_energy,
            'metric_jitter_ms': jitter,
            'metric_fairness_score': fairness,
            'metric_qoe': qoe,
        }
    
    def print_summary(self, ablation_name: str = ""):
        """Print metrics summary."""
        print(f"\n{'='*60}")
        print(f"Metrikleri: {ablation_name}")
        print(f"{'='*60}")
        print(f"  Ortalama Latency: {self.compute_avg_latency():.2f} ms")
        print(f"  P95 Latency: {self.compute_p95_latency():.2f} ms")
        print(f"  Jitter: {self.compute_jitter():.2f} ms")
        print(f"  Deadline Miss Ratio: {self.compute_deadline_miss_ratio()*100:.2f}%")
        print(f"  Ortalama Enerji: {self.compute_avg_energy():.4f} J")
        print(f"{'='*60}\n")


def compute_episode_metrics(episode_logs: List[Dict]) -> Dict:
    """
    Compute metrics from episode logs - episode loglarından metrikleri hesapla.
    
    Args:
        episode_logs: List of dicts with keys: latency, energy, deadline, action, reward
    
    Returns:
        Dictionary of computed metrics - hesaplanan metriklerin sözlüğü
    """
    metrics = OffloadingMetrics()
    
    for log in episode_logs:
        latency = log.get('latency', 0.0)
        energy = log.get('energy', 0.0)
        deadline = log.get('deadline', float('inf'))
        metrics.add_task_result(latency, energy, deadline)
    
    return metrics.compute_all_metrics(success_rate=1.0 - metrics.compute_deadline_miss_ratio())
