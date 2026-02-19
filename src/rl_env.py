import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import math

class OffloadingEnv(gym.Env):
    """
    Custom Gymnasium Environment for IoT Task Offloading.
    Interfaces with the mathematical models defined in simulation_env.py.
    """
    def __init__(self, devices=None, edge_servers=None, cloud_server=None, channel=None):
        super(OffloadingEnv, self).__init__()
        
        # Action Space: 
        # 0: Local (100% Device)
        # 1: 25% Edge (75% Device / 25% Edge)
        # 2: 50% Edge (50% Device / 50% Edge)
        # 3: 75% Edge (25% Device / 75% Edge)
        # 4: Full Edge (100% Edge)
        # 5: Full Cloud (100% Cloud)
        self.action_space = spaces.Discrete(6)
        
        # State Space: [SNR, Task Size, CPU Cycles, Battery %, Edge Load, LLM-Local, LLM-Edge, LLM-Cloud]
        # Normalized values between 0 and 1
        # Updated from (6,) to (8,) - LLM recommendation now one-hot encoded
        # Features 0-4: Physical/Network metrics
        # Features 5-7: LLM one-hot encoding [local_flag, edge_flag, cloud_flag]
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(8,), dtype=np.float32
        )
        
        self.devices = devices
        self.edge_servers = edge_servers
        self.cloud_server = cloud_server
        self.channel = channel
        
        self.current_task = None
        self.current_device = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # If we are in training mode (devices not provided or empty), create mocks
        if not self.devices:
            # Mock device with random battery
            from simulation_env import IoTDevice, Task, TaskType
            mock_device = type('MockDevice', (), {
                'battery': random.uniform(2000, 10000),
                'location': (random.uniform(0, 1000), random.uniform(0, 1000))
            })
            self.current_device = mock_device
            
            # Mock task with random specs
            self.current_task = Task(
                id=random.randint(0, 9999),
                creation_time=0,
                size_bits=random.uniform(5e4, 10e6), # Smaller tasks possible
                cpu_cycles=random.uniform(5e7, 1e10), # 50M to 10G cycles
                task_type=random.choice(list(TaskType)),
                deadline=random.uniform(0.5, 5.0)
            )
            # Add dummy semantic analysis for reward shaping
            self.current_task.semantic_analysis = {
                'priority_score': random.uniform(0.1, 1.0),
                'complexity': self.current_task.cpu_cycles / 1e10
            }
        else:
            self.current_device = random.choice(self.devices)
            # Task should be assigned by the simulator or generated
            
        return self._get_obs(), {}

    def step(self, action):
        """
        Executes the offloading decision and returns reward with LLM-guided Shaping.
        Supports Binary and Partial Offloading.
        """
        if self.current_task is None or self.current_device is None:
            return self._get_obs(), 0, True, False, {}

        # 1. Base Parameters
        if self.edge_servers:
            closest_edge = min(self.edge_servers, key=lambda e: math.dist(self.current_device.location, e.location))
            datarate, _ = self.channel.calculate_datarate(self.current_device, closest_edge)
        else:
            datarate = 10e6 # Mock for training
            closest_edge = None

        transmission_time_full = self.current_task.size_bits / datarate
        tx_energy_pred_full = 0.5 * transmission_time_full # TRANSMISSION_POWER=0.5
        local_comp_energy_pred_full = 1e-28 * (1e9 ** 2) * self.current_task.cpu_cycles # KAPPA=1e-28, FREQ=1e9
        
        delay = 0
        energy = 0
        
        # 2. Split Logic based on Action
        # Action mappings: 0:Local, 1:25% Edge, 2:50% Edge, 3:75% Edge, 4:Full Edge, 5:Full Cloud
        edge_ratios = {0: 0.0, 1: 0.25, 2: 0.5, 3: 0.75, 4: 1.0, 5: 1.0}
        ratio = edge_ratios[action]
        
        if action == 0: # 100% LOCAL
            delay = self.current_task.cpu_cycles / 1e9
            energy = local_comp_energy_pred_full
        elif action == 5: # 100% CLOUD (No partial)
            delay = transmission_time_full + 0.1 + (self.current_task.cpu_cycles / 5e9)
            energy = tx_energy_pred_full
        else: # EDGE cases (Full or Partial)
            # Local part processing
            local_part_lat = ((1 - ratio) * self.current_task.cpu_cycles) / 1e9
            local_part_en = (1 - ratio) * local_comp_energy_pred_full
            
            # Edge part processing (Parallel)
            edge_tx_lat = (ratio * self.current_task.size_bits) / datarate
            edge_comp_lat = (ratio * self.current_task.cpu_cycles) / 2e9
            edge_tx_en = 0.5 * edge_tx_lat
            
            # Parallel Delay: The task is finished when BOTH parts are complete
            delay = max(local_part_lat, edge_tx_lat + edge_comp_lat)
            energy = local_part_en + edge_tx_en

        # 3. Reward Calculation - IMPROVED with LLM Alignment
        # Base success reward (starting point is positive)
        base_reward = 100.0
        reward = base_reward
        
        # Core penalties: Minimize delay and energy
        reward -= (delay * 20.0)
        reward -= (energy * 2.0)
        
        # 🎯 NEW: LLM Alignment Bonus - Model learns to follow good LLM recommendations
        # ✅ OPTION B: Confidence-scaled rewards for reliable learning
        semantic = self.current_task.semantic_analysis
        llm_rec = semantic.get('recommended_target', 'edge') if semantic else 'edge'
        llm_confidence = semantic.get('confidence', 0.5) if semantic else 0.5  # ✅ Get confidence
        
        # Confidence-scaled alignment bonus - high confidence → higher reward
        if llm_rec == 'local' and action == 0:
            reward += 20.0 * llm_confidence  # Strong incentive, scaled by confidence
        elif llm_rec == 'edge' and 1 <= action <= 4:
            reward += 15.0 * llm_confidence  # Medium incentive, scaled by confidence
        elif llm_rec == 'cloud' and action == 5:
            reward += 15.0 * llm_confidence  # Medium incentive, scaled by confidence
        else:
            reward -= 10.0 * llm_confidence  # Penalty also scaled - unreliable LLM less impact
        
        # Special penalties
        if action == 5:
            reward -= 25.0 # Cloud cost penalty (additional to LLM penalty if misaligned)
            
        priority_score = semantic.get('priority_score', 0.5) if semantic else 0.5
        
        # Deadline Penalty
        if delay > self.current_task.deadline:
            reward -= 50.0 * priority_score # Critical tasks suffer more
            
        # Battery Awareness - ENHANCED for action diversity
        battery_pct = (self.current_device.battery / 10000.0) * 100 if hasattr(self.current_device, 'battery') else 100
        
        # Heavy incentive to preserve battery via local/partial offloading
        if battery_pct < 30:
            if action == 0:  # Local processing on low battery
                reward += 15.0 * (1.0 - battery_pct/100.0)  # More bonus as battery gets lower
            elif 1 <= action <= 3:  # Partial offloading on low battery
                reward += 10.0 * (1.0 - battery_pct/100.0)
            else:  # Full cloud on low battery (discourage)
                reward -= 30.0 * (1.0 - battery_pct/100.0)
        
        # Partial Offloading Bonus (Gradularized) - ENHANCED
        if 1 <= action <= 3:
            local_only_delay = self.current_task.cpu_cycles / 1e9
            if delay < local_only_delay:
                # Bonus based on how much faster it is
                improvement = (local_only_delay - delay) / local_only_delay
                reward += 10.0 * improvement + 3.0 # Increased base bonus
            # Energy efficiency bonus for partial offloading
            reward += 2.0 * (1.0 - energy / (local_comp_energy_pred_full * 0.5))

        # 4. Energy/Local Processing Reward - ENHANCED
        if action == 0 and energy < 0.1: # If local and low energy
             reward += 8.0 # Increased explicit reward for successful local execution
             if battery_pct > 40: reward += 5.0 # Extra bonus for maintaining healthy battery

        # Transition to "Next Task" state (randomized for training)
        done = True 
        return self._get_obs(), reward, done, False, {}

    def _get_obs(self):
        if self.current_device and self.current_task:
            # Re-calculating metrics for state
            closest_edge = min(self.edge_servers, key=lambda e: math.dist(self.current_device.location, e.location))
            datarate, _ = self.channel.calculate_datarate(self.current_device, closest_edge)
            
            # Feature Normalization (Features 0-4)
            snr_norm = min(1.0, datarate / 50e6) # Normalize by 50 Mbps max
            size_norm = min(1.0, self.current_task.size_bits / 10e6)
            cpu_norm = min(1.0, self.current_task.cpu_cycles / 1e10)
            batt_norm = self.current_device.battery / 10000.0
            load_norm = min(1.0, closest_edge.current_load / 10.0)
            
            # 🎯 NEW: One-Hot Encoding for LLM Recommendation (Features 5-7)
            # Model can better distinguish between discrete LLM categories
            llm_rec = self.current_task.semantic_analysis.get('recommended_target', 'edge') if self.current_task.semantic_analysis else 'edge'
            
            if llm_rec == 'local':
                llm_onehot = [1.0, 0.0, 0.0]  # [local, edge, cloud]
            elif llm_rec == 'edge':
                llm_onehot = [0.0, 1.0, 0.0]
            else:  # cloud
                llm_onehot = [0.0, 0.0, 1.0]
            
            # Return extended observation: 5 continuous + 3 one-hot = 8 features
            return np.array([snr_norm, size_norm, cpu_norm, batt_norm, load_norm] + llm_onehot, dtype=np.float32)
        
        return np.zeros((8,), dtype=np.float32)
