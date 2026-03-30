import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import math

from state_builder import build_state
from reward import calculate_reward

class OffloadingEnv(gym.Env):
    """
    Custom Gymnasium Environment for IoT Task Offloading.
    Now correctly aligned with simulation dynamics via common state and reward modules.
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
        
        # Multi-step episode setup
        self.max_steps = 50
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        
        if not self.devices:
            # Simulator-backed mock for training
            from simulation_env import Task, TaskType
            self.current_device = type('MockDevice', (), {
                'battery': 10000.0,
                'location': [random.uniform(0, 1000), random.uniform(0, 1000)],
                'velocity': [random.uniform(-2, 2), random.uniform(-2, 2)]
            })
            
            self._generate_next_task()
        else:
            self.current_device = random.choice(self.devices)
            
        return self._get_obs(), {}

    def _generate_next_task(self):
        from simulation_env import Task, TaskType
        self.current_task = Task(
            id=random.randint(0, 99999),
            creation_time=self.current_step,
            size_bits=random.uniform(1e5, 10e6), 
            cpu_cycles=random.uniform(1e8, 1e10),
            task_type=random.choice(list(TaskType)),
            deadline=random.uniform(0.5, 5.0)
        )
        self.current_task.semantic_analysis = {
            'recommended_target': random.choice(['local', 'edge', 'cloud']),
            'confidence': random.uniform(0.5, 0.95),
            'priority_score': random.uniform(0.1, 1.0),
            'complexity': self.current_task.cpu_cycles / 1e10
        }

    def step(self, action):
        """
        Executes the offloading decision.
        Returns state, reward, done, truncated, info.
        """
        if self.current_task is None or self.current_device is None:
            return self._get_obs(), 0, True, False, {}

        # 1. Base Parameters
        if self.edge_servers:
            closest_edge = min(self.edge_servers, key=lambda e: math.dist(getattr(self.current_device, 'location', (0,0)), e.location))
            datarate, _ = self.channel.calculate_datarate(self.current_device, closest_edge)
        else:
            datarate = 10e6
            closest_edge = None

        transmission_time_full = self.current_task.size_bits / datarate
        tx_energy_pred_full = 0.5 * transmission_time_full
        local_comp_energy_pred_full = 1e-28 * (1e9 ** 2) * self.current_task.cpu_cycles
        
        delay = 0
        energy = 0
        
        # 2. Split Logic
        edge_ratios = {0: 0.0, 1: 0.25, 2: 0.5, 3: 0.75, 4: 1.0, 5: 1.0}
        ratio = edge_ratios[action]
        
        if action == 0: # 100% LOCAL
            delay = self.current_task.cpu_cycles / 1e9
            energy = local_comp_energy_pred_full
        elif action == 5: # 100% CLOUD
            delay = transmission_time_full + 0.1 + (self.current_task.cpu_cycles / 5e9)
            energy = tx_energy_pred_full
        else: # EDGE cases
            local_part_lat = ((1 - ratio) * self.current_task.cpu_cycles) / 1e9
            local_part_en = (1 - ratio) * local_comp_energy_pred_full
            edge_tx_lat = (ratio * self.current_task.size_bits) / datarate
            edge_comp_lat = (ratio * self.current_task.cpu_cycles) / 2e9
            edge_tx_en = 0.5 * edge_tx_lat
            
            delay = max(local_part_lat, edge_tx_lat + edge_comp_lat)
            energy = local_part_en + edge_tx_en

        # Calculate Shared Reward
        reward = calculate_reward(action, delay, energy, self.current_task, self.current_device, local_comp_energy_pred_full)

        # Update environment dynamics
        if hasattr(self.current_device, 'battery'):
            self.current_device.battery = max(0.0, self.current_device.battery - energy - 0.5) # drain battery

        if hasattr(self.current_device, 'location') and hasattr(self.current_device, 'velocity'):
            self.current_device.location[0] = (self.current_device.location[0] + self.current_device.velocity[0]) % 1000
            self.current_device.location[1] = (self.current_device.location[1] + self.current_device.velocity[1]) % 1000

        self.current_step += 1
        done = (self.current_step >= self.max_steps) or (getattr(self.current_device, 'battery', 10000.0) <= 0)
        
        if not done:
            self._generate_next_task()
            
        return self._get_obs(), reward, done, False, {}

    def _get_obs(self):
        return build_state(self.current_device, self.current_task, self.edge_servers, self.channel)
