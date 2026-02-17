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
        
        # Action Space: 0: Local, 1: Edge, 2: Cloud
        self.action_space = spaces.Discrete(3)
        
        # State Space: [SNR, Task Size, CPU Cycles, Battery %, Edge Load]
        # Normalized values between 0 and 1
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(5,), dtype=np.float32
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
                size_bits=random.uniform(1e5, 10e6), # 100kb to 10Mb
                cpu_cycles=random.uniform(1e8, 1e10), # 100M to 10G cycles
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
        """
        if self.current_task is None or self.current_device is None:
            return self._get_obs(), 0, True, False, {}

        # 1. Calculate Base Metrics
        # (This logic mirrors what's in IoTDevice.run in simulation_env.py)
        # For training efficiency, we simulate the outcome instead of running full SimPy processes
        
        closest_edge = min(self.edge_servers, key=lambda e: math.dist(self.current_device.location, e.location))
        datarate, distance = self.channel.calculate_datarate(self.current_device, closest_edge)
        
        # Power & Energy Pred
        transmission_time = self.current_task.size_bits / datarate
        tx_energy_pred = 0.5 * transmission_time # TRANSMISSION_POWER=0.5
        local_comp_energy_pred = 1e-28 * (1e9 ** 2) * self.current_task.cpu_cycles # KAPPA=1e-28, FREQ=1e9
        
        # 2. Reward Calculation (Penalty based)
        delay = 0
        energy = 0
        
        if action == 0: # LOCAL
            delay = self.current_task.cpu_cycles / 1e9
            energy = local_comp_energy_pred
        elif action == 1: # EDGE
            delay = transmission_time + (self.current_task.cpu_cycles / (2e9)) # Edge faster 2GHz
            energy = tx_energy_pred
        else: # CLOUD
            delay = transmission_time + 0.1 + (self.current_task.cpu_cycles / (5e9)) # Cloud 5GHz + 100ms
            energy = tx_energy_pred

        # Base Reward: Minimize Delay and Energy
        # Normalization: We want to balance Latency vs. Energy. 
        # 1s Delay = -20 reward. 1 Joule = -2.0 reward.
        reward = -(delay * 20.0) - (energy * 2.0)
        
        # 3. SPECIAL PENALTIES & SHAPING
        # Cloud Usage Cost (Cloud is expensive in terms of monetary/resource cost)
        if action == 2:
            reward -= 15.0 # Constant penalty for using the most remote tier
        semantic = self.current_task.semantic_analysis
        priority_score = semantic['priority_score'] if semantic else 0.5
        
        # Deadline Penalty
        if delay > self.current_task.deadline:
            penalty = -50.0 * priority_score # Critical tasks suffer more from missing deadline
            reward += penalty
            
        # Battery Awareness
        battery_pct = (self.current_device.battery / 10000.0) * 100
        if battery_pct < 15 and action == 0:
            reward -= 20.0 # Heavy penalty for local processing on low battery
            
        # Edge/Cloud Preference for High Data
        if self.current_task.task_type.name == "HIGH_DATA" and action != 0:
            reward += 5.0 # Positive reinforcement for offloading data-heavy tasks

        # Transition to "Next Task" state (randomized for training)
        done = True 
        return self._get_obs(), reward, done, False, {}

    def _get_obs(self):
        if self.current_device and self.current_task:
            # Re-calculating metrics for state
            closest_edge = min(self.edge_servers, key=lambda e: math.dist(self.current_device.location, e.location))
            datarate, _ = self.channel.calculate_datarate(self.current_device, closest_edge)
            
            # Feature Normalization
            snr_norm = min(1.0, datarate / 50e6) # Normalize by 50 Mbps max
            size_norm = min(1.0, self.current_task.size_bits / 10e6)
            cpu_norm = min(1.0, self.current_task.cpu_cycles / 1e10)
            batt_norm = self.current_device.battery / 10000.0
            load_norm = min(1.0, closest_edge.current_load / 10.0)
            
            return np.array([snr_norm, size_norm, cpu_norm, batt_norm, load_norm], dtype=np.float32)
        
        return np.zeros((5,), dtype=np.float32)
