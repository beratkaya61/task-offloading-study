import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import math
from collections import deque
from types import SimpleNamespace

from src.env.state_builder import build_state
from src.core.reward import calculate_reward


class OffloadingEnv(gym.Env):
    """
    Custom Gymnasium Environment for IoT Task Offloading.
    Phase 5 ablation flags + Phase 6 trace-driven episode support.
    """

    def __init__(
        self,
        devices=None,
        edge_servers=None,
        cloud_server=None,
        channel=None,
        disable_semantics=False,
        disable_reward_shaping=False,
        disable_semantic_prior=False,
        disable_confidence_weighting=False,
        disable_partial_offloading=False,
        disable_battery_awareness=False,
        disable_queue_awareness=False,
        disable_mobility_features=False,
    ):
        super().__init__()

        self.ablation_flags = {
            "disable_semantics": disable_semantics,
            "disable_reward_shaping": disable_reward_shaping,
            "disable_semantic_prior": disable_semantic_prior,
            "disable_confidence_weighting": disable_confidence_weighting,
            "disable_partial_offloading": disable_partial_offloading,
            "disable_battery_awareness": disable_battery_awareness,
            "disable_queue_awareness": disable_queue_awareness,
            "disable_mobility_features": disable_mobility_features,
        }

        # 6 discrete actions; mid actions masked when partial offloading disabled
        self.action_space = spaces.Discrete(6)
        self.valid_actions = [0, 1, 2, 3, 4, 5]
        if disable_partial_offloading:
            self.valid_actions = [0, 4, 5]

        # Observation = 5 physical + 6 semantic priors
        self.observation_space = spaces.Box(low=0, high=1, shape=(11,), dtype=np.float32)

        self.devices = devices
        self.edge_servers = edge_servers
        self.cloud_server = cloud_server
        self.channel = channel

        self.current_task = None
        self.current_device = None
        self.task_queue: deque = deque()

        self.max_steps = 50
        self.current_step = 0

    def reset(self, seed=None, options=None, episode_tasks=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.task_queue = deque(episode_tasks) if episode_tasks else deque()

        if not self.devices:
            # Mock device fallback
            from src.env.simulation_env import Task, TaskType

            self.current_device = type(
                "MockDevice",
                (),
                {
                    "battery": 10000.0,
                    "location": [random.uniform(0, 1000), random.uniform(0, 1000)],
                    "velocity": [random.uniform(-2, 2), random.uniform(-2, 2)],
                },
            )
        else:
            self.current_device = random.choice(self.devices)
            self.current_device.battery = getattr(self.current_device, "battery_capacity", 10000.0)
            self.current_device.location = [random.uniform(0, 1000), random.uniform(0, 1000)]
            self.current_device.velocity = [random.uniform(-2, 2), random.uniform(-2, 2)]

        self._generate_next_task()
        return self._get_obs(), {}

    def _generate_next_task(self):
        from src.env.simulation_env import Task, TaskType

        if self.task_queue:
            trace_task = self.task_queue.popleft()

            size_bits = getattr(trace_task, "data_size", 0) * 8 * 1024
            deadline_abs = getattr(trace_task, "deadline", 1.0)
            arrival_time = getattr(trace_task, "arrival_time", 0.0)
            deadline = max(0.1, deadline_abs - arrival_time)

            self.current_task = SimpleNamespace(
                id=getattr(trace_task, "task_id", random.randint(0, 99999)),
                creation_time=self.current_step,
                size_bits=size_bits if size_bits > 0 else random.uniform(1e5, 1e6),
                cpu_cycles=getattr(trace_task, "cpu_cycles", random.uniform(1e8, 1e10)),
                task_type=random.choice(list(TaskType)),
                deadline=deadline,
                semantic_analysis={},
            )

            priority = getattr(trace_task, "priority", 1)
            if priority >= 3:
                rec = "edge"
            elif priority == 2:
                rec = "cloud"
            else:
                rec = "local"
            self.current_task.semantic_analysis = {
                "recommended_target": rec,
                "confidence": 0.7,
                "priority_score": float(priority) / 3.0,
                "complexity": self.current_task.cpu_cycles / 1e10,
            }
        else:
            self.current_task = Task(
                id=random.randint(0, 99999),
                creation_time=self.current_step,
                size_bits=random.uniform(1e5, 10e6),
                cpu_cycles=random.uniform(1e8, 1e10),
                task_type=random.choice(list(TaskType)),
                deadline=random.uniform(0.5, 5.0),
            )
            self.current_task.semantic_analysis = {
                "recommended_target": random.choice(["local", "edge", "cloud"]),
                "confidence": random.uniform(0.5, 0.95),
                "priority_score": random.uniform(0.1, 1.0),
                "complexity": self.current_task.cpu_cycles / 1e10,
            }

    def step(self, action):
        if self.current_task is None or self.current_device is None:
            return self._get_obs(), 0.0, True, False, {}

        if isinstance(action, np.ndarray):
            action = int(action.item())
        else:
            action = int(action)

        if self.ablation_flags.get("disable_partial_offloading", False):
            action = self.valid_actions[min(action // 2, len(self.valid_actions) - 1)]

        if self.edge_servers:
            closest_edge = min(
                self.edge_servers,
                key=lambda e: math.dist(getattr(self.current_device, "location", (0, 0)), e.location),
            )
            datarate, _ = self.channel.calculate_datarate(self.current_device, closest_edge)
        else:
            datarate = 10e6

        transmission_time_full = self.current_task.size_bits / datarate
        tx_energy_pred_full = 0.5 * transmission_time_full
        local_comp_energy_pred_full = 1e-28 * (1e9**2) * self.current_task.cpu_cycles

        edge_ratios = {0: 0.0, 1: 0.25, 2: 0.5, 3: 0.75, 4: 1.0, 5: 1.0}
        ratio = edge_ratios[action]

        if action == 0:
            delay = self.current_task.cpu_cycles / 1e9
            energy = local_comp_energy_pred_full
        elif action == 5:
            delay = transmission_time_full + 0.1 + (self.current_task.cpu_cycles / 5e9)
            energy = tx_energy_pred_full
        else:
            local_part_lat = ((1 - ratio) * self.current_task.cpu_cycles) / 1e9
            local_part_en = (1 - ratio) * local_comp_energy_pred_full
            edge_tx_lat = (ratio * self.current_task.size_bits) / datarate
            edge_comp_lat = (ratio * self.current_task.cpu_cycles) / 2e9
            edge_tx_en = 0.5 * edge_tx_lat
            delay = max(local_part_lat, edge_tx_lat + edge_comp_lat)
            energy = local_part_en + edge_tx_en

        if self.ablation_flags.get("disable_reward_shaping", False):
            reward = -delay - 0.5 * energy
        else:
            reward = calculate_reward(
                action,
                delay,
                energy,
                self.current_task,
                self.current_device,
                local_comp_energy_pred_full,
            )

        if hasattr(self.current_device, "battery") and not self.ablation_flags.get(
            "disable_battery_awareness", False
        ):
            self.current_device.battery = max(0.0, self.current_device.battery - energy - 0.5)

        if (
            hasattr(self.current_device, "location")
            and hasattr(self.current_device, "velocity")
            and not self.ablation_flags.get("disable_mobility_features", False)
        ):
            self.current_device.location[0] = (self.current_device.location[0] + self.current_device.velocity[0]) % 1000
            self.current_device.location[1] = (self.current_device.location[1] + self.current_device.velocity[1]) % 1000

        self.current_step += 1
        done = (self.current_step >= self.max_steps) or (getattr(self.current_device, "battery", 10000.0) <= 0)
        if not self.task_queue:
            done = True

        if self.current_task and not self.ablation_flags.get("disable_semantics", False):
            from src.agents.semantic_prior import generate_action_prior, log_semantic_explanation

            prior = generate_action_prior(self.current_task.semantic_analysis)
            log_semantic_explanation(self.current_task, action, prior)

        success = delay <= getattr(self.current_task, "deadline", 1.0)

        info = {"task_success": success, "delay": delay, "energy": energy}

        if not done:
            self._generate_next_task()

        return self._get_obs(), reward, done, False, info

    def _get_obs(self):
        full_state = build_state(self.current_device, self.current_task, self.edge_servers, self.channel)
        if self.ablation_flags.get("disable_semantics", False):
            full_state[5:11] = 0.0
        return full_state.astype(np.float32)


class OffloadingEnv_v2(OffloadingEnv):
    """Alias kept for Phase 6 orchestrator imports."""
