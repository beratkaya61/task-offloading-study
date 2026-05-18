import math
import random
from collections import deque
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.core.reward import calculate_reward
from src.env.state_builder import build_state


class OffloadingEnv(gym.Env):
    """
    Unified RL environment used by synthetic training/evaluation and trace-driven episodes.

    Observation layout is kept compatible with the Phase 5 synthetic checkpoints:
    6 physical features + 6 semantic prior features = 12 dimensions.
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
        use_deadline_features=False,
        max_steps=50,
        success_bonus=0.0,
        cloud_fixed_latency=0.1,
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
            "use_deadline_features": use_deadline_features,
        }

        self.action_space = spaces.Discrete(6)
        self.valid_actions = [0, 1, 2, 3, 4, 5]
        if disable_partial_offloading:
            self.valid_actions = [0, 4, 5]

        # 6 physical features + optional 3 deadline features + 6 semantic prior features
        obs_dim = 15 if use_deadline_features else 12
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)

        self.devices = devices or []
        self.edge_servers = edge_servers or []
        self.cloud_server = cloud_server
        self.channel = channel

        self.current_task = None
        self.current_device = None
        self.task_queue = deque()
        self.trace_mode = False
        self.device_lookup = {}

        self.max_steps = max_steps
        self.current_step = 0
        self.success_bonus = float(success_bonus)
        self.cloud_fixed_latency = float(cloud_fixed_latency)
        self.previous_action = None
        self.edge_load_decay = 0.82

    def reset(self, seed=None, options=None, episode_tasks=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.previous_action = None
        self.task_queue = deque(episode_tasks) if episode_tasks else deque()
        self.trace_mode = bool(self.task_queue)
        self.device_lookup = {
            int(getattr(device, "id", index)): device
            for index, device in enumerate(self.devices)
        }

        for edge in self.edge_servers:
            edge.current_load = 0.0
            edge.queue_length = 0
            edge.remaining_energy = float(getattr(edge, 'energy_budget', 5000.0))
        if self.cloud_server is not None:
            self.cloud_server.current_load = 0.0
            self.cloud_server.queue_length = 0

        if self.trace_mode:
            for device in self.devices:
                device.battery = getattr(device, "battery_capacity", 10000.0)
                if not hasattr(device, "location"):
                    device.location = [500.0, 500.0]
                if not hasattr(device, "velocity"):
                    device.velocity = [0.0, 0.0]
            self.current_device = None
        elif not self.devices:
            self.current_device = type(
                "MockDevice",
                (),
                {
                    "battery": 10000.0,
                    "battery_capacity": 10000.0,
                    "location": [random.uniform(0, 1000), random.uniform(0, 1000)],
                    "velocity": [random.uniform(-2, 2), random.uniform(-2, 2)],
                },
            )()
        else:
            self.current_device = random.choice(self.devices)
            self.current_device.battery = getattr(self.current_device, "battery_capacity", 10000.0)
            self.current_device.location = [random.uniform(0, 1000), random.uniform(0, 1000)]
            self.current_device.velocity = [random.uniform(-2, 2), random.uniform(-2, 2)]

        self._generate_next_task()
        return self._get_obs(), {}

    def _generate_next_task(self):
        from src.env.simulation_env import Task, TaskType

        if self.trace_mode and self.task_queue:
            trace_task = self.task_queue.popleft()
            trace_device = self._resolve_trace_device(trace_task)
            self.current_device = trace_device
            self.current_device.location = [float(trace_task.location[0]), float(trace_task.location[1])]
            self.current_device.velocity = [0.0, 0.0]
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
                cpu_norm_hint=getattr(trace_task, "cpu_norm_hint", None),
                size_norm_hint=getattr(trace_task, "size_norm_hint", None),
                server_id=getattr(trace_task, "server_id", None),
                server_cpu_utilization=getattr(trace_task, "server_cpu_utilization", None),
                server_mem_utilization=getattr(trace_task, "server_mem_utilization", None),
            )

            priority = int(getattr(trace_task, "priority", 1))
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
            return

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

        closest_edge = None
        edge_energy_cost = 0.0
        edge_energy_ratio = None

        if self.edge_servers:
            closest_edge = min(
                self.edge_servers,
                key=lambda e: math.dist(getattr(self.current_device, "location", (0, 0)), e.location),
            )
            datarate, snr = self.channel.calculate_datarate(self.current_device, closest_edge)
            link_quality_factor = min(1.0, snr / 20.0)
        else:
            datarate = 10e6
            link_quality_factor = 0.5

        transmission_time_full = self.current_task.size_bits / max(datarate, 1e-6)
        tx_energy_pred_full = 0.5 * transmission_time_full
        local_comp_energy_pred_full = 1e-28 * (1e9 ** 2) * self.current_task.cpu_cycles
        edge_queue_delay = 0.0
        cloud_congestion_delay = 0.0

        edge_ratios = {0: 0.0, 1: 0.25, 2: 0.5, 3: 0.75, 4: 1.0, 5: 1.0}
        ratio = edge_ratios[action]

        # Adaptive switching overhead for partial offloading.
        overhead = 0.0
        if 1 <= action <= 4:
            size_factor = min(1.0, self.current_task.size_bits / 10e6)
            coordination_factor = 1.0 if action in (1, 2, 3) else 0.35
            mobility_penalty = (1.0 - link_quality_factor) * 0.03
            transition_penalty = 0.015 if self.previous_action is not None and self.previous_action != action else 0.0
            overhead = coordination_factor * (0.01 + 0.02 * size_factor + mobility_penalty + transition_penalty)

        if action == 0:
            delay = self.current_task.cpu_cycles / 1e9
            energy = local_comp_energy_pred_full
        elif action == 5:
            cloud_queue = float(getattr(self.cloud_server, 'queue_length', 0)) if self.cloud_server is not None else 0.0
            cloud_load = float(getattr(self.cloud_server, 'current_load', 0.0)) if self.cloud_server is not None else 0.0
            cloud_congestion_delay = 0.02 * cloud_queue + 0.03 * cloud_load
            delay = transmission_time_full + self.cloud_fixed_latency + (self.current_task.cpu_cycles / 5e9) + cloud_congestion_delay
            energy = tx_energy_pred_full
        else:
            local_part_lat = ((1 - ratio) * self.current_task.cpu_cycles) / 1e9
            local_part_en = (1 - ratio) * local_comp_energy_pred_full
            edge_tx_lat = (ratio * self.current_task.size_bits) / max(datarate, 1e-6)
            edge_comp_lat = (ratio * self.current_task.cpu_cycles) / 2e9
            edge_tx_en = 0.5 * edge_tx_lat

            if closest_edge is not None:
                edge_queue_delay = 0.015 * float(getattr(closest_edge, 'queue_length', 0)) + 0.02 * float(getattr(closest_edge, 'current_load', 0.0))
            delay = max(local_part_lat, edge_tx_lat + edge_comp_lat + edge_queue_delay) + overhead
            energy = local_part_en + edge_tx_en

            if closest_edge is not None:
                edge_energy_cost = 1e-28 * (2e9 ** 2) * (ratio * self.current_task.cpu_cycles)
                edge_energy_budget = max(1e-6, float(getattr(closest_edge, "energy_budget", 5000.0)))
                edge_remaining = float(getattr(closest_edge, "remaining_energy", edge_energy_budget))
                edge_energy_ratio = max(0.0, min(1.0, edge_remaining / edge_energy_budget))

        if self.ablation_flags.get("disable_reward_shaping", False):
            reward = -(delay * 15.0 + energy * 2.0)
        else:
            reward = calculate_reward(
                action,
                delay,
                energy,
                self.current_task,
                self.current_device,
                local_comp_energy_pred_full,
                edge_energy_ratio=edge_energy_ratio,
                edge_energy_cost=edge_energy_cost,
                success_bonus=self.success_bonus,
                use_confidence_weighting=not self.ablation_flags.get("disable_confidence_weighting", False),
            )
            if not self.ablation_flags.get("disable_mobility_features", False) and action != 0:
                reward -= (1.0 - link_quality_factor) * 10.0

        if hasattr(self.current_device, "battery") and not self.ablation_flags.get("disable_battery_awareness", False):
            self.current_device.battery = max(0.0, self.current_device.battery - energy - 0.5)

        for edge in self.edge_servers:
            edge.current_load = max(0.0, float(getattr(edge, 'current_load', 0.0)) * self.edge_load_decay)
            edge.queue_length = max(0, int(round(float(getattr(edge, 'queue_length', 0)) * self.edge_load_decay)))

        if self.cloud_server is not None:
            self.cloud_server.current_load = max(0.0, float(getattr(self.cloud_server, 'current_load', 0.0)) * 0.88)
            self.cloud_server.queue_length = max(0, int(round(float(getattr(self.cloud_server, 'queue_length', 0)) * 0.88)))

        if 1 <= action <= 4 and closest_edge is not None:
            closest_edge.current_load = min(10.0, float(getattr(closest_edge, 'current_load', 0.0)) + (0.6 if action in (1, 2, 3) else 1.0))
            closest_edge.queue_length = min(10, int(getattr(closest_edge, 'queue_length', 0)) + 1)
            closest_edge.remaining_energy = max(
                0.0,
                float(getattr(closest_edge, "remaining_energy", 5000.0)) - edge_energy_cost,
            )
        elif action == 5 and self.cloud_server is not None:
            self.cloud_server.current_load = min(12.0, float(getattr(self.cloud_server, 'current_load', 0.0)) + 1.0)
            self.cloud_server.queue_length = min(20, int(getattr(self.cloud_server, 'queue_length', 0)) + 1)

        if (
            hasattr(self.current_device, "location")
            and hasattr(self.current_device, "velocity")
            and not self.ablation_flags.get("disable_mobility_features", False)
            and not self.trace_mode
        ):
            self.current_device.location[0] = (self.current_device.location[0] + self.current_device.velocity[0]) % 1000
            self.current_device.location[1] = (self.current_device.location[1] + self.current_device.velocity[1]) % 1000

        self.current_step += 1
        battery_empty = getattr(self.current_device, "battery", 10000.0) <= 0
        done = self.current_step >= self.max_steps or battery_empty
        if self.trace_mode and not self.task_queue:
            done = True

        if self.current_task and not self.ablation_flags.get("disable_semantics", False):
            from src.agents.semantic_prior import generate_action_prior, log_semantic_explanation

            prior = generate_action_prior(self.current_task.semantic_analysis)
            log_semantic_explanation(self.current_task, action, prior)

        success = delay <= getattr(self.current_task, "deadline", 1.0)
        info = {
            "task_success": success,
            "delay": delay,
            "energy": energy,
            "deadline": getattr(self.current_task, "deadline", 1.0),
            "action": action,
            "offload_ratio": ratio,
            "partial_offload": 1 <= action <= 3,
            "switching_overhead": overhead,
            "edge_queue_delay": edge_queue_delay,
            "cloud_queue_delay": cloud_congestion_delay,
            "queue_delay": edge_queue_delay if 1 <= action <= 4 else cloud_congestion_delay,
            "edge_energy_cost": edge_energy_cost,
            "edge_energy_ratio": edge_energy_ratio if edge_energy_ratio is not None else 1.0,
            "battery_empty": battery_empty,
            "battery_level": getattr(self.current_device, "battery", 10000.0),
        }

        self.previous_action = action

        if not done:
            self._generate_next_task()

        return self._get_obs(), reward, done, False, info

    def _resolve_trace_device(self, trace_task):
        if not self.devices:
            return type(
                "TraceDevice",
                (),
                {
                    "id": int(getattr(trace_task, "device_id", 0)),
                    "battery": 10000.0,
                    "battery_capacity": 10000.0,
                    "location": [float(trace_task.location[0]), float(trace_task.location[1])],
                    "velocity": [0.0, 0.0],
                },
            )()

        device_id = int(getattr(trace_task, "device_id", 0))
        device = self.device_lookup.get(device_id)
        if device is not None:
            return device

        fallback_index = device_id % len(self.devices)
        return self.devices[fallback_index]

    def _get_obs(self):
        full_state = build_state(
            self.current_device,
            self.current_task,
            self.edge_servers,
            self.channel,
            self.ablation_flags,
        )
        if self.ablation_flags.get("disable_semantics", False):
            full_state[6:12] = 0.0
        return full_state.astype(np.float32)


class OffloadingEnv_v2(OffloadingEnv):
    """Alias kept for trace orchestrator imports."""
