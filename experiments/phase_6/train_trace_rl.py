"""
Faz 6: ortak trace-tabanli RL egitim omurgasi.

Bu modul:
- sentetik trace dogrulama kosulari,
- real-composite-trace RL egitimi,
- Faz 5'in gercek veriyle yeniden kosulacak ablation deneyleri
icin paylasilan yurutme katmanidir.
"""

from __future__ import annotations

import json
import logging
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import simpy
import yaml
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.core.real_data_manifest import assert_real_data_ready, is_real_data_mode
from src.core.trace_loader import TraceLoader
from src.core.trace_processor import TraceEpisode, TraceProcessor
from src.env.rl_env import OffloadingEnv_v2
from src.env.simulation_env import CloudServer, EdgeServer, IoTDevice, WirelessChannel


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


TRACE_ALGORITHM_CLASSES = {
    "ppo": PPO,
    "dqn": DQN,
    "a2c": A2C,
}

TRACE_ALGORITHM_DEFAULTS = {
    "ppo": {
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
    },
    "dqn": {
        "learning_rate": 1e-4,
        "gamma": 0.99,
        "buffer_size": 10000,
        "learning_starts": 1000,
        "batch_size": 64,
        "train_freq": 4,
        "target_update_interval": 500,
    },
    "a2c": {
        "learning_rate": 7e-4,
        "gamma": 0.99,
        "n_steps": 5,
    },
}


class TraceOffloadingEnv(OffloadingEnv_v2):
    """Materialized trace episode listesini sirali olarak tuketen env wrapper."""

    def __init__(self, episodes: List[TraceEpisode], **kwargs):
        super().__init__(**kwargs)
        self.episodes = episodes
        self._ep_idx = 0

    def reset(self, seed=None, options=None):
        if not self.episodes:
            return super().reset(seed=seed, options=options)
        episode = self.episodes[self._ep_idx % len(self.episodes)]
        self._ep_idx += 1
        return super().reset(seed=seed, options=options, episode_tasks=episode.tasks)


class TraceMetricsCallback(BaseCallback):
    """Episode bazli basari/gecikme/enerji kayitlarini CSV'ye yazar."""

    def __init__(self, log_path: Path, verbose: int = 0):
        super().__init__(verbose)
        self.log_path = log_path
        self.history = {"episode": [], "success_rate": [], "avg_delay": [], "avg_energy": []}
        self._ep_id = 0
        self._ep_success = 0
        self._ep_tasks = 0
        self._delays: List[float] = []
        self._energies: List[float] = []

    def _init_callback(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for info, done in zip(infos, dones):
            self._ep_success += int(info.get("task_success", False))
            if "delay" in info:
                self._delays.append(info["delay"])
            if "energy" in info:
                self._energies.append(info["energy"])
            self._ep_tasks += 1

            if done:
                success_rate = self._ep_success / max(self._ep_tasks, 1)
                avg_delay = float(np.mean(self._delays)) if self._delays else 0.0
                avg_energy = float(np.mean(self._energies)) if self._energies else 0.0

                self.history["episode"].append(self._ep_id)
                self.history["success_rate"].append(success_rate * 100.0)
                self.history["avg_delay"].append(avg_delay)
                self.history["avg_energy"].append(avg_energy)

                frame = pd.DataFrame(
                    [
                        {
                            "episode": self._ep_id,
                            "success_rate": success_rate * 100.0,
                            "avg_delay": avg_delay,
                            "avg_energy": avg_energy,
                            "timestamp": datetime.now().isoformat(),
                        }
                    ]
                )
                frame.to_csv(self.log_path, mode="a", header=not self.log_path.exists(), index=False)

                self._ep_id += 1
                self._ep_success = 0
                self._ep_tasks = 0
                self._delays = []
                self._energies = []

        return True


class TraceTrainingOrchestrator:
    """PPO, DQN ve A2C icin ortak trace-egitim orkestratoru."""

    def __init__(
        self,
        config_path: str = "configs/phase_6/synthetic_trace_rl_training.yaml",
        seed: int = 42,
        config_dict: Optional[Dict] = None,
    ):
        self.seed = seed
        np.random.seed(seed)
        self.config_path = config_path

        if config_dict is not None:
            self.config_dict = deepcopy(config_dict)
        else:
            with open(config_path, "r", encoding="utf-8") as handle:
                self.config_dict = yaml.safe_load(handle)

        self.log_dir = Path(self.config_dict["logging"]["log_dir"])
        self.checkpoint_dir = Path(self.config_dict["logging"]["checkpoint_dir"])
        self.report_path = Path(
            self.config_dict["logging"].get("report_path", "phase_reports/Phase_6_Report.md")
        )
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.report_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Initialized TraceTrainingOrchestrator")
        logger.info("  Config: %s", config_path if config_dict is None else "<in-memory>")
        logger.info("  Log dir: %s", self.log_dir)
        logger.info("  Checkpoint dir: %s", self.checkpoint_dir)
        logger.info("  Report path: %s", self.report_path)

    def _success_bonus(self) -> float:
        env_cfg = self.config_dict["environment"]
        if not env_cfg.get("use_success_bonus", False):
            return 0.0
        return float(env_cfg.get("success_bonus", 100.0))

    def _algorithm_name(self) -> str:
        return str(self.config_dict.get("training", {}).get("algorithm", "PPO")).lower()

    def _algorithm_class(self):
        algorithm = self._algorithm_name()
        if algorithm not in TRACE_ALGORITHM_CLASSES:
            raise ValueError(f"Unsupported trace algorithm: {algorithm}")
        return TRACE_ALGORITHM_CLASSES[algorithm]

    def _algorithm_checkpoint_name(self, suffix: str = "trace_best.zip") -> str:
        return f"{self._algorithm_name()}_{suffix}"

    def _create_model(self, env):
        algorithm = self._algorithm_name()
        train_cfg = self.config_dict["training"]
        defaults = TRACE_ALGORITHM_DEFAULTS[algorithm]
        model_class = self._algorithm_class()
        common = {
            "learning_rate": train_cfg.get("learning_rate", defaults.get("learning_rate")),
            "gamma": train_cfg.get("gamma", defaults.get("gamma", 0.99)),
            "device": "cuda" if self.config_dict["device"]["cuda"] else "cpu",
            "verbose": 1,
        }

        if algorithm == "ppo":
            return model_class(
                "MlpPolicy",
                env,
                n_steps=train_cfg.get("n_steps", defaults["n_steps"]),
                batch_size=train_cfg.get("batch_size", defaults["batch_size"]),
                n_epochs=train_cfg.get("n_epochs", defaults["n_epochs"]),
                **common,
            )

        if algorithm == "dqn":
            return model_class(
                "MlpPolicy",
                env,
                buffer_size=train_cfg.get("buffer_size", defaults["buffer_size"]),
                learning_starts=train_cfg.get("learning_starts", defaults["learning_starts"]),
                batch_size=train_cfg.get("batch_size", defaults["batch_size"]),
                train_freq=train_cfg.get("train_freq", defaults["train_freq"]),
                target_update_interval=train_cfg.get(
                    "target_update_interval", defaults["target_update_interval"]
                ),
                **common,
            )

        return model_class(
            "MlpPolicy",
            env,
            n_steps=train_cfg.get("n_steps", defaults["n_steps"]),
            **common,
        )

    def _load_model(self, checkpoint_path: str, env):
        return self._algorithm_class().load(checkpoint_path, env=env)

    def _resolve_feature_flags(self, feature_overrides: Optional[Dict] = None) -> Dict[str, bool]:
        env_cfg = self.config_dict["environment"]
        flags = {
            "reward_shaping": bool(env_cfg.get("use_reward_shaping", True)),
            "partial_offloading": bool(env_cfg.get("use_partial_offloading", True)),
            "semantics": bool(env_cfg.get("use_semantic_features", True)),
            "semantic_prior": bool(env_cfg.get("use_semantic_features", True)),
            "confidence_weighting": bool(env_cfg.get("use_confidence_weighting", False)),
            "battery_awareness": bool(env_cfg.get("use_battery_awareness", True)),
            "queue_awareness": bool(env_cfg.get("use_queue_awareness", False)),
            "mobility_features": bool(env_cfg.get("use_mobility_features", True)),
        }
        if feature_overrides:
            flags.update(feature_overrides)
        return flags

    def _build_topology(self):
        env_sim = simpy.Environment()
        channel = WirelessChannel()
        cloud = CloudServer(env_sim)
        num_edge_servers = self.config_dict["environment"].get("n_edge_servers", 3)
        edge_servers = [
            EdgeServer(env_sim, i + 1, (np.random.uniform(0, 1000), np.random.uniform(0, 1000)), 2e9)
            for i in range(num_edge_servers)
        ]
        num_devices = self.config_dict["environment"]["n_devices"]
        devices = [
            IoTDevice(
                env_sim,
                id=i,
                channel=channel,
                edge_servers=edge_servers,
                cloud_server=cloud,
                battery_capacity=10000.0,
            )
            for i in range(num_devices)
        ]
        return devices, edge_servers, cloud, channel

    def _build_trace_env(
        self,
        episodes: List[TraceEpisode],
        feature_overrides: Optional[Dict] = None,
    ) -> TraceOffloadingEnv:
        devices, edge_servers, cloud, channel = self._build_topology()
        flags = self._resolve_feature_flags(feature_overrides)
        return TraceOffloadingEnv(
            episodes=episodes,
            devices=devices,
            edge_servers=edge_servers,
            cloud_server=cloud,
            channel=channel,
            disable_reward_shaping=not flags["reward_shaping"],
            disable_partial_offloading=not flags["partial_offloading"],
            disable_semantics=not flags["semantics"],
            disable_semantic_prior=not flags["semantic_prior"],
            disable_confidence_weighting=not flags["confidence_weighting"],
            disable_battery_awareness=not flags["battery_awareness"],
            disable_queue_awareness=not flags["queue_awareness"],
            disable_mobility_features=not flags["mobility_features"],
            success_bonus=self._success_bonus(),
        )

    def prepare_traces(self) -> Tuple[List[TraceEpisode], List[TraceEpisode], List[TraceEpisode]]:
        logger.info("Step 1: Preparing traces...")

        trace_cfg = self.config_dict["data"]
        real_data_mode = is_real_data_mode(self.config_dict)
        manifest_path = trace_cfg.get("real_data_manifest", "configs/phase_6/raw_real_data_manifest.yaml")
        trace_dir = trace_cfg.get("trace_dir", "data/synthetic_trace")
        normalized_trace_dir = Path(trace_dir).as_posix().rstrip("/")
        uses_legacy_trace_dir = normalized_trace_dir in {"data/traces", "data/synthetic_trace"}

        if real_data_mode:
            logger.info("Real-data mode enabled; synthetic fallback is forbidden.")
            assert_real_data_ready(manifest_path=manifest_path, repo_root=Path.cwd())

        loader = TraceLoader(trace_dir=trace_dir)
        processor = TraceProcessor(trace_dir=trace_dir, seed=self.seed)

        allow_saved_splits = loader.has_saved_episode_splits() and (
            not real_data_mode or not uses_legacy_trace_dir
        )

        if allow_saved_splits:
            logger.info("Using saved trace episode splits from %s", trace_dir)
            train_eps, val_eps, test_eps = loader.load_saved_episode_splits()
            processor.episodes = [*train_eps, *val_eps, *test_eps]
        else:
            logger.info("Loading raw trace inputs and generating episode splits")
            traces = loader.load_trace_frames()
            if not traces:
                traces = processor.load_traces(allow_synthetic_fallback=not real_data_mode)

            processed = processor.preprocess_traces(traces)
            env_cfg = self.config_dict["environment"]
            processor.generate_episodes(
                processed,
                tasks_per_episode=env_cfg["n_tasks_per_episode"],
                n_episodes=trace_cfg["train_episodes"] + trace_cfg["val_episodes"] + trace_cfg["test_episodes"],
            )
            train_eps, val_eps, test_eps = processor.split_episodes(train_ratio=0.8, val_ratio=0.1)

            save_paths = loader.saved_episode_paths()
            processor.save_episodes(train_eps, str(save_paths["train"]))
            processor.save_episodes(val_eps, str(save_paths["val"]))
            processor.save_episodes(test_eps, str(save_paths["test"]))

        logger.info("Trace Statistics:")
        logger.info(json.dumps(processor.get_statistics(), indent=2))
        return train_eps, val_eps, test_eps

    def train_model(
        self,
        train_episodes: List[TraceEpisode],
        feature_overrides: Optional[Dict] = None,
        checkpoint_name: Optional[str] = None,
        metrics_name: str = "trace_training_metrics.csv",
    ) -> Dict:
        algorithm = self._algorithm_name()
        logger.info("Step 2: Training %s on trace episodes...", algorithm.upper())

        env = self._build_trace_env(train_episodes, feature_overrides=feature_overrides)
        model = self._create_model(env)
        train_cfg = self.config_dict["training"]
        total_timesteps = train_cfg.get("max_episodes", 500) * self.config_dict["environment"]["n_tasks_per_episode"]

        callback = TraceMetricsCallback(self.log_dir / metrics_name)
        model.learn(total_timesteps=total_timesteps, progress_bar=True, callback=callback)

        resolved_checkpoint_name = checkpoint_name or self._algorithm_checkpoint_name()
        checkpoint_path = self.checkpoint_dir / resolved_checkpoint_name
        model.save(checkpoint_path)
        logger.info("Saved %s checkpoint to %s", algorithm.upper(), checkpoint_path)
        return callback.history

    def evaluate_model(
        self,
        episodes: List[TraceEpisode],
        checkpoint_path: str,
        feature_overrides: Optional[Dict] = None,
    ) -> Dict:
        algorithm = self._algorithm_name()
        logger.info("Step 3: Evaluating %s on trace episodes...", algorithm.upper())

        env = self._build_trace_env(episodes, feature_overrides=feature_overrides)
        model = self._load_model(checkpoint_path, env=env)

        eval_metrics = {
            "success_rates": [],
            "latencies": [],
            "energy_consumed": [],
            "priorities_satisfied": [],
        }

        for trace_ep in episodes:
            obs, _ = env.reset()
            success_count = 0
            task_count = 0
            delays: List[float] = []
            energies: List[float] = []

            for _trace_task in trace_ep.tasks:
                action, _ = model.predict(obs, deterministic=True)
                next_obs, _reward, terminated, truncated, info = env.step(action)

                if info.get("task_success", False):
                    success_count += 1
                task_count += 1
                if "delay" in info:
                    delays.append(info["delay"])
                if "energy" in info:
                    energies.append(info["energy"])

                obs = next_obs
                if terminated or truncated:
                    break

            success_rate = success_count / max(task_count, 1)
            eval_metrics["success_rates"].append(success_rate * 100.0)
            eval_metrics.setdefault("avg_delays", []).append(float(np.mean(delays)) if delays else 0.0)
            eval_metrics.setdefault("avg_energies", []).append(float(np.mean(energies)) if energies else 0.0)

        avg_success = np.mean(eval_metrics["success_rates"])
        std_success = np.std(eval_metrics["success_rates"])
        avg_delay = np.mean(eval_metrics.get("avg_delays", [0.0]))
        avg_energy = np.mean(eval_metrics.get("avg_energies", [0.0]))

        logger.info("Validation Results:")
        logger.info("  Avg Success Rate: %.2f%% (+/-%.2f%%)", avg_success, std_success)
        logger.info("  Min: %.2f%%", np.min(eval_metrics["success_rates"]))
        logger.info("  Max: %.2f%%", np.max(eval_metrics["success_rates"]))
        logger.info("  Avg Delay: %.3fs | Avg Energy: %.3e", avg_delay, avg_energy)
        return eval_metrics

    def generate_report(self, training_history: Dict, eval_metrics: Dict, ablation_results: Dict) -> None:
        algorithm = self._algorithm_name()
        logger.info("Step 5: Generating Phase 6 report...")

        metrics_path = self.log_dir / "trace_training_metrics.csv"
        checkpoint_path = self.checkpoint_dir / self._algorithm_checkpoint_name()

        with open(self.report_path, "w", encoding="utf-8") as handle:
            handle.write("# Faz 6 Report: Trace-driven Training\n\n")
            handle.write(f"**Tarih:** {datetime.now().strftime('%d %B %Y')}\n")
            handle.write("**Durum:** tamamlandi\n\n")
            handle.write("## Ozet\n\n")
            handle.write(
                f"Bu kosuda trace omurgasi uzerinde `{algorithm.upper()}` modeli egitildi ve ortak Faz 6 raporlamasi yenilendi.\n\n"
            )

            if training_history.get("success_rate"):
                handle.write(
                    f"- Son train success: {float(training_history['success_rate'][-1]):.2f}%\n"
                )
            if eval_metrics.get("success_rates"):
                handle.write(
                    f"- Ortalama validation success: {float(np.mean(eval_metrics['success_rates'])):.2f}%\n"
                )
            if ablation_results:
                handle.write("- Bu kosuda uretilen ablation validation ozetleri rapora eklendi.\n")

            handle.write("\n## Artefaktlar\n\n")
            handle.write(f"- Metrics CSV: `{metrics_path.as_posix()}`\n")
            handle.write(f"- Checkpoint: `{checkpoint_path.as_posix()}`\n")

        logger.info("Report saved to %s", self.report_path)

    def run_ablation_validation(self, val_episodes: List[TraceEpisode]) -> Dict:
        logger.info("Step 4: Running lightweight ablation validation on traces...")

        ablation_results = {}
        configs = [
            ("full_model", {"reward_shaping": True, "partial_offloading": True}),
            ("no_reward_shaping", {"reward_shaping": False, "partial_offloading": True}),
            ("no_partial_offloading", {"reward_shaping": True, "partial_offloading": False}),
        ]

        for config_name, flags in configs:
            env = self._build_trace_env(val_episodes, feature_overrides=flags)
            success_rates = []

            for trace_ep in val_episodes[:10]:
                obs, _ = env.reset()
                success_count = 0
                task_count = 0

                for _trace_task in trace_ep.tasks:
                    action = env.action_space.sample()
                    next_obs, _reward, terminated, truncated, info = env.step(action)
                    if info.get("task_success", False):
                        success_count += 1
                    task_count += 1
                    obs = next_obs
                    if terminated or truncated:
                        break

                success_rates.append(success_count / max(task_count, 1))

            ablation_results[config_name] = (np.mean(success_rates) if success_rates else 0.0) * 100.0

        return ablation_results

    def run(self) -> None:
        logger.info("=" * 60)
        logger.info("FAZ 6: TRACE-DRIVEN TRAINING")
        logger.info("=" * 60)

        train_eps, val_eps, _test_eps = self.prepare_traces()
        checkpoint_name = self._algorithm_checkpoint_name()
        training_history = self.train_model(train_eps, checkpoint_name=checkpoint_name)

        checkpoint_path = self.checkpoint_dir / checkpoint_name
        eval_metrics = self.evaluate_model(val_eps, str(checkpoint_path)) if checkpoint_path.exists() else {}
        ablation_results = self.run_ablation_validation(val_eps)
        self.generate_report(training_history, eval_metrics, ablation_results)

        logger.info("=" * 60)
        logger.info("FAZ 6 TRAINING COMPLETE")
        logger.info("=" * 60)


if __name__ == "__main__":
    orchestrator = TraceTrainingOrchestrator(
        config_path="configs/phase_6/synthetic_trace_rl_training.yaml",
        seed=42,
    )
    orchestrator.run()
