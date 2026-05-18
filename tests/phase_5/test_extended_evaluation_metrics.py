import unittest

import numpy as np

from src.core.evaluation import EXPERIMENT_LOG_COLUMNS, build_experiment_log_entry, evaluate_policy, summarize_step_logs


class TinyPolicy:
    def __init__(self):
        self._actions = [1, 5]
        self._index = 0

    def predict(self, obs, deterministic=True):
        action = self._actions[self._index % len(self._actions)]
        self._index += 1
        return action, None


class TinyEnv:
    def reset(self):
        self._step = 0
        return np.zeros(12, dtype=np.float32), {}

    def step(self, action):
        self._step += 1
        success = self._step == 1
        info = {
            "task_success": success,
            "delay": 0.2 if success else 0.8,
            "deadline": 0.5,
            "energy": 1.0 if success else 3.0,
            "queue_delay": 0.01,
            "battery_empty": False,
            "partial_offload": action in (1, 2, 3),
        }
        return np.zeros(12, dtype=np.float32), 5.0, self._step >= 2, False, info


class ExtendedEvaluationMetricsTest(unittest.TestCase):
    def test_empty_logs_return_zero_metrics(self):
        summary = summarize_step_logs([], total_reward=0.0, action_counts={index: 0 for index in range(6)})

        self.assertEqual(summary["config_total_tasks"], 0)
        self.assertEqual(summary["metric_success_rate"], 0.0)
        self.assertEqual(summary["metric_deadline_miss_ratio"], 0.0)
        self.assertEqual(summary["metric_avg_latency"], 0.0)
        self.assertEqual(summary["metric_energy_per_success"], 0.0)

    def test_success_failure_mix_metrics(self):
        logs = [
            {
                "success": True,
                "delay": 0.2,
                "deadline": 0.5,
                "energy": 1.0,
                "queue_delay": 0.01,
                "battery_empty": False,
                "partial_offload": True,
                "action": 1,
                "decision_overhead_ms": 2.0,
            },
            {
                "success": False,
                "delay": 0.8,
                "deadline": 0.5,
                "energy": 3.0,
                "queue_delay": 0.03,
                "battery_empty": True,
                "partial_offload": False,
                "action": 0,
                "decision_overhead_ms": 4.0,
            },
        ]

        summary = summarize_step_logs(logs, total_reward=10.0, action_counts={0: 1, 1: 1, 2: 0, 3: 0, 4: 0, 5: 0})

        self.assertEqual(summary["config_total_tasks"], 2)
        self.assertEqual(summary["metric_success_rate"], 0.5)
        self.assertEqual(summary["metric_deadline_miss_ratio"], 0.5)
        self.assertEqual(summary["metric_avg_latency"], 0.5)
        self.assertEqual(summary["metric_avg_energy"], 2.0)
        self.assertEqual(summary["metric_energy_per_success"], 4.0)
        self.assertEqual(summary["metric_cvar95_latency"], 0.8)
        self.assertEqual(summary["metric_avg_deadline_slack"], 0.15)
        self.assertEqual(summary["metric_avg_deadline_overrun"], 0.3)
        self.assertEqual(summary["metric_action_0_success_rate"], 0.0)
        self.assertEqual(summary["metric_action_1_success_rate"], 1.0)
        self.assertGreater(summary["metric_action_entropy"], 0.0)
        self.assertGreater(summary["metric_action_jain_fairness"], 0.0)
        self.assertEqual(summary["metric_battery_depletion_rate"], 0.5)
        self.assertEqual(summary["metric_partial_offload_ratio"], 0.5)
        self.assertEqual(summary["metric_decision_overhead_ms"], 3.0)

    def test_log_entry_uses_canonical_columns(self):
        entry = build_experiment_log_entry(
            run_name="PPO",
            semantic_mode="action_prior",
            config_seed=42,
            total_reward=0.0,
            action_counts={index: 0 for index in range(6)},
            step_logs=[],
            extra_fields={"config_eval_group": "unit_test"},
        )

        self.assertEqual(list(entry.keys()), EXPERIMENT_LOG_COLUMNS)
        self.assertEqual(entry["config_model_type"], "PPO")
        self.assertEqual(entry["config_eval_group"], "unit_test")

    def test_evaluate_policy_returns_extended_canonical_columns(self):
        entry = evaluate_policy(
            TinyEnv(),
            TinyPolicy(),
            num_episodes=1,
            run_name="Tiny",
            semantic_mode="unit",
            csv_path=None,
        )

        self.assertEqual(list(entry.keys()), EXPERIMENT_LOG_COLUMNS)
        self.assertEqual(entry["metric_success_rate"], 0.5)
        self.assertEqual(entry["metric_deadline_miss_ratio"], 0.5)
        self.assertEqual(entry["metric_partial_offload_ratio"], 0.5)
        self.assertGreaterEqual(entry["metric_decision_overhead_ms"], 0.0)


if __name__ == "__main__":
    unittest.main()
