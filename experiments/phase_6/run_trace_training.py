#!/usr/bin/env python3
"""
Phase 6 trace-training CLI.

Config'e gore sentetik trace ya da real-composite-trace RL kosusunu baslatir.
"""

from __future__ import annotations

import argparse

from experiments.phase_6.train_trace_rl import TraceTrainingOrchestrator


def main() -> None:
    parser = argparse.ArgumentParser(description="Run trace-based RL training")
    parser.add_argument(
        "--config",
        default="configs/phase_6/real_composite_trace_rl_training.yaml",
        help="Trace RL config path",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    orchestrator = TraceTrainingOrchestrator(config_path=args.config, seed=args.seed)
    orchestrator.run()


if __name__ == "__main__":
    main()
