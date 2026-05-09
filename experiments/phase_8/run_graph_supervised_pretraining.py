from pathlib import Path
import argparse
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.training.pretrain_graph_policy import run_graph_supervised_pretraining


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Phase 8 graph policy supervised warm-start")
    parser.add_argument(
        "--config",
        default="configs/phase_8/graph_supervised_pretraining.yaml",
        help="Graph supervised pretraining config path",
    )
    parser.add_argument(
        "--fusion",
        default=None,
        choices=["none", "input", "late", "input_late"],
        help="Optional semantic prior fusion mode override",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional seed override")
    parser.add_argument("--output_suffix", default=None, help="Optional suffix for checkpoint/report/metrics outputs")
    parser.add_argument("--write_metrics", action="store_true", help="Write per-epoch metrics CSV for debugging")
    parser.add_argument("--write_report", action="store_true", help="Write optional single-run markdown report")
    args = parser.parse_args()

    result = run_graph_supervised_pretraining(
        args.config,
        semantic_prior_fusion=args.fusion,
        seed_override=args.seed,
        output_suffix=args.output_suffix,
        write_report=args.write_report,
        write_metrics=args.write_metrics,
    )
    print(f"[INFO] Seed: {result['seed']}")
    print(f"[INFO] Teacher policy: {result['teacher_policy']}")
    print(f"[INFO] Semantic prior fusion: {result['semantic_prior_fusion']}")
    print(f"[INFO] Samples: {result['num_samples']}")
    print(f"[INFO] Executed epochs: {result['executed_epochs']}")
    print(f"[INFO] Best epoch: {result['best_epoch']}")
    print(f"[INFO] Best val accuracy: {float(result['best_val_accuracy']) * 100:.2f}%")
    print(f"[INFO] Test accuracy: {float(result['final_test']['accuracy']) * 100:.2f}%")
    print(f"[INFO] Test prediction diversity: {float(result['final_test']['prediction_diversity']):.4f}")
    if result["metrics_csv"]:
        print(f"[INFO] Metrics CSV: {result['metrics_csv']}")
    if args.write_report:
        print(f"[INFO] Report: {result['report_path']}")
    print(f"[INFO] Checkpoint: {result['checkpoint_path']}")

