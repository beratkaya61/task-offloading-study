from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.training.pretrain_policy import run_supervised_pretraining


if __name__ == "__main__":
    result = run_supervised_pretraining("configs/synthetic/supervised_pretraining.yaml")
    print(f"[INFO] Metrics CSV: {result['metrics_csv']}")
    print(f"[INFO] Report: {result['report_path']}")
    print(f"[INFO] Checkpoint: {result['checkpoint_path']}")
    print(f"[INFO] Best epoch: {result['best_epoch']}")
    print(f"[INFO] Best val accuracy: {result['best_val_accuracy']}%")
    print(f"[INFO] Test accuracy: {result['test_accuracy']}%")
