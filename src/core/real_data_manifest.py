from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import yaml


DEFAULT_MANIFEST_PATH = Path("configs/phase_6/raw_real_data_manifest.yaml")


def load_real_data_manifest(manifest_path: str | Path = DEFAULT_MANIFEST_PATH) -> dict:
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Real-data manifest not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def required_dataset_statuses(manifest: dict) -> Dict[str, str]:
    datasets = manifest.get("datasets", {})
    statuses: Dict[str, str] = {}
    for dataset_name, payload in datasets.items():
        if payload.get("required_for_real_data_claim", False):
            statuses[dataset_name] = str(payload.get("status", "not_downloaded"))
    return statuses


def required_dataset_roots(manifest: dict) -> Dict[str, Path]:
    datasets = manifest.get("datasets", {})
    roots: Dict[str, Path] = {}
    for dataset_name, payload in datasets.items():
        if payload.get("required_for_real_data_claim", False):
            roots[dataset_name] = Path(payload.get("local_root", ""))
    return roots


def dataset_expected_globs(manifest: dict, required_only: bool = False) -> Dict[str, List[str]]:
    datasets = manifest.get("datasets", {})
    globs: Dict[str, List[str]] = {}
    for dataset_name, payload in datasets.items():
        if required_only and not payload.get("required_for_real_data_claim", False):
            continue
        globs[dataset_name] = list(payload.get("expected_globs", []))
    return globs


def synthetic_fallback_allowed(manifest: dict) -> bool:
    return bool(manifest.get("real_data_policy", {}).get("synthetic_fallback_allowed", True))


def missing_required_datasets(manifest: dict) -> Dict[str, str]:
    missing: Dict[str, str] = {}
    for dataset_name, status in required_dataset_statuses(manifest).items():
        if status not in {"downloaded", "validated"}:
            missing[dataset_name] = status
    return missing


def missing_required_roots(manifest: dict, repo_root: str | Path = ".") -> Dict[str, str]:
    root_base = Path(repo_root)
    missing: Dict[str, str] = {}
    for dataset_name, local_root in required_dataset_roots(manifest).items():
        if not local_root:
            missing[dataset_name] = "missing_local_root"
            continue
        full_path = root_base / local_root
        if not full_path.exists():
            missing[dataset_name] = f"path_not_found:{local_root.as_posix()}"
    return missing


def missing_required_files(manifest: dict, repo_root: str | Path = ".") -> Dict[str, str]:
    root_base = Path(repo_root)
    missing: Dict[str, str] = {}
    roots = required_dataset_roots(manifest)
    globs = dataset_expected_globs(manifest, required_only=True)

    for dataset_name, local_root in roots.items():
        full_path = root_base / local_root
        if not full_path.exists():
            continue

        patterns = globs.get(dataset_name, [])
        if not patterns:
            continue

        has_match = False
        for pattern in patterns:
            if any(full_path.glob(pattern)):
                has_match = True
                break

        if not has_match:
            missing[dataset_name] = f"no_files_matching:{'|'.join(patterns)}"

    return missing


def dataset_inventory(manifest: dict, repo_root: str | Path = ".") -> List[dict]:
    root_base = Path(repo_root)
    rows: List[dict] = []
    datasets = manifest.get("datasets", {})

    for dataset_name, payload in datasets.items():
        local_root = Path(payload.get("local_root", ""))
        full_path = root_base / local_root if local_root else None
        expected_patterns = list(payload.get("expected_globs", []))
        matched_files = 0
        if full_path and full_path.exists():
            seen = set()
            for pattern in expected_patterns:
                for match in full_path.glob(pattern):
                    if match.is_file():
                        seen.add(match.resolve())
            matched_files = len(seen)

        rows.append(
            {
                "dataset": dataset_name,
                "required": bool(payload.get("required_for_real_data_claim", False)),
                "status": str(payload.get("status", "not_downloaded")),
                "local_root": local_root.as_posix() if local_root else "",
                "path_exists": bool(full_path and full_path.exists()),
                "matched_files": matched_files,
                "expected_globs": expected_patterns,
            }
        )
    return rows


def assert_real_data_ready(manifest_path: str | Path = DEFAULT_MANIFEST_PATH, repo_root: str | Path = ".") -> dict:
    manifest = load_real_data_manifest(manifest_path)

    if synthetic_fallback_allowed(manifest):
        raise RuntimeError(
            "Real-data manifest allows synthetic fallback. "
            "Set real_data_policy.synthetic_fallback_allowed=false before running real-data mode."
        )

    missing_status = missing_required_datasets(manifest)
    if missing_status:
        detail = ", ".join(f"{name}={status}" for name, status in missing_status.items())
        raise RuntimeError(f"Required real datasets are not ready in manifest: {detail}")

    missing_roots = missing_required_roots(manifest, repo_root=repo_root)
    if missing_roots:
        detail = ", ".join(f"{name}={reason}" for name, reason in missing_roots.items())
        raise RuntimeError(f"Required real dataset paths are missing: {detail}")

    missing_files = missing_required_files(manifest, repo_root=repo_root)
    if missing_files:
        detail = ", ".join(f"{name}={reason}" for name, reason in missing_files.items())
        raise RuntimeError(f"Required real dataset files are missing: {detail}")

    return manifest


def is_real_data_mode(config: dict) -> bool:
    env_cfg = config.get("environment", {})
    data_cfg = config.get("data", {})
    trace_source = str(env_cfg.get("trace_source", "")).strip().lower()
    return bool(
        data_cfg.get("real_data_mode", False)
        or env_cfg.get("real_data_mode", False)
        or trace_source == "real_data"
    )

