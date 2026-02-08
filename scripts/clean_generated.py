"""
Cleanup generated artifacts.

Usage examples:
    python scripts/clean_generated.py --all --dry-run
    python scripts/clean_generated.py --all
    python scripts/clean_generated.py --models --results --pycache
"""

import argparse
import shutil
from pathlib import Path


def dir_size_bytes(path: Path) -> int:
    """Return recursive size for a directory in bytes."""
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            try:
                total += item.stat().st_size
            except OSError:
                # Ignore transient files that disappear during scan.
                pass
    return total


def human_size(num_bytes: int) -> str:
    """Format bytes in a human-readable form."""
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def collect_targets(root: Path, remove_models: bool, remove_results: bool, remove_feature_json: bool, remove_pycache: bool):
    """Collect files/directories to remove."""
    targets = []

    if remove_models:
        models_dir = root / "models"
        if models_dir.exists():
            targets.extend(sorted([p for p in models_dir.iterdir() if p.is_dir()]))

    if remove_results:
        results_dir = root / "results"
        if results_dir.exists():
            targets.extend(sorted([p for p in results_dir.iterdir() if p.is_dir()]))

    if remove_feature_json:
        config_dir = root / "config"
        if config_dir.exists():
            targets.extend(sorted(config_dir.glob("feature_*.json")))

    if remove_pycache:
        targets.extend(sorted(root.rglob("__pycache__")))

    # De-duplicate while preserving order.
    seen = set()
    unique_targets = []
    for target in targets:
        target_str = str(target.resolve())
        if target_str not in seen:
            seen.add(target_str)
            unique_targets.append(target)

    return unique_targets


def delete_targets(targets, dry_run: bool):
    """Delete target paths and return deletion summary."""
    deleted_count = 0
    reclaimed_bytes = 0

    for target in targets:
        if not target.exists():
            continue

        size = dir_size_bytes(target) if target.is_dir() else target.stat().st_size

        if dry_run:
            print(f"[DRY-RUN] Would remove: {target}")
        else:
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
            print(f"Removed: {target}")

        deleted_count += 1
        reclaimed_bytes += size

    return deleted_count, reclaimed_bytes


def main():
    parser = argparse.ArgumentParser(description="Remove generated artifacts that bloat the project directory")
    parser.add_argument("--models", action="store_true", help="Remove run directories under models/")
    parser.add_argument("--results", action="store_true", help="Remove run directories under results/")
    parser.add_argument("--feature-json", action="store_true", help="Remove config/feature_*.json files")
    parser.add_argument("--pycache", action="store_true", help="Remove all __pycache__/ directories")
    parser.add_argument("--all", action="store_true", help="Remove all generated artifacts listed above")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be removed without deleting")
    args = parser.parse_args()

    if args.all:
        args.models = True
        args.results = True
        args.feature_json = True
        args.pycache = True

    if not any([args.models, args.results, args.feature_json, args.pycache]):
        parser.error("No cleanup target selected. Use --all or choose one/more target flags.")

    root = Path(__file__).resolve().parent.parent
    targets = collect_targets(
        root=root,
        remove_models=args.models,
        remove_results=args.results,
        remove_feature_json=args.feature_json,
        remove_pycache=args.pycache,
    )

    if not targets:
        print("Nothing to clean.")
        return

    print(f"Found {len(targets)} target(s).")
    count, reclaimed = delete_targets(targets, dry_run=args.dry_run)

    action = "Would remove" if args.dry_run else "Removed"
    print(f"\n{action} {count} item(s), reclaiming ~{human_size(reclaimed)}")


if __name__ == "__main__":
    main()
