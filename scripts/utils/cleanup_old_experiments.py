"""Cleanup old experiment directories.

Usage:
    uv run python scripts/cleanup_old_experiments.py --older-than 7 [--dry-run]
    uv run python scripts/cleanup_old_experiments.py --list
"""
import argparse
import shutil
from datetime import datetime, timedelta
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Clean up old experiment directories")
    parser.add_argument("--output-dir", default="outputs/", help="Base output directory")
    parser.add_argument("--older-than", type=int, help="Delete experiments older than N days")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be deleted")
    parser.add_argument("--list", action="store_true", help="List all experiments with ages")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(f"Output directory does not exist: {output_dir}")
        return

    # Find experiment dirs (format: YYYYMMDD_HHMMSS_*)
    exp_dirs = sorted(output_dir.glob("20??????_??????_*"), key=lambda d: d.name)

    if args.list or args.older_than is None:
        for d in exp_dirs:
            if d.is_dir():
                size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / (1024**2)
                print(f"{d.name}  {size:.1f} MB")
        print(f"\nTotal: {len(exp_dirs)} experiments")
        return

    cutoff = datetime.now() - timedelta(days=args.older_than)
    deleted = 0
    for d in exp_dirs:
        if not d.is_dir():
            continue
        try:
            ts = datetime.strptime(d.name[:15], "%Y%m%d_%H%M%S")
        except ValueError:
            continue
        if ts < cutoff:
            if args.dry_run:
                print(f"Would delete: {d.name}")
            else:
                shutil.rmtree(d, ignore_errors=True)
                print(f"Deleted: {d.name}")
            deleted += 1
    print(f"\n{'Would delete' if args.dry_run else 'Deleted'}: {deleted} experiments")


if __name__ == "__main__":
    main()
