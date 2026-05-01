from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
VENDOR = ROOT / ".vendor"
if VENDOR.exists():
    sys.path.insert(0, str(VENDOR))

from project2 import run_full_project2_backtest


def main() -> None:
    project1_dir = ROOT / "Project1"
    output_dir = ROOT / "Project2_outputs" / "full"
    run_full_project2_backtest(project1_dir=project1_dir, output_dir=output_dir)
    print(f"Full Project2 outputs exported to: {output_dir}")


if __name__ == "__main__":
    main()
