from __future__ import annotations

import argparse
from pathlib import Path

from .phase_experiments import run_phase


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a gated Paper 2 experiment phase")
    parser.add_argument("--phase", required=True, choices=["E2", "E3", "E4", "E5"])
    parser.add_argument("--config", type=Path, default=Path("configs/e2_e5.yaml"))
    arguments = parser.parse_args()
    run_phase(arguments.phase, arguments.config)


if __name__ == "__main__":
    main()
