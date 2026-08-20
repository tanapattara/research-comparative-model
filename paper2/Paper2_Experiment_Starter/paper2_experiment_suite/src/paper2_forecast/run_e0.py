from __future__ import annotations

import argparse
from pathlib import Path

from .e0 import write_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Paper 2 E0 real-data audit gate")
    parser.add_argument("--config", type=Path, default=Path("configs/e0.yaml"))
    parser.add_argument(
        "--allow-blocked-exit-zero",
        action="store_true",
        help="Write diagnostics but return zero even when the E0 modelling gate is blocked.",
    )
    arguments = parser.parse_args()
    manifest, manifest_path, report_path = write_outputs(arguments.config)
    print(f"E0_STATUS={manifest['status']}")
    print(f"BLOCKING_ISSUES={len(manifest['blocking_issues'])}")
    print(f"MANIFEST={manifest_path}")
    print(f"REPORT={report_path}")
    if manifest["status"] != "PASS" and not arguments.allow_blocked_exit_zero:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
