from __future__ import annotations

import argparse
from pathlib import Path

from .e0_resolved import write_resolved_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run resolved Paper 2 E0 gate")
    parser.add_argument("--config", type=Path, default=Path("configs/e0_resolved.yaml"))
    arguments = parser.parse_args()
    manifest, manifest_path, report_path = write_resolved_outputs(arguments.config)
    print(f"E0_STATUS={manifest['status']}")
    print(f"BLOCKING_ISSUES={len(manifest['blocking_issues'])}")
    print(f"MANIFEST={manifest_path}")
    print(f"REPORT={report_path}")
    if manifest["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
