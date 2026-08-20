from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper 2 phase safety gate")
    parser.add_argument("--phase", required=True, choices=["E1", "E2", "E3", "E4", "E5"])
    arguments = parser.parse_args()
    raise SystemExit(
        f"{arguments.phase} is intentionally disabled: complete E0 with status PASS and obtain "
        "explicit approval before implementing or training models."
    )


if __name__ == "__main__":
    main()
