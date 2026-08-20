from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from paper2_forecast.availability import filter_complete_case_origins
from paper2_forecast.e0_resolved import build_resolved_manifest
from paper2_forecast.raw_dataset import build_raw_preserved_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parents[2]
RAW_DIRECTORY = REPOSITORY_ROOT / "P1" / "data"
RESOLVED_CONFIG = PROJECT_ROOT / "configs" / "e0_resolved.yaml"


class E0ResolvedPolicyTests(unittest.TestCase):
    def test_resolved_real_data_gate_passes(self) -> None:
        manifest = build_resolved_manifest(RESOLVED_CONFIG)
        self.assertEqual(manifest["status"], "PASS")
        self.assertEqual(manifest["blocking_issues"], [])
        audit = manifest["dataset"]["processed_audit"]
        self.assertEqual(audit["start"], "2007-01-01")
        self.assertEqual(audit["end"], "2025-11-11")
        self.assertEqual(audit["missing_calendar_days"], 0)

    def test_builder_preserves_sources_and_missing_target(self) -> None:
        stations = ("CSA", "LUA", "CKH", "VIE", "NON")
        before = {station: (RAW_DIRECTORY / f"{station}.csv").read_bytes() for station in stations}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "derived.csv"
            manifest_path = Path(directory) / "manifest.json"
            frame, manifest = build_raw_preserved_dataset(
                RAW_DIRECTORY,
                output,
                manifest_path,
                start="2014-01-01",
                end="2014-12-31",
            )
            self.assertTrue(manifest["sources_unchanged"])
            self.assertFalse(manifest["interpolation_applied"])
            self.assertFalse(manifest["target_imputation_applied"])
            self.assertGreater(int(frame["NON"].isna().sum()), 0)
        after = {station: (RAW_DIRECTORY / f"{station}.csv").read_bytes() for station in stations}
        self.assertEqual(before, after)

    def test_missing_input_or_any_missing_target_removes_origin(self) -> None:
        frame = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=12, freq="D"),
                "CSA": np.arange(12, dtype=float),
                "NON": np.arange(12, dtype=float),
            }
        )
        candidates = np.array([3, 4, 5, 6])
        frame.loc[1, "CSA"] = np.nan
        frame.loc[8, "NON"] = np.nan
        kept = filter_complete_case_origins(
            frame,
            candidates,
            stations=("CSA", "NON"),
            target="NON",
            lookback_days=3,
            output_days=2,
        )
        self.assertEqual(kept.tolist(), [4, 5])
        self.assertTrue(pd.isna(frame.loc[8, "NON"]))


if __name__ == "__main__":
    unittest.main()
