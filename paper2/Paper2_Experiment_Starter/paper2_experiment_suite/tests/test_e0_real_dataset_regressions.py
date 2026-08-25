from __future__ import annotations

import unittest
from pathlib import Path

from paper2_forecast.e0 import build_manifest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
E0_CONFIG = PROJECT_ROOT / "configs" / "e0.yaml"


class E0RealDatasetRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = build_manifest(E0_CONFIG)

    def test_real_processed_schema_and_date_gaps_are_locked(self) -> None:
        audit = self.manifest["dataset"]["processed_audit"]
        self.assertEqual(audit["actual_columns"], ["date", "CSA", "LUA", "CKH", "VIE", "NON"])
        self.assertEqual(audit["rows"], 7011)
        self.assertEqual(audit["start"], "2006-01-01")
        self.assertEqual(audit["end"], "2025-11-11")
        self.assertEqual(audit["missing_calendar_days"], 244)
        self.assertEqual(audit["duplicate_dates"], 0)

    def test_raw_missingness_cannot_be_hidden_by_processed_table(self) -> None:
        raw = self.manifest["raw_sources"]
        self.assertEqual(raw["common_range"]["days_with_any_station_missing"], 590)
        self.assertGreater(raw["stations"]["NON"]["zero_value_missing_candidates"], 0)
        self.assertGreater(raw["stations"]["NON"]["blank_or_non_numeric_missing_candidates"], 0)

    def test_final_period_uses_actual_end_and_full_target_purge(self) -> None:
        final_fold = next(
            fold for fold in self.manifest["fold_coverage"] if fold["name"] == "final_period"
        )
        test = final_fold["partitions"]["test"]
        self.assertFalse(test["fully_covered"])
        self.assertEqual(test["effective_end"], "2025-11-11")
        self.assertEqual(test["latest_issue_date_after_14_day_purge"], "2025-10-28")


if __name__ == "__main__":
    unittest.main()
