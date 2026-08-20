from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from paper2_forecast.config import load_config
from paper2_forecast.e0 import audit_processed_frame, audit_thresholds, build_manifest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
E0_CONFIG = PROJECT_ROOT / "configs" / "e0.yaml"


class E0AuditTests(unittest.TestCase):
    def test_e0_reports_dates_and_missing_target_without_imputation(self) -> None:
        frame = pd.DataFrame(
            {
                "date": ["2020-01-01", "2020-01-03", "2020-01-03"],
                "CSA": [1.0, np.nan, 3.0],
                "LUA": [1.0, 2.0, 3.0],
                "CKH": [1.0, 2.0, 3.0],
                "VIE": [1.0, 2.0, 3.0],
                "NON": [1.0, np.nan, 3.0],
            }
        )
        audit = audit_processed_frame(
            frame,
            expected_columns=["date", "CSA", "LUA", "CKH", "VIE", "NON"],
            date_column="date",
            stations=["CSA", "LUA", "CKH", "VIE", "NON"],
        )
        self.assertEqual(audit["duplicate_dates"], 1)
        self.assertEqual(audit["missing_calendar_days"], 1)
        self.assertEqual(audit["stations"]["NON"]["missing_or_non_numeric"], 1)
        self.assertTrue(pd.isna(frame.loc[1, "NON"]))

    def test_e0_rejects_schema_reordering(self) -> None:
        frame = pd.DataFrame(columns=["date", "NON", "CSA", "LUA", "CKH", "VIE"])
        audit = audit_processed_frame(
            frame,
            expected_columns=["date", "CSA", "LUA", "CKH", "VIE", "NON"],
            date_column="date",
            stations=["CSA", "LUA", "CKH", "VIE", "NON"],
        )
        self.assertFalse(audit["exact_schema_and_order"])

    def test_e0_detects_alarm_datum_inconsistency(self) -> None:
        values = load_config(E0_CONFIG)["nong_khai_thresholds"]
        audit = audit_thresholds(values)
        self.assertAlmostEqual(audit.derived_alarm_gauge_height_m, 11.4)
        self.assertAlmostEqual(audit.alarm_difference_m, 0.4)
        self.assertFalse(audit.alarm_consistent)
        self.assertTrue(audit.flood_consistent)

    def test_e0_real_dataset_is_blocked_by_pre_split_interpolation(self) -> None:
        manifest = build_manifest(E0_CONFIG)
        codes = {issue["code"] for issue in manifest["blocking_issues"]}
        self.assertEqual(manifest["status"], "BLOCKED")
        self.assertIn("GLOBAL_INTERPOLATION_PROVENANCE", codes)
        self.assertIn("RAW_MISSINGNESS_REQUIRES_POLICY", codes)
        self.assertIn("NON_ALARM_THRESHOLD_INCONSISTENT", codes)

    def test_e0_config_forbids_target_imputation_and_test_clipping(self) -> None:
        policies = load_config(E0_CONFIG)["policies"]
        self.assertFalse(policies["interpolate_before_split"])
        self.assertFalse(policies["impute_missing_target"])
        self.assertEqual(policies["scaler_fit_scope"], "train_only")
        self.assertFalse(policies["clip_validation_test"])


if __name__ == "__main__":
    unittest.main()
