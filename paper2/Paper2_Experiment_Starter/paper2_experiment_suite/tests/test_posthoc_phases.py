import unittest

from paper2_forecast.phase_experiments import _post_hoc_metadata


class PostHocPhaseTests(unittest.TestCase):
    def test_frozen_prerequisite_adds_no_post_hoc_metadata(self) -> None:
        self.assertEqual(_post_hoc_metadata({"post_hoc": False}, phase="E4"), {})

    def test_e4_metadata_records_validation_only_provenance(self) -> None:
        metadata = _post_hoc_metadata({"post_hoc": True}, phase="E4")
        self.assertTrue(metadata["post_hoc"])
        self.assertFalse(metadata["test_values_read"])
        self.assertTrue(metadata["does_not_replace_frozen_e2_e5"])

    def test_e5_metadata_records_test_reuse(self) -> None:
        metadata = _post_hoc_metadata({"post_hoc": True}, phase="E5")
        self.assertTrue(metadata["test_results_already_opened_before_run"])
        self.assertEqual(metadata["interpretation"], "exploratory_test_reuse")


if __name__ == "__main__":
    unittest.main()
