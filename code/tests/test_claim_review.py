"""Fast unit tests that do not require network access or an API key."""

from __future__ import annotations

import unittest
from pathlib import Path

from claim_review.constants import OUTPUT_COLUMNS
from claim_review.models import ClaimContext, ReviewResult
from claim_review.pipeline import write_output


class ReviewResultTests(unittest.TestCase):
    def setUp(self) -> None:
        """Create a mock ClaimContext used for multiple test scenarios."""
        self.context = ClaimContext(
            source_row={
                "user_id": "user_test",
                "image_paths": "images/test/case/img_1.jpg",
                "user_claim": "The screen is cracked.",
                "claim_object": "laptop",
            },
            image_files=(Path("img_1.jpg"),),
            image_ids=("img_1",),
            # Mocking history to check if the model correctly respects/merges these flags
            user_history={"history_flags": "user_history_risk"},
            evidence_requirements=(),
        )

    def test_normalizes_and_merges_history_flags(self) -> None:
        """Verify that user history risks are correctly injected into the final flags."""
        result = ReviewResult.from_model_output(
            {
                "evidence_standard_met": True,
                "evidence_standard_met_reason": "Screen is visible.",
                "risk_flags": [],
                "issue_type": "crack",
                "object_part": "screen",
                "claim_status": "supported",
                "claim_status_justification": "img_1 shows a crack.",
                "supporting_image_ids": ["img_1", "not_supplied"], # Testing filtering of fake IDs
                "valid_image": True,
                "severity": "medium",
            },
            self.context,
        )
        # Check that the test history flag was merged and 'not_supplied' was stripped
        self.assertEqual(result.risk_flags, ["user_history_risk"])
        self.assertEqual(result.supporting_image_ids, ["img_1"])

    def test_unusable_images_force_manual_review(self) -> None:
        """Ensure that invalid images trigger an automatic downgrade to 'not_enough_information'."""
        result = ReviewResult.from_model_output(
            {
                "evidence_standard_met": True,
                "evidence_standard_met_reason": "Incorrect model value.",
                "risk_flags": ["blurry_image"],
                "issue_type": "crack",
                "object_part": "screen",
                "claim_status": "supported",
                "claim_status_justification": "Incorrect model value.",
                "supporting_image_ids": ["img_1"],
                "valid_image": False, # Explicitly test the False case
                "severity": "high",
            },
            self.context,
        )
        # Validate business logic overrides: invalid images must force manual review
        self.assertTrue(result.evidence_standard_met)
        self.assertEqual(result.claim_status, "not_enough_information")
        self.assertEqual(result.supporting_image_ids, [])
        self.assertIn("manual_review_required", result.risk_flags)

    def test_csv_writer_preserves_exact_column_order(self) -> None:
        """Validate that the CSV output file perfectly matches the required schema headers."""
        output = Path(__file__).resolve().parent / "_test_output.csv"
        try:
            # Generate a temporary test CSV
            write_output(output, [{column: "value" for column in OUTPUT_COLUMNS}])
            # Read back only the header row
            header = output.read_text(encoding="utf-8").splitlines()[0]
            # Ensure the order is exactly as defined in constants.py
            self.assertEqual(header.split(","), OUTPUT_COLUMNS)
        finally:
            # Clean up the test file after execution
            output.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()