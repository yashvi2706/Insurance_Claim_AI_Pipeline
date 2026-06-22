from __future__ import annotations

import argparse
import csv
import json
import re # Used for regex matching  crack - cracked
import sys
from collections import Counter # Counts frequencies.
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from claim_review.constants import OUTPUT_COLUMNS

SCORED_FIELDS = [
    "evidence_standard_met",
    "risk_flags",
    "issue_type",
    "object_part",
    "claim_status",
    "supporting_image_ids",
    "valid_image",
    "severity",
]

ISSUE_PATTERNS = [
    ("glass_shatter", r"\bshatter(?:ed)?\b"),
    ("crushed_packaging", r"\bcrush(?:ed)?\b|\bdab gaya\b"),
    ("torn_packaging", r"\btorn\b|\btear\b|\bopen jaisa\b"),
    ("water_damage", r"\bwater\b|\bwet\b|\bliquid damage\b"),
    ("missing_part", r"\bmissing\b|\bfaltan\b"),
    ("broken_part", r"\bbroke(?:n)?\b|\btoot\b"),
    ("scratch", r"\bscratch(?:ed)?\b|\bscrape\b"),
    ("crack", r"\bcrack(?:ed)?\b"),
    ("dent", r"\bdent(?:ed|s)?\b"),
    ("stain", r"\bstain\b|\bmark\b"),
]
PART_PATTERNS = {
    "car": [
        ("rear_bumper", r"\brear bumper\b|\bback bumper\b|parachoques trasero"),
        ("front_bumper", r"\bfront bumper\b"),
        ("windshield", r"\bwindshield\b|\bfront glass\b"),
        ("side_mirror", r"\bside mirror\b|\bmirror\b"),
        ("headlight", r"\bheadlight\b"),
        ("taillight", r"\btaillight\b|\bback light\b"),
        ("quarter_panel", r"\bquarter panel\b"),
        ("fender", r"\bfender\b"),
        ("hood", r"\bhood\b"),
        ("door", r"\bdoor\b"),
        ("body", r"\bbody\b|\bpanel\b"),
    ],
    "laptop": [
        ("trackpad", r"\btrackpad\b"),
        ("keyboard", r"\bkeyboard\b|\bkeycaps?\b|\bkeys?\b|\bteclas\b"),
        ("hinge", r"\bhinge\b"),
        ("screen", r"\bscreen\b|\bpantalla\b"),
        ("lid", r"\blid\b"),
        ("corner", r"\bcorner\b"),
        ("port", r"\bport\b"),
        ("base", r"\bbase\b"),
        ("body", r"\bbody\b|\bouter\b"),
    ],
    "package": [
        ("package_corner", r"\bcorner\b"),
        ("package_side", r"\bside\b"),
        ("seal", r"\bseal\b|\btorn.open\b"),
        ("label", r"\blabel\b"),
        ("contents", r"\bcontents?\b|\binside\b|\bproduct\b"),
        ("item", r"\bitem\b"),
        ("box", r"\bbox\b|\bpackage\b|\bparcel\b"),
    ],
}

#Ground Truth CSV
#        │
#        ▼
#  Baseline Models
#        │
#        ▼
# Vision Pipeline CSV
#        │
#        ▼
#   Accuracy/F1
#        │
#        ▼
#  metrics.json

# Converts values into comparable form.
def _normalize_val(val: str) -> str:
    v = val.strip().lower()
    if v in ("true", "false"):
        return v
    return v

def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))

# Idea:Ignore images.Only read claim text.
def conversation_baseline(
    rows: list[dict[str, str]],
    history: dict[str, dict[str, str]],
    use_history: bool,
) -> list[dict[str, str]]:
    predictions = []
    for row in rows:
        text = row["user_claim"].lower()
        issue = next(
            (value for value, pattern in ISSUE_PATTERNS if re.search(pattern, text)),
            "unknown",
        )
        part = next(
            (
                value
                for value, pattern in PART_PATTERNS[row["claim_object"]]
                if re.search(pattern, text)
            ),
            "unknown",
        )
        risk_flags = "none"
        if use_history:
            risk_flags = history.get(row["user_id"], {}).get("history_flags", "none")

        predictions.append(
            {
                **{column: row[column] for column in OUTPUT_COLUMNS[:4]},
                "evidence_standard_met": "true",
                "evidence_standard_met_reason": (
                    "Text-only baseline assumes submitted evidence is sufficient."
                ),
                "risk_flags": risk_flags,
                "issue_type": issue,
                "object_part": part,
                "claim_status": "supported",
                "claim_status_justification": (
                    "Text-only baseline mirrors the conversation; images not reviewed."
                ),
                "supporting_image_ids": row["image_paths"]
                .split(";")[0]
                .rsplit("/", 1)[-1]
                .rsplit(".", 1)[0],
                "valid_image": "true",
                "severity": "medium",
            }
        )
    return predictions

def exact_accuracy(
    expected: list[dict[str, str]],
    predicted: list[dict[str, str]],
    field: str,
) -> float:
    return sum(
        _normalize_val(actual[field]) == _normalize_val(guess[field])
        for actual, guess in zip(expected, predicted, strict=True)
    ) / len(expected)

def macro_f1(
    expected: list[dict[str, str]],
    predicted: list[dict[str, str]],
    field: str,
) -> float:
    labels = sorted({_normalize_val(row[field]) for row in expected} | {_normalize_val(row[field]) for row in predicted})
    scores = []
    for label in labels:
        true_positive = sum(
            _normalize_val(actual[field]) == label and _normalize_val(guess[field]) == label
            for actual, guess in zip(expected, predicted, strict=True)
        )
        false_positive = sum(
            _normalize_val(actual[field]) != label and _normalize_val(guess[field]) == label
            for actual, guess in zip(expected, predicted, strict=True)
        )
        false_negative = sum(
            _normalize_val(actual[field]) == label and _normalize_val(guess[field]) != label
            for actual, guess in zip(expected, predicted, strict=True)
        )
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(2 * true_positive / denominator if denominator else 0.0)
    return sum(scores) / len(scores)

def score(
    expected: list[dict[str, str]],
    predicted: list[dict[str, str]],
) -> dict:
    if len(expected) != len(predicted):
        raise ValueError(
            f"Expected {len(expected)} rows but prediction file has {len(predicted)}"
        )
    field_accuracy = {
        field: round(exact_accuracy(expected, predicted, field), 4)
        for field in SCORED_FIELDS
    }
    return {
        "rows": len(expected),
        "field_accuracy": field_accuracy,
        "mean_field_accuracy": round(
            sum(field_accuracy.values()) / len(field_accuracy), 4
        ),
        "claim_status_macro_f1": round(
            macro_f1(expected, predicted, "claim_status"), 4
        ),
        "expected_status_distribution": dict(
            Counter(row["claim_status"] for row in expected)
        ),
        "predicted_status_distribution": dict(
            Counter(row["claim_status"] for row in predicted)
        ),
    }

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample",
        type=Path,
        default=REPO_ROOT / "dataset" / "sample_claims.csv",
    )
    parser.add_argument(
        "--history",
        type=Path,
        default=REPO_ROOT / "dataset" / "user_history.csv",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        help="Optional API-generated sample prediction CSV to score.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path(__file__).resolve().parent / "metrics.json",
    )
    args = parser.parse_args()

    sample = read_csv(args.sample)
    history = {row["user_id"]: row for row in read_csv(args.history)}
    strategies = {
        "conversation_only_baseline": score(
            sample,
            conversation_baseline(sample, history, use_history=False),
        ),
        "conversation_plus_history_baseline": score(
            sample,
            conversation_baseline(sample, history, use_history=True),
        ),
    }
    if args.predictions:
        strategies["vision_pipeline"] = score(sample, read_csv(args.predictions))

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(strategies, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(strategies, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())