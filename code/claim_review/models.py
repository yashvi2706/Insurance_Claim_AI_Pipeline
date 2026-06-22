from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .constants import (
    CLAIM_STATUSES,
    ISSUE_TYPES,
    OBJECT_PARTS,
    RISK_FLAGS,
    SEVERITIES,
)

@dataclass(frozen=True)
class ClaimContext:
    """Immutable container holding all verified inputs for a single claim run."""
    source_row: dict[str, str]
    image_files: tuple[Path, ...]
    image_ids: tuple[str, ...]
    user_history: dict[str, str]
    evidence_requirements: tuple[dict[str, str], ...]

    @property   #Turns a method into an attribute.
    def claim_object(self) -> str:
        return self.source_row["claim_object"]


#   output side of your pipeline.
#   Think of the flow as:
#   
#   CSV + Images
#         ↓
#   ClaimContext
#         ↓
#   OpenAI Vision Model
#         ↓
#   Raw JSON Output
#         ↓
#   ReviewResult.from_model_output()
#         ↓
#   Clean/Safe ReviewResult
#         ↓
#   CSV Row
# ReviewResult is a sanitization firewall between the model and your output file.
@dataclass
class ReviewResult:
    """Container for the sanitized output from the Vision API."""
    evidence_standard_met: bool
    evidence_standard_met_reason: str
    risk_flags: list[str]
    issue_type: str
    object_part: str
    claim_status: str
    claim_status_justification: str
    supporting_image_ids: list[str]
    valid_image: bool
    severity: str
    # from_model_output() cleans everything and enforces business rules before anything reaches output.csv
    @classmethod
    def from_model_output(
        cls,
        raw: dict[str, Any],
        context: ClaimContext,
    ) -> "ReviewResult":
        """ Takes raw JSON dict from OpenAI, strips out hallucinations, and forces the data to comply with our predefined constants. """
        # 1. Sanitize Risk Flags: Keep only valid flags defined in constants
        raw_risk_flags = _ensure_list(raw.get("risk_flags", []))
        risk_flags = _ordered_unique(
            flag for flag in raw_risk_flags if flag in RISK_FLAGS
        )

        # Inject historical user risks directly, overriding the model if necessary
        # Historical risk data > Model output
        history_flags = context.user_history.get("history_flags", "none").split(";")
        for flag in history_flags:
            if flag in ("user_history_risk", "manual_review_required"):
                risk_flags.append(flag)
        risk_flags = _ordered_unique(risk_flags)

        # 2. Strict Enums: Force issue_type, object_part, and status into approved vocab
        issue_type = _choice(raw.get("issue_type"), ISSUE_TYPES, "unknown")
        object_part = _choice(
            raw.get("object_part"),
            OBJECT_PARTS.get(context.claim_object, ()),
            "unknown",
        )
        claim_status = _choice(
            raw.get("claim_status"),
            CLAIM_STATUSES,
            "not_enough_information",
        )
        severity = _choice(raw.get("severity"), SEVERITIES, "unknown")
        
        # 3. Boolean Casts
        evidence_standard_met = bool(raw.get("evidence_standard_met"))
        valid_image = bool(raw.get("valid_image"))

        # 4. Filter Supporting Images: Prevent model from making up fake image IDs
        valid_ids = set(context.image_ids)
        raw_supporting_ids = _ensure_list(raw.get("supporting_image_ids", []))
        supporting_ids = _ordered_unique(
            image_id
            for image_id in raw_supporting_ids
            if image_id in valid_ids
        )

        # --- HARD BUSINESS LOGIC OVERRIDES ---
        
        # If the image is garbage, the claim cannot be mathematically "supported"
        if not valid_image and claim_status == "supported":
            claim_status = "not_enough_information"
            supporting_ids = []
            severity = "unknown"
            if "manual_review_required" not in risk_flags:
                risk_flags.append("manual_review_required")

        # If evidence rules failed, the claim cannot be supported
        if not evidence_standard_met and claim_status == "supported":
            claim_status = "not_enough_information"
            severity = "unknown"

        # Cleanup edge cases for consistency
        if claim_status == "not_enough_information" and not supporting_ids:
            severity = "unknown"
        if issue_type == "none":
            severity = "none"


        # _short_text()
        # Used for explanations.
        # evidence_standard_met_reason=_short_text(...)
        # Likely does: - convert to string
        #              - trim length
        #              - remove weird values
        return cls(
            evidence_standard_met=evidence_standard_met,
            evidence_standard_met_reason=_short_text(
                raw.get("evidence_standard_met_reason"),
                "The submitted images are not sufficient for automated review.",
            ),
            risk_flags=_ordered_unique(risk_flags),
            issue_type=issue_type,
            object_part=object_part,
            claim_status=claim_status,
            claim_status_justification=_short_text(
                raw.get("claim_status_justification"),
                "The available image evidence does not establish the claim.",
            ),
            supporting_image_ids=supporting_ids,
            valid_image=valid_image,
            severity=severity,
        )

    @classmethod
    def error_fallback(cls, message: str) -> "ReviewResult":
        """ The safety net we used for Claim 44. Generates a valid 'failed' row if the API crashes, preventing the pipeline from halting."""
        return cls(
            evidence_standard_met=False,
            evidence_standard_met_reason="Automated image review could not be completed.",
            risk_flags=["manual_review_required"],
            issue_type="unknown",
            object_part="unknown",
            claim_status="not_enough_information",
            claim_status_justification=f"Manual review required: {message[:180]}",
            supporting_image_ids=[],
            valid_image=False,
            severity="unknown",
        )

    def as_csv_fields(self) -> dict[str, str]:
        """Converts the Python object back into strings for the final CSV write."""
        return {
            "evidence_standard_met": str(self.evidence_standard_met).lower(),
            "evidence_standard_met_reason": self.evidence_standard_met_reason,
            "risk_flags": ";".join(self.risk_flags) if self.risk_flags else "none",
            "issue_type": self.issue_type,
            "object_part": self.object_part,
            "claim_status": self.claim_status,
            "claim_status_justification": self.claim_status_justification,
            "supporting_image_ids": (
                ";".join(self.supporting_image_ids)
                if self.supporting_image_ids
                else "none"
            ),
            "valid_image": str(self.valid_image).lower(),
            "severity": self.severity,
        }

# --- Utility Helpers for Data Sanitization ---

def _choice(value: Any, choices: tuple[str, ...], default: str) -> str:
    """Returns the value if it's in the allowed list; otherwise returns default."""
    return value if isinstance(value, str) and value in choices else default

def _short_text(value: Any, default: str) -> str:
    """Strips whitespace and hard-caps text at 500 chars to prevent CSV bloat."""
    if not isinstance(value, str) or not value.strip():
        return default
    return " ".join(value.split())[:500]

def _ordered_unique(values: Any) -> list[str]:
    """Removes duplicates from a list while preserving original order."""
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result

def _ensure_list(value: Any) -> list[str]:
    """Safely converts unknown JSON structures into a flat list of strings."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(v) for v in value]
    return []