# The exact column headers required for the final final_predictions.csv output file.
# These must match the expected grading schema perfectly.
OUTPUT_COLUMNS = [
    "user_id",
    "image_paths",
    "user_claim",
    "claim_object",
    "evidence_standard_met",
    "evidence_standard_met_reason",
    "risk_flags",
    "issue_type",
    "object_part",
    "claim_status",
    "claim_status_justification",
    "supporting_image_ids",
    "valid_image",
    "severity",
]

# The three acceptable final decisions for a claim after evaluation.
CLAIM_STATUSES = ("supported", "contradicted", "not_enough_information")

# Standardized vocabulary for classifying the specific type of damage detected.
ISSUE_TYPES = (
    "dent",
    "scratch",
    "crack",
    "glass_shatter",
    "broken_part",
    "missing_part",
    "torn_packaging",
    "crushed_packaging",
    "water_damage",
    "stain",
    "none",
    "unknown",
)

# Hierarchical mapping of supported claim objects to their valid physical parts.
# This acts as a strict guardrail so the vision model doesn't hallucinate invalid parts (e.g., a car's "keyboard").
OBJECT_PARTS = {
    "car": (
        "front_bumper",
        "rear_bumper",
        "door",
        "hood",
        "windshield",
        "side_mirror",
        "headlight",
        "taillight",
        "fender",
        "quarter_panel",
        "body",
        "unknown",
    ),
    "laptop": (
        "screen",
        "keyboard",
        "trackpad",
        "hinge",
        "lid",
        "corner",
        "port",
        "base",
        "body",
        "unknown",
    ),
    "package": (
        "box",
        "package_corner",
        "package_side",
        "seal",
        "label",
        "contents",
        "item",
        "unknown",
    ),
}

# Tags used to flag suspicious submissions, poor image quality, or prompt-injection attempts (like sticky notes with instructions).
RISK_FLAGS = (
    "none",
    "blurry_image",
    "cropped_or_obstructed",
    "low_light_or_glare",
    "wrong_angle",
    "wrong_object",
    "wrong_object_part",
    "damage_not_visible",
    "claim_mismatch",
    "possible_manipulation",
    "non_original_image",
    "text_instruction_present",
    "user_history_risk",
    "manual_review_required",
)

# Standardized scale for grading the extent of the actual physical damage.
SEVERITIES = ("none", "low", "medium", "high", "unknown")

# Cache invalidation key. If you update your prompts or logic, changing this date 
# forces the script to ignore the .cache/ folder and re-evaluate everything from scratch.
PROMPT_VERSION = "2026-06-19.v1"