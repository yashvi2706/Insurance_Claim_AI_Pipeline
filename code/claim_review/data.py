"""CSV loading, path resolution, and input validation."""

from __future__ import annotations

import csv
from pathlib import Path

from .models import ClaimContext

# Define the exact headers expected in the claims.csv file
INPUT_COLUMNS = ("user_id", "image_paths", "user_claim", "claim_object")
# Strict whitelist of allowed claim types to prevent processing invalid data
VALID_OBJECTS = {"car", "laptop", "package"}


def read_csv(path: Path) -> list[dict[str, str]]:
    # Open the file using 'utf-8-sig' to automatically strip invisible Byte Order Marks (BOM) often added by Excel, which can corrupt column headers.
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        # csv.DictReader converts each row into a dictionary mapped to the column headers
        return list(csv.DictReader(file))
# Opening the File
# with path.open("r", encoding="utf-8-sig", newline="") as file:
# Equivalent to:
# file = open(path, "r", encoding="utf-8-sig")
# but with automatically closes the file afterwards.

# newline=""
# Recommended when using Python's csv module.
# It prevents issues such as extra blank lines on some operating systems.
# without list for loop req but with list everything at once is received




# This function is the data preparation layer of the whole pipeline.
# Its job is:
#   Read CSVs
#       ↓
#   Validate rows
#       ↓
#   Validate images
#       ↓
#   Attach user history
#       ↓
#   Attach evidence requirements
#       ↓
#   Create ClaimContext objects
#   
# and return them.
def load_contexts(
    claims_path: Path,
    dataset_dir: Path,
    history_path: Path,
    requirements_path: Path,
) -> list[ClaimContext]:
    """Load every claim and join history/rule context without mutating inputs."""

    # Load all three datasets into memory
    claims = read_csv(claims_path)
    # Use dictionary comprehension to map user_id directly to their history row for instant lookups
    histories = {row["user_id"]: row for row in read_csv(history_path)}
    requirements = read_csv(requirements_path)
    
    contexts: list[ClaimContext] = []

    # Start enumerating at 2 because row 1 in a CSV is the header (helps with accurate error logging)
    for row_number, row in enumerate(claims, start=2):
        
        # --- Validation Phase ---
        # Check if any required columns are missing from the current row
        missing = [column for column in INPUT_COLUMNS if column not in row]
        if missing:
            raise ValueError(
                f"{claims_path}: row {row_number} is missing columns: {missing}"
            )
        
        # Reject the row immediately if the object isn't a car, laptop, or package
        if row["claim_object"] not in VALID_OBJECTS:
            raise ValueError(
                f"{claims_path}: row {row_number} has invalid claim_object "
                f"{row['claim_object']!r}"
            )

        # --- Image Path Resolution ---
        # Split the semicolon-separated string into a list of individual image paths
        relative_paths = [part.strip() for part in row["image_paths"].split(";")]
        # Convert relative paths (e.g., 'images/test/img1.jpg') to absolute system paths
        image_files = tuple((dataset_dir / path).resolve() for path in relative_paths)
        
        # Ping the hard drive to verify every image file actually exists before proceeding
        missing_images = [str(path) for path in image_files if not path.is_file()]
        if missing_images:
            raise FileNotFoundError(
                f"{claims_path}: row {row_number} references missing images: "
                f"{missing_images}"
            )

        # --- Context Assembly ---
        # Filter the global requirements to only include universal rules ("all") and rules specific to this row's object (e.g., only "car" rules)
        object_requirements = tuple(
            rule
            for rule in requirements
            if rule["claim_object"] in ("all", row["claim_object"])
        )
        
        # Package the validated row, absolute image paths, user history, and specific rules into a single ClaimContext object to safely pass to the OpenAI API
        contexts.append(
            ClaimContext(
                source_row={column: row[column] for column in INPUT_COLUMNS},
                image_files=image_files,
                image_ids=tuple(path.stem for path in image_files), # Extract just the filename without extension
                user_history=histories.get(row["user_id"], {}),     # Fallback to empty dict if no history exists
                evidence_requirements=object_requirements,
            )
        )

    return contexts

# in Image IDs stem means filename without extension

# Rule of thumb
# Use: list
# when you expect:
# append()
# remove()
# sort()

# Use:  tuple
# when the collection represents:
# fixed data
# configuration
# context
# coordinates
# record fields