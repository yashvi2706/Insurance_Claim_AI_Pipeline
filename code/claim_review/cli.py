"""Argument parsing for the public command-line entry point."""

from __future__ import annotations

import argparse #argparse → command-line arguments
import os #os → operating system operations
from pathlib import Path #Path → file/folder path handling (modern alternative to many os.path functions)

from .data import load_contexts
from .pipeline import ReviewPipeline


# Dynamically resolve the absolute paths so the script works from any directory
CODE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_DIR.parent
# __file__ is the path of the current Python file.
# .resolve() converts it into an absolute path:
# The parents list contains all parent directories:
# Path("/home/yashvi/project/src/utils/helper.py").parents
# [0] /home/yashvi/project/src/utils
# [1] /home/yashvi/project/src      this is CODE_DIR
# [2] /home/yashvi/project          this is REPO_ROOT
# [3] /home/yashvi
# ...



# this whole function is basically building the interface through which a user can control the script from the terminal.
# A type hint is extra information you add to Python code to indicate what type of data a variable, parameter, or return value is expected to have.
# -> argparse.ArgumentParser is a type hint saying this function returns an ArgumentParser object.

def build_parser() -> argparse.ArgumentParser:
    # Initialize the argument parser with a description of the tool
    parser = argparse.ArgumentParser(
        description="Verify damage claims using conversations, images, and history."
    )
    
    # --- File Path Arguments ---
    # Define paths for the dataset, histories, requirements, and outputs.
    # These default to the standard repository structure but can be overridden.
    parser.add_argument(
        "--claims",
        type=Path,
        default=REPO_ROOT / "dataset" / "claims.csv",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=REPO_ROOT / "dataset",
    )
    parser.add_argument(
        "--history",
        type=Path,
        default=REPO_ROOT / "dataset" / "user_history.csv",
    )
    parser.add_argument(
        "--requirements",
        type=Path,
        default=REPO_ROOT / "dataset" / "evidence_requirements.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "output.csv",
    )
    
    # --- API and Execution Arguments ---
    # Pull the OpenAI model from environment variables, defaulting to gpt-5.4-mini
    # if someone does export OPENAI_MODEL=gpt-5 them args.model=gpt-5 else its the default one
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENAI_MODEL", "gpt-5.4-mini"),
    )
    # Define the number of concurrent threads to use (default is 2)
    # for parallel processing
    parser.add_argument("--workers", type=int, default=2)
    
    # Control how detailed the images sent to OpenAI should be
    parser.add_argument(
        "--image-detail",
        choices=("low", "high", "original", "auto"),
        default="high",
    )
    
    # Set a rate limit throttle, pulling from environment variables if available
    parser.add_argument(
        "--requests-per-minute",
        type=int,
        default=int(os.environ.get("OPENAI_RPM", "0")),
        help="Optional client-side throttle; 0 disables it.",
    )
    
    # Define where the cache JSON files and run metadata will be saved locally
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=CODE_DIR / ".cache" / "reviews",
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=CODE_DIR / "evaluation" / "last_run.json",
    )
    
    # --- Safety and Validation Flags ---
    # When passed in the terminal, this sets continue_on_error to True (catches crashes)
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Emit conservative manual-review rows instead of stopping.",
    )
    
    # When passed, this stops the script from calling the API and only checks local files
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate CSVs and referenced images without making model calls.",
    )
    return parser


# If the command is: python3 main.py \
#                           --workers 4 \
#                           --model gpt-5 \
#                           --validate-only
# then:
#     args.workers        # 4
#     args.model          # "gpt-5"
#     args.validate_only # True
#     args.output         # default output.csv



# main() is essentially the controller that decides whether to just validate the dataset or run the full claim-review pipeline and generate predictions.

def main(argv: list[str] | None = None) -> int:
    # Parse the arguments provided in the terminal command
    args = build_parser().parse_args(argv)
    
    # If the user only wants to validate files, do that and exit immediately
    if args.validate_only:
        contexts = load_contexts(
            args.claims,
            args.dataset_dir,
            args.history,
            args.requirements,
        )
        print(
            f"Validated {len(contexts)} rows and "
            f"{sum(len(item.image_files) for item in contexts)} images."
        )
        return 0  # 0 indicates a successful exit without errors

    # Initialize the main ReviewPipeline using the parsed arguments
    pipeline = ReviewPipeline(
        dataset_dir=args.dataset_dir,
        model=args.model,
        cache_dir=args.cache_dir,
        workers=args.workers,
        image_detail=args.image_detail,
        requests_per_minute=args.requests_per_minute,
        continue_on_error=args.continue_on_error,
    )
    
    # Execute the pipeline to process claims and write the output CSV
    pipeline.run(
        claims_path=args.claims,
        history_path=args.history,
        requirements_path=args.requirements,
        output_path=args.output,
        run_metadata_path=args.metadata_output,
    )
    
    # Notify the user where the final results were safely saved
    print(f"Wrote predictions to {args.output.resolve()}")
    return 0