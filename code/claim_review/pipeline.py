from __future__ import annotations

import csv # Used for reading and writing CSV files.
import json # Used for: API payloads, API responses, Cache files
import time # Used in retry logic.
from concurrent.futures import ThreadPoolExecutor, as_completed # Used to run tasks in parallel threads.
                                                                # Lets you process tasks as soon as they finish.
from pathlib import Path # Modern file handling.

from .constants import OUTPUT_COLUMNS
from .data import load_contexts
from .models import ClaimContext, ReviewResult
from .openai_vision import VisionClient

class ReviewPipeline:
    def __init__(
        self,
        dataset_dir: Path,
        model: str,
        cache_dir: Path,
        workers: int = 2,
        image_detail: str = "high",
        requests_per_minute: int = 0,
        continue_on_error: bool = True,
    ) -> None:
        # Calculate interval between requests if a rate limit is enforced (RPM -> seconds)
        interval = 60.0 / requests_per_minute if requests_per_minute else 0.0
        self.dataset_dir = dataset_dir.resolve()
        self.workers = max(1, workers)
        # Force continue_on_error to True to ensure corrupted images don't crash the whole run
        self.continue_on_error = True 
        self.client = VisionClient(
            model=model,
            cache_dir=cache_dir,
            image_detail=image_detail,
            minimum_interval_seconds=interval,
        )

    #Load Claims
    #     ↓
    #Create Context Objects
    #     ↓
    #Launch Worker Threads
    #     ↓
    #Review Claims Concurrently
    #     ↓
    #Collect Results
    #     ↓
    #Aggregate Token Usage
    #     ↓
    #Generate CSV Output
    #     ↓
    #Write Metadata
    #     ↓
    #Return Results
    def run(
        self,
        claims_path: Path,
        history_path: Path,
        requirements_path: Path,
        output_path: Path,
        run_metadata_path: Path | None = None,
    ) -> list[dict[str, str]]:
        # Load and validate all data (claims, user history, and validation rules)
        print(f"Loading dataset from {claims_path}...", flush=True)
        contexts = load_contexts(
            claims_path=claims_path,
            dataset_dir=self.dataset_dir,
            history_path=history_path,
            requirements_path=requirements_path,
        )
        started = time.monotonic()
        # Initialize a list to hold results in the exact same order as the input claims
        reviewed: list[tuple[ReviewResult, dict] | None] = [None] * len(contexts)

        print(f"Dispatching {len(contexts)} claims to {self.workers} concurrent workers...", flush=True)

        # Use ThreadPoolExecutor for concurrent API calls (essential for speed)
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = {} # store Future Object → Original Index mapping
            for index, context in enumerate(contexts):
                log_prefix = f"[Claim {index + 1}/{len(contexts)} | {context.source_row.get('user_id', 'unknown')}] "
                # Submit tasks to the executor
                futures[executor.submit(self._review_one, context, log_prefix)] = index
            
            # as_completed handles results as soon as they arrive, regardless of order
            for future in as_completed(futures):
                index = futures[future]
                reviewed[index] = future.result()
                print(f"Finished evaluating claim {index + 1}/{len(contexts)}", flush=True)

        # Aggregate results and calculate total token usage
        output_rows: list[dict[str, str]] = []
        usage_totals: dict[str, int] = {}
        for context, item in zip(contexts, reviewed, strict=True):
            assert item is not None
            result, usage = item
            for key, value in usage.items():
                if isinstance(value, int):
                    usage_totals[key] = usage_totals.get(key, 0) + value
            # Merge original source data with our processed AI results
            output_rows.append({**context.source_row, **result.as_csv_fields()})

        # Save the final CSV and run metadata
        write_output(output_path, output_rows)
        if run_metadata_path:
            run_metadata_path.parent.mkdir(parents=True, exist_ok=True)
            run_metadata_path.write_text(
                json.dumps({
                    "rows": len(contexts),
                    "images": sum(len(item.image_files) for item in contexts),
                    "model_calls": len(contexts),
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                    "usage": usage_totals,
                }, indent=2),
                encoding="utf-8",
            )
        return output_rows

    def _review_one(self, context: ClaimContext, log_prefix: str = "") -> tuple[ReviewResult, dict]:
        # Single worker task: attempt an API review, catch errors if they occur
        try:
            raw_result, usage = self.client.review(context, log_prefix)
            return ReviewResult.from_model_output(raw_result, context), usage
        except Exception as error:
            # If error recovery is off, let the app crash; otherwise, provide a fallback result
            if not self.continue_on_error:
                raise
            # Return an empty usage dict and a "manual_review_required" ReviewResult
            return ReviewResult.error_fallback(str(error)), {}

def write_output(path: Path, rows: list[dict[str, str]]) -> None:
    # Safely write the final list of dictionaries to a CSV file
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)