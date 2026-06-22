from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
import os
import random
import time
import urllib.error
import urllib.request
from pathlib import Path
from threading import Lock
# Reads/writes files (os, Path)
# Sends API requests (urllib.request)
# Processes JSON responses (json)
# Uploads images/files (base64, mimetypes)
# Creates hashes/checksums (hashlib)
# Handles retries and delays (time, random)
# Supports concurrent execution safely (Lock)

from .constants import PROMPT_VERSION
from .models import ClaimContext
from .prompting import SYSTEM_PROMPT, build_context_text, result_schema

class VisionClient:
    def __init__(
        self,
        model: str,
        cache_dir: Path,
        image_detail: str = "high",
        timeout_seconds: int = 120,
        max_retries: int = 4,
        minimum_interval_seconds: float = 0.0,
    ) -> None:
        # Securely retrieve the API key from environment variables
        self.api_key = os.environ.get("OPENAI_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured.")
        
        self.model = model
        self.cache_dir = cache_dir
        self.image_detail = image_detail
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.minimum_interval_seconds = minimum_interval_seconds
        
        # Threading lock ensures we don't violate rate limits when running concurrent workers
        self._rate_lock = Lock()
        self._last_request_at = 0.0
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    #This review() function is the main workflow of the VisionClient. It follows a very common pattern
    #Check Cache
    #     ↓
    #If found → Return cached result
    #     ↓
    #Else
    #     ↓
    #Build API request
    #     ↓
    #Call Vision Model
    #     ↓
    #Parse Response
    #     ↓
    #Save to Cache
    #     ↓
    #Return Result
    def review(self, context: ClaimContext, log_prefix: str = "") -> tuple[dict, dict]:
        # 1. Attempt to load from disk cache first to save time and API credits
        cache_key = self._cache_key(context)
        cache_path = self.cache_dir / f"{cache_key}.json"
        if cache_path.is_file():
            print(f"  [Cache Hit] {log_prefix}Using cached result for {context.claim_object}", flush=True)
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            return cached["result"], cached.get("usage", {})

        # 2. If not cached, prepare and send the request
        print(f"  [API Call] {log_prefix}Requesting vision review for {context.claim_object}...", flush=True)
        payload = self._build_payload(context)
        response = self._request_with_retries(payload, log_prefix)
        
        # 3. Extract, parse, and persist the successful result to disk
        result = json.loads(_extract_output_text(response))
        cache_path.write_text(
            json.dumps({"result": result, "usage": response.get("usage", {})}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return result, response.get("usage", {})
    


    #This function creates the request payload that will be sent to the vision model API.
    #Think of it as:
    #Claim Context
    #     ↓
    #Convert text + images
    #     ↓
    #Create API JSON
    #     ↓
    #Send to OpenAI
    def _build_payload(self, context: ClaimContext) -> dict:
        # Package the claim text and all referenced images into a base64 encoded structure
        content: list[dict] = [{"type": "input_text", "text": build_context_text(context)}]
        for image_path in context.image_files:
            media_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
            encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
            content.append({
                "type": "input_image",
                "image_url": f"data:{media_type};base64,{encoded}",
                "detail": self.image_detail, # Controls token cost vs. resolution (low/high)
            })

        # Return a JSON schema payload to force the LLM to output predictable, structured data
        return {
            "model": self.model,
            "instructions": SYSTEM_PROMPT,
            "input": [{"role": "user", "content": content}],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "damage_claim_review",
                    "strict": True,
                    "schema": result_schema(context.claim_object),
                }
            },
            "max_output_tokens": 700,
            "store": False,
        }
    


    #This function is the actual API caller. Its job is:
    #Build HTTP Request
    #       ↓
    #Send to OpenAI
    #       ↓
    #Success ? Return Response
    #       ↓
    #Failure ? Retry
    #       ↓
    #Give up after max_retries
    def _request_with_retries(self, payload: dict, log_prefix: str = "") -> dict:
        # Standard urllib request; body is the JSON payload of the claim
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            "https://api.openai.com/v1/responses",
            data=body,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )

        # Retry loop for transient network errors or temporary rate limits
        for attempt in range(self.max_retries + 1):
            try:
                self._wait_for_rate_slot() # Ensure we don't spam the API faster than allowed
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    return json.loads(response.read().decode("utf-8"))
            
            except urllib.error.HTTPError as error:
                # Handle OpenAI-specific HTTP error codes
                # 408	Request Timeout
                # 409	Conflict
                # 429	Rate Limit
                # 500+	Server Problems
                retryable = error.code in (408, 409, 429) or error.code >= 500
                if not retryable or attempt >= self.max_retries:
                    raise RuntimeError(f"OpenAI API returned HTTP {error.code}") from error
                
                # Special handling for 429 (Rate Limit): sleep 62s to allow bucket reset
                delay = 62.0 if error.code == 429 else min(30.0, (2**attempt) + random.random())
                print(f"  [API Retry] {log_prefix}Status {error.code}. Sleeping {delay}s...", flush=True)
                time.sleep(delay)
                
            except (urllib.error.URLError, TimeoutError) as error:
                # Handle connection-level failures
                if attempt >= self.max_retries:
                    raise RuntimeError(f"OpenAI API request failed: {error}") from error
                time.sleep(min(30.0, (2**attempt) + random.random()))

        raise AssertionError("retry loop exited unexpectedly")

    def _wait_for_rate_slot(self) -> None:
        # If a minimum interval is set, enforce a delay between requests to keep RPM within limits
        with self._rate_lock:
            elapsed = time.monotonic() - self._last_request_at
            if elapsed < self.minimum_interval_seconds:
                time.sleep(self.minimum_interval_seconds - elapsed)
            self._last_request_at = time.monotonic()

    def _cache_key(self, context: ClaimContext) -> str:
        # Generate a unique SHA-256 hash based on input data. 
        # If ANY data in the claim changes, the hash changes, triggering a re-run.
        digest = hashlib.sha256()
        digest.update(PROMPT_VERSION.encode()) # Updates if prompts change
        digest.update(self.model.encode())
        digest.update(self.image_detail.encode())
        digest.update(json.dumps(context.source_row, sort_keys=True).encode())
        digest.update(json.dumps(context.user_history, sort_keys=True).encode())
        for path in context.image_files:
            digest.update(path.read_bytes()) # Also includes the raw image binary data
        return digest.hexdigest()

def _extract_output_text(response: dict) -> str:
    # Dig through the OpenAI response structure to find the specific "output_text" node
    for output_item in response.get("output", []):
        if output_item.get("type") != "message": continue
        for content_item in output_item.get("content", []):
            if content_item.get("type") == "output_text":
                return content_item["text"]
            if content_item.get("type") == "refusal":
                raise RuntimeError(f"Model refused the review: {content_item}")
    raise RuntimeError("Responses API payload did not contain output_text")