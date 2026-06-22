# Evaluation Report

## Dataset and approach

The labeled sample contains 20 claims and 30 images. The final test set contains
44 claims and 82 images. The production strategy makes one multimodal call per
claim, includes all images for that claim in the same request, uses strict
structured output, then applies deterministic validation and history-risk
enrichment.

## Strategies compared

Two fully offline text baselines are included so evaluation remains reproducible
without an API key:

1. **Conversation-only baseline** — extracts issue and part keywords, assumes
   that evidence is valid, and predicts `supported`.
2. **Conversation + history baseline** — uses the same extraction but also
   carries risk flags from `user_history.csv`.

The selected production strategy is **vision + conversation + evidence rules +
history**, using `gpt-5.4-mini` with `detail=high`. It directly inspects images,
handles contradictions and insufficient evidence, detects wrong objects or
parts, and treats visible instruction text as untrusted.

Run:

```bash
python3 code/evaluation/main.py
```

The exact baseline metrics are written to `code/evaluation/metrics.json`.
An API-backed sample benchmark can be added with:

```bash
python3 code/main.py \
  --claims dataset/sample_claims.csv \
  --output code/evaluation/sample_predictions.csv
python3 code/evaluation/main.py \
  --predictions code/evaluation/sample_predictions.csv
```

No API key was available in the build environment, so an API-backed sample
metric is not claimed in this report. This limitation is stated explicitly
rather than presenting manually reviewed labels as model performance.

## Operational analysis

### Calls and images

| Split | Claims / model calls | Images |
|---|---:|---:|
| Sample | 20 | 30 |
| Test | 44 | 82 |
| Combined development run | 64 | 112 |

The implementation uses one call per claim rather than one call per image. This
lets the model reason across wide shots and close-ups while reducing repeated
prompt tokens.

### Token and cost assumptions

Approximate usage depends heavily on image dimensions. For planning, assume
roughly 2,000 image/input tokens per image at high detail, 1,000 text/input
tokens per claim, and 250 output tokens per claim.

Estimated test usage:

- Input: `82 × 2,000 + 44 × 1,000 ≈ 208,000` tokens
- Output: `44 × 250 ≈ 11,000` tokens

Pricing assumption (OpenAI standard pricing checked June 19, 2026):
`gpt-5.4-mini` at $0.75 per 1M input tokens and $4.50 per 1M output tokens.

- Input cost: about `$0.156`
- Output cost: about `$0.050`
- Estimated full test cost: about **$0.21**, excluding retry overhead

Actual token usage is recorded from API responses in
`code/evaluation/last_run.json`, allowing this estimate to be replaced with
measured cost after execution.

### Latency, TPM, and RPM

At an expected 4–10 seconds per claim, sequential test runtime is roughly
3–8 minutes. The default two-worker configuration should usually complete in
about 2–4 minutes, subject to account rate limits and image upload speed.

Rate-limit controls:

- `--requests-per-minute` / `OPENAI_RPM` enables a shared client-side throttle.
- HTTP 408, 409, 429, and 5xx responses use capped exponential backoff with
  jitter.
- A content-addressed disk cache prevents repeated calls for unchanged claims.
- `--workers` controls concurrency.
- `--continue-on-error` can emit conservative manual-review rows for operational
  runs; strict mode remains the default for final generation.

For low TPM tiers, reduce workers to 1 and set RPM explicitly. For larger
production datasets, use the Batch API or partition work by token budget rather
than submitting unbounded concurrent requests.
