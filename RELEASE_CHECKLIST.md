# Release Checklist

Use this before creating a Git tag or GitHub release.

## Release Gate

1. Run the automated test suite:

```bash
.venv/bin/python -m pytest tests -v
```

2. Verify the small Apple Silicon path still works inside a clean temporary `NANOCHAT_BASE_DIR`:
- download at least 2 shards
- train the tokenizer
- run a tiny base train
- run a tiny SFT pass
- confirm CLI chat works
- confirm chat eval works

3. Verify the quickstart flow:
- empty-state errors are short and actionable
- `/status` reflects real progress instead of guessing
- model load, chat, and unload work after training or import

4. Verify a real Hugging Face import:

```bash
python -m scripts.convert_from_hf --repo nanochat-students/base-d20 --memory-limit-gb=8
```

The release is not ready if any of the following happen:
- tokenizer training succeeds with fewer than 2 shards
- train, SFT, chat, or eval fail with raw setup tracebacks instead of plain guidance
- quickstart marks skipped work as complete
- converted Hugging Face checkpoints fail verification after import

## Notes For This Repo

- The practical release target is an Apple Silicon laptop with 24 GB of shared memory.
- The routine smoke path should stay at depth 4 with low memory caps.
- This repo tracks `karpathy/nanochat` behavior more closely than `karpathy/autoresearch`, so default training and fine-tuning settings should be compared against `nanochat` first.

## Last Verified

Verified on March 24, 2026:

- `29 passed, 1 skipped` on `.venv/bin/python -m pytest tests -v`
- real Hugging Face import succeeded for `nanochat-students/base-d20`
- tiny local base-train, SFT, chat, eval, and quickstart smoke tests all passed within the small-machine target
