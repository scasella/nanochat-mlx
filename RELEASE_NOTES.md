# Release Notes

## March 24, 2026

This release is focused on making `nanochat-mlx` safer to run, easier to recover from setup mistakes, and more dependable on a 24 GB Apple Silicon laptop.

### What improved

- Setup problems now fail with short, plain guidance instead of raw tracebacks.
- Tokenizer training now refuses to run unless there is enough downloaded data to create both training and validation splits.
- Training, fine-tuning, chat, and eval now check their prerequisites before starting.
- The quickstart web flow now blocks impossible steps, reports errors more clearly, and keeps its progress display in sync with the real backend state.
- Hugging Face import now has a friendlier failure path when optional conversion dependencies are missing, and `--help` works without the full conversion stack installed.
- The default training ratio was updated to match current upstream `nanochat`.
- Training now stops immediately if the loss becomes invalid or obviously blows up.

### Validation

- Automated test suite: `29 passed, 1 skipped`
- Real Hugging Face import verified with `nanochat-students/base-d20`
- Tiny end-to-end local runs verified for tokenizer, base training, fine-tuning, chat, eval, and quickstart

### Practical target

The release gate for this repo is a small Apple Silicon machine with 24 GB of shared memory. The routine smoke path should stay on the small depth-4 configuration with low memory caps.
