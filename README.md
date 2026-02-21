# VitEmoNet

EEG emotion recognition (SEED) with a Transformer-based pipeline.

## What likely caused accuracy ~ random

After auditing the code, there were two high-impact issues:

1. **Input shape mismatch in preprocessing**
   - Old code added an extra channel axis in `preprocess()` (`frames[..., tf.newaxis]`).
   - But the transformer model expects `(5, 25, 62)`, not `(5, 25, 62, 1)`.
   - This can break feature semantics and destabilize learning.

2. **Label indexing mismatch risk**
   - Data scripts generate labels with `+1` (`{1,2,3}`), while `SparseCategoricalCrossentropy` for 3 classes expects `{0,1,2}`.
   - This can severely hurt training/evaluation validity.

## Fixes applied

- Removed extra axis in `preprocess()`; keep float32 tensor shape compatible with transformer input.
- Added robust label normalization in training:
  - auto-convert `{1,2,3}` -> `{0,1,2}`
  - validate unexpected ranges with explicit error.
- Fixed class-weight calculation to match normalized class IDs.
- Enabled deterministic seeding.
- Added random-baseline logging for sanity check.

## How to run

Place processed files in one of:
- `./input_data_1d/` (default):
  - `train_data.npy`, `train_label.npy`
  - `val_data.npy`, `val_label.npy`
  - `test_data.npy`, `test_label.npy`

Then run:

```bash
python3 train.py
```

Logs are written to `save/<timestamp>/train.log`.

## Before vs After metrics

In this environment, dataset `.npy` files were not present, so full numeric A/B on your real split could not be executed here.

Once data is available, compare:
- `random baseline accuracy` in log
- `final test accuracy` in log

Expected healthy behavior: test accuracy should be **consistently above random (~33% for 3 classes)**.

## Notes

- This patch focuses on training correctness first.
- Next step if needed: subject-wise CV + per-subject metrics + confusion matrix export.
