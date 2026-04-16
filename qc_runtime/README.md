# QuantConnect Runtime Folder

This `qc_runtime/` folder is the **only folder you need to use for QuantConnect backtests**.

## Entrypoint
- `main.py` (class `SimplifiedCryptoStrategy`)

## What is included here
- The latest runnable/backtestable QC algorithm entrypoint (`main.py`)
- All Python modules that `main.py` needs directly or indirectly for runtime behavior
- A minimal `nextgen` bridge subset used by the current algorithm's optional risk gate:
  - `nextgen/core/types.py`
  - `nextgen/risk/engine.py`

## What is NOT required for QC runtime
You do **not** need to upload/use these for normal backtests from this runtime folder:
- `tests/`
- `NEXTGEN_ARCHITECTURE.md`
- the rest of the full root `nextgen/` scaffold (research/portfolio/signals/etc.)
- other repo/dev files outside `qc_runtime/`

## Is `nextgen/` required?
- The full root `nextgen/` scaffold is **not required**.
- Only the small bridge included inside `qc_runtime/nextgen/` is used by this algorithm path.

If you upload/sync this folder to QC, keep these files together and use `main.py` as the algorithm file.
