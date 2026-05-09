# exp_0280 — CLAP cache build

Date: 2026-04-13

## Goal

Build a `row_id`-aligned CLAP embedding cache for the same trusted soundscape rows used by `exp_027a`, so that `exp_028a_clap_perch_complementarity_benchmark` can run directly.

## Notebook

- [exp_0280_clap_cache_build.ipynb](/Users/yaroslav/Documents/Kaggle/BirdCLEF_2026/notebooks/exp_0280_clap_cache_build.ipynb)

## Output Contract

Writes into:

- `experiments/outputs/exp_0280_clap_cache_build/clap_meta.parquet`
- `experiments/outputs/exp_0280_clap_cache_build/clap_arrays.npz`
- `experiments/outputs/exp_0280_clap_cache_build/full_clap_meta.parquet`
- `experiments/outputs/exp_0280_clap_cache_build/full_clap_arrays.npz`
- `experiments/outputs/exp_0280_clap_cache_build/setup_snapshot.json`
- `experiments/outputs/exp_0280_clap_cache_build/report_snapshot.json`

The saved arrays use key:

- `clap_emb_full`

This matches the cache contract already supported by `exp_028a`.

## Result

- Local run completed successfully
- `708` rows
- `59` files
- embedding dim `512`
- `row_id_match = True`
- runtime about `26.5s` on local `mps`

## Interpretation

This is the clean infrastructure outcome we needed for the CLAP branch.

- the cache aligns perfectly to the trusted `exp_027a` rows
- the downstream benchmark no longer needs any format conversion
- the next real question is now purely modeling: whether CLAP adds local complementarity over the fixed `exp_015d` teacher
