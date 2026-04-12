# exp_024_geo_regime_ablation

## Goal

Test whether the stable `exp_015d` V18 artifactized path still leaves useful `site/hour` signal on the table in the weakest geo regimes.

Notebook:

- `notebooks/exp_024_geo_regime_ablation.ipynb`

## Why This Exists

The project already uses geography-aware information heavily:

- `site/hour/site-hour` prior tables
- in-model `site/hour` embeddings inside `ProtoSSM`
- downstream fusion built on those metadata-aware scores

So the next sensible geo question is not a new standalone geo model.
It is a much narrower causal test:

- are there still weak `site/hour` regimes where a little extra geo emphasis helps
- and if so, is that effect concentrated in `Amphibia/Insecta`

## Design

`exp_024` is scaffolded from the fixed-artifact replay notebook:

- `notebooks/exp_019_v18_postproc_ablation.ipynb`

It keeps the whole `exp_015d` scorer fixed and changes only the final benchmark stage.

Main ablation ideas:

- weak-site geo prior boost
- weak-hour geo prior boost
- weak `site+hour` geo prior boost
- taxon-aware geo scaling for:
  - `Amphibia`
  - `Insecta`
  - combined texture taxa
- optional cheap geo-only OOF correction via `LogisticRegression`

Default weak regimes come from `exp_017`:

- weak sites: `S19`, `S08`, `S03`, `S13`, `S15`
- weak hours: `07`, `19`, `21`, `06`, `02`

## Outputs

The notebook writes:

- `variant_results.csv`
- `site_hour_report.csv`
- `taxon_geo_report.csv`
- `report_snapshot.json`
- optionally `geo_lr_classwise.csv`

## Intended Interpretation

If a weak-regime geo variant beats the manifest baseline locally, the next step would be a very thin submit patch on top of `exp_015d`.

If all variants stay neutral, that is strong evidence that the current V18 path has already extracted most of the useful geo signal.

## First Run Result

The first run came back clearly negative / closed.

Main snapshot:

- baseline file macro AUC: `0.992126`
- best overall variant: `manifest_baseline`
- best weak-regime variant: `manifest_baseline`
- baseline weak `site+hour` file macro AUC: `0.995090`
- no tested geo variant improved either the overall file metric or the weak-regime file metric

The closest variants still regressed:

- `weak_site_insecta_w15`: `0.992002`
- `weak_hour_insecta_w15`: `0.992002`
- all texture-prior boost variants were worse than baseline

The optional geo-only OOF correction was especially poor:

- trained classes: `36`
- improved: `1`
- worsened: `28`
- neutral: `7`
- mean AUC delta: about `-0.08096`
- median AUC delta: about `-0.01738`

The best geo-LR improvement was tiny:

- `23158`: `+0.000429`

But several classes collapsed badly:

- `22961`: about `-0.772`
- `47158son01`: about `-0.548`
- `47158son03`: about `-0.266`

## Interpretation

This is a strong causal result:

- the current `exp_015d` path already appears to extract almost all useful deployable geo signal
- extra weak-regime geo emphasis does not improve the local proxy
- a cheap geo-only correction layer is actively harmful in this setup

Practical consequence:

- do **not** build a thin public geo patch from this line
- treat the current deployable-form geo-regime branch as closed
- future work should move away from extra geo ablations and back toward new supervision / modeling ideas
