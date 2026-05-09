import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "notebooks/exp_040_v18_strict_filelevel_proxy_audit.ipynb"
DST = ROOT / "notebooks/exp_041_v18_texture_targeted_graft_audit.ipynb"


def src_of(cell):
    return "".join(cell.get("source", []))


def to_source(text):
    if not text.endswith("\n"):
        text += "\n"
    return text.splitlines(True)


nb = json.loads(SRC.read_text())
cells = nb["cells"]
code_cells = [c for c in cells if c.get("cell_type") == "code"]

# Cell 1: write outputs to exp_041.
cell1 = src_of(code_cells[1])
cell1 = cell1.replace(
    "Path('/kaggle/working/exp_040_v18_strict_filelevel_proxy_audit')",
    "Path('/kaggle/working/exp_041_v18_texture_targeted_graft_audit')",
)
cell1 = cell1.replace(
    "Path('experiments/outputs/exp_040_v18_strict_filelevel_proxy_audit')",
    "Path('experiments/outputs/exp_041_v18_texture_targeted_graft_audit')",
)
code_cells[1]["source"] = to_source(cell1)

# Cell 8: targeted texture-only graft audit.
code_cells[8]["source"] = to_source(
    r"""# Cell 8 — Texture-targeted graft audit

def file_level_confidence_scale(preds, n_windows=12, top_k=2):
    N, C = preds.shape
    assert N % n_windows == 0
    view = preds.reshape(-1, n_windows, C)
    sorted_view = np.sort(view, axis=1)
    top_k_mean = sorted_view[:, -top_k:, :].mean(axis=1, keepdims=True)
    scaled = view * top_k_mean
    return np.clip(scaled.reshape(N, C), 0.0, 1.0)


def rank_aware_scaling(scores, n_windows=12, power=0.5):
    N, C = scores.shape
    assert N % n_windows == 0
    view = scores.reshape(-1, n_windows, C)
    file_max = view.max(axis=1, keepdims=True)
    scale = np.power(np.clip(file_max, 0.0, 1.0), power)
    scaled = view * scale
    return np.clip(scaled.reshape(N, C), 0.0, 1.0)


def adaptive_delta_smooth(scores, n_windows=12, base_alpha=0.20):
    result = scores.copy().reshape(-1, n_windows, scores.shape[1])
    original = scores.reshape(-1, n_windows, scores.shape[1])
    for i in range(1, n_windows - 1):
        conf = original[:, i, :].max(axis=-1, keepdims=True)
        a = base_alpha * (1.0 - conf)
        neighbor_avg = (original[:, i - 1, :] + original[:, i + 1, :]) / 2.0
        result[:, i, :] = (1.0 - a) * original[:, i, :] + a * neighbor_avg
    return np.clip(result.reshape(scores.shape), 0.0, 1.0)


def apply_per_class_thresholds(scores, thresholds):
    N, C = scores.shape
    assert C == len(thresholds)
    scaled = np.copy(scores)
    for c in range(C):
        t = float(thresholds[c])
        mask_above = scores[:, c] > t
        scaled[mask_above, c] = 0.5 + 0.5 * (scores[mask_above, c] - t) / (1 - t + 1e-8)
        scaled[~mask_above, c] = 0.5 * scores[~mask_above, c] / (t + 1e-8)
    return np.clip(scaled, 0.0, 1.0)


def apply_file_level_calibrators(scores, calibrators, n_windows=12):
    scores_files = scores.reshape(-1, n_windows, scores.shape[1]).copy()
    file_max = scores_files.max(axis=1)
    calibrated_max = file_max.copy()
    for ci, cal in enumerate(calibrators):
        if cal is None:
            continue
        calibrated_max[:, ci] = np.clip(cal.transform(file_max[:, ci]), 0.0, 1.0)
    scale = calibrated_max / np.clip(file_max, 1e-6, None)
    scale = np.clip(scale, 0.25, 4.0)
    scores_files *= scale[:, None, :]
    return np.clip(scores_files.reshape(scores.shape), 0.0, 1.0)


def build_tempered_probs(logit_scores):
    class_temperatures = np.ones(N_CLASSES, dtype=np.float32) * float(CFG['temperature']['aves'])
    for ci, label in enumerate(PRIMARY_LABELS):
        if CLASS_NAME_MAP.get(label, 'Aves') in TEXTURE_TAXA:
            class_temperatures[ci] = float(CFG['temperature']['texture'])
    return sigmoid(logit_scores / class_temperatures[None, :]).astype(np.float32)


def macro_auc_subset(y_true, y_score, idx):
    idx = np.array(idx, dtype=np.int64)
    idx = idx[(idx >= 0) & (idx < y_true.shape[1])]
    if len(idx) == 0:
        return float('nan'), 0
    keep = y_true[:, idx].sum(axis=0) > 0
    if keep.sum() == 0:
        return float('nan'), 0
    score = roc_auc_score(y_true[:, idx][:, keep], y_score[:, idx][:, keep], average='macro')
    return float(score), int(keep.sum())


def macro_auc_with_count(y_true, y_score):
    keep = y_true.sum(axis=0) > 0
    if keep.sum() == 0:
        return float('nan'), 0
    score = roc_auc_score(y_true[:, keep], y_score[:, keep], average='macro')
    return float(score), int(keep.sum())


TAXON_TO_IDX = {}
for taxon in sorted(set(CLASS_NAME_MAP.get(label, 'Unknown') for label in PRIMARY_LABELS)):
    TAXON_TO_IDX[taxon] = np.array(
        [i for i, label in enumerate(PRIMARY_LABELS) if CLASS_NAME_MAP.get(label, 'Unknown') == taxon],
        dtype=np.int32,
    )

TEXTURE_ALL_IDX = np.array(
    [i for i, label in enumerate(PRIMARY_LABELS) if CLASS_NAME_MAP.get(label, 'Unknown') in TEXTURE_TAXA],
    dtype=np.int32,
)
EVENT_ALL_IDX = np.array(
    [i for i in range(N_CLASSES) if i not in set(TEXTURE_ALL_IDX.tolist())],
    dtype=np.int32,
)


def build_variant_probs(replay, top_k, rank_aware, rank_power, delta_alpha,
                        use_thresholds, aves_smooth_alpha=0.0, use_calibration=False):
    probs = build_tempered_probs(replay['final_scores'])
    if use_calibration and CALIBRATORS is not None:
        probs = apply_file_level_calibrators(probs, CALIBRATORS, n_windows=N_WINDOWS)
    if top_k > 0:
        probs = file_level_confidence_scale(probs, n_windows=N_WINDOWS, top_k=top_k)
    if rank_aware:
        probs = rank_aware_scaling(probs, n_windows=N_WINDOWS, power=rank_power)
    if aves_smooth_alpha > 0:
        probs = smooth_cols_fixed12(probs, idx_mapped_active_event, alpha=aves_smooth_alpha)
    if delta_alpha > 0:
        probs = adaptive_delta_smooth(probs, n_windows=N_WINDOWS, base_alpha=delta_alpha)
    if use_thresholds:
        probs = apply_per_class_thresholds(probs, PER_CLASS_THRESHOLDS)
    return np.clip(probs, 0.0, 1.0).astype(np.float32, copy=False)


def graft_columns(base_probs, donor_probs, cols, weight=1.0):
    out = base_probs.copy()
    cols = np.array(cols, dtype=np.int64)
    if len(cols):
        out[:, cols] = (1.0 - weight) * base_probs[:, cols] + weight * donor_probs[:, cols]
    return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)


def score_probs(name, probs, metadata):
    file_probs = probs.reshape(-1, N_WINDOWS, N_CLASSES).max(axis=1)
    file_y = Y_FULL.reshape(-1, N_WINDOWS, N_CLASSES).max(axis=1)

    row_macro_auc, row_scored = macro_auc_with_count(Y_FULL, probs)
    file_macro_auc, file_scored = macro_auc_with_count(file_y, file_probs)
    row_texture_auc, row_texture_scored = macro_auc_subset(Y_FULL, probs, idx_mapped_active_texture)
    file_texture_auc, file_texture_scored = macro_auc_subset(file_y, file_probs, idx_mapped_active_texture)
    row_event_auc, row_event_scored = macro_auc_subset(Y_FULL, probs, idx_mapped_active_event)
    file_event_auc, file_event_scored = macro_auc_subset(file_y, file_probs, idx_mapped_active_event)

    taxon_scores = {}
    for taxon, idx in TAXON_TO_IDX.items():
        auc, scored = macro_auc_subset(file_y, file_probs, idx)
        if scored > 0 and np.isfinite(auc):
            taxon_scores[taxon] = {'auc': float(auc), 'scored_classes': int(scored)}

    min_taxon_file_auc = min((v['auc'] for v in taxon_scores.values()), default=float('nan'))
    texture_event_balance = min(
        x for x in [file_texture_auc, file_event_auc]
        if np.isfinite(x)
    ) if np.isfinite(file_texture_auc) or np.isfinite(file_event_auc) else float('nan')

    row = {
        'variant': name,
        'row_macro_auc': row_macro_auc,
        'row_scored_classes': row_scored,
        'file_macro_auc': file_macro_auc,
        'file_scored_classes': file_scored,
        'row_texture_auc': row_texture_auc,
        'row_texture_scored_classes': row_texture_scored,
        'file_texture_auc': file_texture_auc,
        'file_texture_scored_classes': file_texture_scored,
        'row_event_auc': row_event_auc,
        'row_event_scored_classes': row_event_scored,
        'file_event_auc': file_event_auc,
        'file_event_scored_classes': file_event_scored,
        'min_taxon_file_auc': float(min_taxon_file_auc),
        'texture_event_balance': float(texture_event_balance),
        'prob_mean': float(probs.mean()),
        'prob_max': float(probs.max()),
        'taxon_scores': taxon_scores,
    }
    row.update(metadata)
    return row


manifest_tta = tuple(CFG.get('tta_shifts', [0]))
base_top_k = int(CFG.get('file_level_top_k', 0))
base_rank = bool(CFG.get('rank_aware_scale', False))
base_rank_power = float(CFG.get('rank_aware_power', 0.5))
base_delta = float(CFG.get('delta_shift_alpha', 0.0))
base_cal = CALIBRATORS is not None

def spec(top_k=base_top_k, rank_aware=base_rank, rank_power=base_rank_power,
         delta_alpha=base_delta, use_thresholds=True, aves_smooth_alpha=0.0,
         use_calibration=base_cal):
    return {
        'top_k': top_k,
        'rank_aware': rank_aware,
        'rank_power': rank_power,
        'delta_alpha': delta_alpha,
        'use_thresholds': use_thresholds,
        'aves_smooth_alpha': aves_smooth_alpha,
        'use_calibration': use_calibration,
    }

replay = replay_v18_stack(manifest_tta)
base_spec = spec()
base_probs = build_variant_probs(replay, **base_spec)

donor_specs = {
    'no_file_scale': spec(top_k=0),
    'topk1': spec(top_k=1),
    'topk3': spec(top_k=3),
    'no_rank_aware': spec(rank_aware=False),
    'rank035': spec(rank_aware=True, rank_power=0.35),
}
donor_probs = {name: build_variant_probs(replay, **cfg) for name, cfg in donor_specs.items()}

results = []
taxon_rows = []

def add_result(name, probs, metadata):
    row = score_probs(name, probs, metadata)
    taxon_scores = row.pop('taxon_scores')
    results.append(row)
    for taxon, info in taxon_scores.items():
        taxon_rows.append({
            'variant': name,
            'taxon': taxon,
            'file_taxon_auc': info['auc'],
            'scored_classes': info['scored_classes'],
        })
    print(
        f"{name}: file_auc={row['file_macro_auc']:.6f}, "
        f"texture_file_auc={row['file_texture_auc']:.6f}, "
        f"event_file_auc={row['file_event_auc']:.6f}, row_auc={row['row_macro_auc']:.6f}"
    )

add_result('manifest_baseline', base_probs, {
    'variant_family': 'baseline',
    'donor_variant': 'manifest_baseline',
    'graft_cols': 'none',
    'graft_weight': 0.0,
    **base_spec,
})

for donor_name, probs in donor_probs.items():
    add_result(f'full_{donor_name}', probs, {
        'variant_family': 'full_donor_diagnostic',
        'donor_variant': donor_name,
        'graft_cols': 'all',
        'graft_weight': 1.0,
        **donor_specs[donor_name],
    })

for donor_name in ['no_file_scale', 'topk1', 'topk3', 'rank035', 'no_rank_aware']:
    for weight in [0.25, 0.50, 0.75, 1.00]:
        probs = graft_columns(base_probs, donor_probs[donor_name], TEXTURE_ALL_IDX, weight=weight)
        add_result(f'texture_graft_{donor_name}_w{int(weight * 100):03d}', probs, {
            'variant_family': 'texture_graft',
            'donor_variant': donor_name,
            'graft_cols': 'Amphibia+Insecta',
            'graft_weight': float(weight),
            **donor_specs[donor_name],
        })

variant_results = pd.DataFrame(results)
taxon_results = pd.DataFrame(taxon_rows)
baseline = variant_results.loc[variant_results['variant'] == 'manifest_baseline'].iloc[0].to_dict()

delta_cols = [
    'row_macro_auc',
    'file_macro_auc',
    'file_texture_auc',
    'file_event_auc',
    'min_taxon_file_auc',
    'texture_event_balance',
]
for col in delta_cols:
    variant_results[f'{col}_delta'] = variant_results[col] - float(baseline[col])

drop_cols = ['file_macro_auc_delta', 'file_texture_auc_delta', 'file_event_auc_delta', 'min_taxon_file_auc_delta']
variant_results['worst_file_regime_delta'] = variant_results[drop_cols].min(axis=1)
variant_results['strict_file_score'] = (
    variant_results['file_macro_auc']
    + 0.20 * variant_results['file_texture_auc_delta'].clip(upper=0)
    + 0.20 * variant_results['file_event_auc_delta'].clip(upper=0)
    + 0.20 * variant_results['min_taxon_file_auc_delta'].clip(upper=0)
)
variant_results['strict_public_candidate'] = (
    (variant_results['file_macro_auc_delta'] > 0.00010)
    & (variant_results['worst_file_regime_delta'] >= -0.00010)
    & (variant_results['variant_family'] == 'texture_graft')
)

strict_results = (
    variant_results
    .sort_values(['strict_public_candidate', 'strict_file_score', 'file_macro_auc', 'file_texture_auc_delta'],
                 ascending=[False, False, False, False])
    .reset_index(drop=True)
)
display_cols = [
    'variant',
    'strict_public_candidate',
    'variant_family',
    'donor_variant',
    'graft_weight',
    'strict_file_score',
    'file_macro_auc',
    'file_macro_auc_delta',
    'file_texture_auc',
    'file_texture_auc_delta',
    'file_event_auc',
    'file_event_auc_delta',
    'min_taxon_file_auc',
    'min_taxon_file_auc_delta',
    'worst_file_regime_delta',
    'row_macro_auc',
    'row_macro_auc_delta',
]
display(strict_results[display_cols])

if len(taxon_results):
    taxon_pivot = taxon_results.pivot(index='variant', columns='taxon', values='file_taxon_auc').reset_index()
    display(taxon_pivot)
else:
    taxon_pivot = pd.DataFrame()

candidate_rows = strict_results[strict_results['strict_public_candidate']]
best_candidate = candidate_rows.iloc[0].to_dict() if len(candidate_rows) else None
best_file = strict_results.iloc[0].to_dict()

report_snapshot = {
    'experiment_id': 'exp_041',
    'experiment_name': 'v18_texture_targeted_graft_audit',
    'artifacts_dir': str(ARTIFACTS_DIR),
    'cache_meta': str(CACHE_META),
    'device': str(DEVICE),
    'n_variants': int(len(strict_results)),
    'baseline_variant': baseline['variant'],
    'baseline_file_macro_auc': float(baseline['file_macro_auc']),
    'baseline_file_texture_auc': float(baseline['file_texture_auc']),
    'baseline_file_event_auc': float(baseline['file_event_auc']),
    'best_by_strict_file_score': best_file['variant'],
    'best_strict_file_score': float(best_file['strict_file_score']),
    'best_file_macro_auc': float(best_file['file_macro_auc']),
    'best_file_macro_auc_delta': float(best_file['file_macro_auc_delta']),
    'strict_public_candidate': None if best_candidate is None else best_candidate['variant'],
    'strict_public_candidate_file_macro_auc_delta': None if best_candidate is None else float(best_candidate['file_macro_auc_delta']),
    'strict_public_candidate_worst_file_regime_delta': None if best_candidate is None else float(best_candidate['worst_file_regime_delta']),
    'texture_columns': int(len(TEXTURE_ALL_IDX)),
    'event_columns': int(len(EVENT_ALL_IDX)),
    'known_public_negatives': {
        'full_no_rank_aware': 0.928,
        'full_no_file_scale': 0.922,
    },
    'selection_note': 'Only texture_graft variants can be public candidates; full donor variants are diagnostics.',
}
print(json.dumps(report_snapshot, indent=2, default=str))

if SAVE_RESULTS:
    strict_results.to_csv(OUTPUT_DIR / 'texture_graft_variant_results.csv', index=False)
    taxon_results.to_csv(OUTPUT_DIR / 'taxon_file_auc.csv', index=False)
    if len(taxon_pivot):
        taxon_pivot.to_csv(OUTPUT_DIR / 'taxon_file_auc_pivot.csv', index=False)
    (OUTPUT_DIR / 'report_snapshot.json').write_text(json.dumps(report_snapshot, indent=2, default=str))
    print('Saved texture graft audit results to:', OUTPUT_DIR)
"""
)

for cell in cells:
    if cell.get("cell_type") == "code":
        cell["execution_count"] = None
        cell["outputs"] = []

DST.parent.mkdir(parents=True, exist_ok=True)
DST.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
print(DST)
