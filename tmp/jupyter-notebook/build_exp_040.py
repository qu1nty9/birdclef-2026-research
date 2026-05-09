import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "notebooks/exp_019_v18_postproc_ablation.ipynb"
DST = ROOT / "notebooks/exp_040_v18_strict_filelevel_proxy_audit.ipynb"


def src_of(cell):
    return "".join(cell.get("source", []))


def to_source(text):
    if not text.endswith("\n"):
        text += "\n"
    return text.splitlines(True)


nb = json.loads(SRC.read_text())
cells = nb["cells"]
code_cells = [c for c in cells if c.get("cell_type") == "code"]

# Cell 0: retitle controls for the stricter audit.
cell0 = src_of(code_cells[0])
cell0 = cell0.replace("RUN_TTA_VARIANTS = True", "RUN_TTA_VARIANTS = False")
cell0 = cell0.replace("SAVE_RESULTS = True", "SAVE_RESULTS = True")
code_cells[0]["source"] = to_source(cell0)

# Cell 1: write outputs to exp_040.
cell1 = src_of(code_cells[1])
cell1 = cell1.replace(
    "Path('/kaggle/working/exp_019_v18_postproc_ablation')",
    "Path('/kaggle/working/exp_040_v18_strict_filelevel_proxy_audit')",
)
cell1 = cell1.replace(
    "Path('experiments/outputs/exp_019_v18_postproc_ablation')",
    "Path('experiments/outputs/exp_040_v18_strict_filelevel_proxy_audit')",
)
code_cells[1]["source"] = to_source(cell1)

# Cell 8: replace row-AUC-first sweep with a stricter file-level/regime audit.
code_cells[8]["source"] = to_source(
    r"""# Cell 8 — Strict file-level and regime-aware postprocess audit

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


def score_variant(name, replay, spec):
    probs = build_variant_probs(
        replay,
        top_k=spec['top_k'],
        rank_aware=spec['rank_aware'],
        rank_power=spec['rank_power'],
        delta_alpha=spec['delta_alpha'],
        use_thresholds=spec['use_thresholds'],
        aves_smooth_alpha=spec.get('aves_smooth_alpha', 0.0),
        use_calibration=spec.get('use_calibration', False),
    )
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

    return {
        'variant': name,
        'tta_shifts': list(replay['tta_shifts']),
        'replay_wall_time': float(replay['wall_time']),
        'use_calibration': bool(spec.get('use_calibration', False) and CALIBRATORS is not None),
        'top_k': int(spec['top_k']),
        'rank_aware': bool(spec['rank_aware']),
        'rank_power': float(spec['rank_power']),
        'delta_alpha': float(spec['delta_alpha']),
        'aves_smooth_alpha': float(spec.get('aves_smooth_alpha', 0.0)),
        'use_thresholds': bool(spec['use_thresholds']),
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


manifest_tta = tuple(CFG.get('tta_shifts', [0]))
base_top_k = int(CFG.get('file_level_top_k', 0))
base_rank = bool(CFG.get('rank_aware_scale', False))
base_rank_power = float(CFG.get('rank_aware_power', 0.5))
base_delta = float(CFG.get('delta_shift_alpha', 0.0))
base_cal = CALIBRATORS is not None

variant_specs = [
    {'name': 'manifest_baseline', 'tta_shifts': manifest_tta, 'top_k': base_top_k, 'rank_aware': base_rank, 'rank_power': base_rank_power, 'delta_alpha': base_delta, 'aves_smooth_alpha': 0.0, 'use_thresholds': True, 'use_calibration': base_cal},
    {'name': 'topk1_manifest', 'tta_shifts': manifest_tta, 'top_k': 1, 'rank_aware': base_rank, 'rank_power': base_rank_power, 'delta_alpha': base_delta, 'aves_smooth_alpha': 0.0, 'use_thresholds': True, 'use_calibration': base_cal},
    {'name': 'topk3_manifest', 'tta_shifts': manifest_tta, 'top_k': 3, 'rank_aware': base_rank, 'rank_power': base_rank_power, 'delta_alpha': base_delta, 'aves_smooth_alpha': 0.0, 'use_thresholds': True, 'use_calibration': base_cal},
    {'name': 'rank035_manifest', 'tta_shifts': manifest_tta, 'top_k': base_top_k, 'rank_aware': True, 'rank_power': 0.35, 'delta_alpha': base_delta, 'aves_smooth_alpha': 0.0, 'use_thresholds': True, 'use_calibration': base_cal},
    {'name': 'rank045_manifest', 'tta_shifts': manifest_tta, 'top_k': base_top_k, 'rank_aware': True, 'rank_power': 0.45, 'delta_alpha': base_delta, 'aves_smooth_alpha': 0.0, 'use_thresholds': True, 'use_calibration': base_cal},
    {'name': 'rank050_manifest', 'tta_shifts': manifest_tta, 'top_k': base_top_k, 'rank_aware': True, 'rank_power': 0.50, 'delta_alpha': base_delta, 'aves_smooth_alpha': 0.0, 'use_thresholds': True, 'use_calibration': base_cal},
    {'name': 'delta010_manifest', 'tta_shifts': manifest_tta, 'top_k': base_top_k, 'rank_aware': base_rank, 'rank_power': base_rank_power, 'delta_alpha': 0.10, 'aves_smooth_alpha': 0.0, 'use_thresholds': True, 'use_calibration': base_cal},
    {'name': 'delta015_manifest', 'tta_shifts': manifest_tta, 'top_k': base_top_k, 'rank_aware': base_rank, 'rank_power': base_rank_power, 'delta_alpha': 0.15, 'aves_smooth_alpha': 0.0, 'use_thresholds': True, 'use_calibration': base_cal},
    {'name': 'delta025_manifest', 'tta_shifts': manifest_tta, 'top_k': base_top_k, 'rank_aware': base_rank, 'rank_power': base_rank_power, 'delta_alpha': 0.25, 'aves_smooth_alpha': 0.0, 'use_thresholds': True, 'use_calibration': base_cal},
    {'name': 'no_thresholds', 'tta_shifts': manifest_tta, 'top_k': base_top_k, 'rank_aware': base_rank, 'rank_power': base_rank_power, 'delta_alpha': base_delta, 'aves_smooth_alpha': 0.0, 'use_thresholds': False, 'use_calibration': base_cal},
    {'name': 'no_delta_smooth', 'tta_shifts': manifest_tta, 'top_k': base_top_k, 'rank_aware': base_rank, 'rank_power': base_rank_power, 'delta_alpha': 0.0, 'aves_smooth_alpha': 0.0, 'use_thresholds': True, 'use_calibration': base_cal},
    {'name': 'no_rank_aware_closed_public_negative', 'tta_shifts': manifest_tta, 'top_k': base_top_k, 'rank_aware': False, 'rank_power': base_rank_power, 'delta_alpha': base_delta, 'aves_smooth_alpha': 0.0, 'use_thresholds': True, 'use_calibration': base_cal},
    {'name': 'no_file_scale_closed_public_negative', 'tta_shifts': manifest_tta, 'top_k': 0, 'rank_aware': base_rank, 'rank_power': base_rank_power, 'delta_alpha': base_delta, 'aves_smooth_alpha': 0.0, 'use_thresholds': True, 'use_calibration': base_cal},
]

if RUN_TTA_VARIANTS:
    variant_specs.extend([
        {'name': 'tta_shift_p1', 'tta_shifts': (0, 1), 'top_k': base_top_k, 'rank_aware': base_rank, 'rank_power': base_rank_power, 'delta_alpha': base_delta, 'aves_smooth_alpha': 0.0, 'use_thresholds': True, 'use_calibration': base_cal},
        {'name': 'tta_shift_pm1', 'tta_shifts': (0, 1, -1), 'top_k': base_top_k, 'rank_aware': base_rank, 'rank_power': base_rank_power, 'delta_alpha': base_delta, 'aves_smooth_alpha': 0.0, 'use_thresholds': True, 'use_calibration': base_cal},
    ])

results = []
taxon_rows = []
for spec in variant_specs:
    replay = replay_v18_stack(spec['tta_shifts'])
    result = score_variant(spec['name'], replay, spec)
    taxon_scores = result.pop('taxon_scores')
    results.append(result)
    for taxon, info in taxon_scores.items():
        taxon_rows.append({
            'variant': spec['name'],
            'taxon': taxon,
            'file_taxon_auc': info['auc'],
            'scored_classes': info['scored_classes'],
        })
    print(
        f"{result['variant']}: file_auc={result['file_macro_auc']:.6f}, "
        f"texture_file_auc={result['file_texture_auc']:.6f}, "
        f"row_auc={result['row_macro_auc']:.6f}, min_taxon={result['min_taxon_file_auc']:.6f}"
    )

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
    & (~variant_results['variant'].str.contains('closed_public_negative'))
)

strict_results = (
    variant_results
    .sort_values(['strict_public_candidate', 'strict_file_score', 'file_macro_auc', 'worst_file_regime_delta'],
                 ascending=[False, False, False, False])
    .reset_index(drop=True)
)
display_cols = [
    'variant',
    'strict_public_candidate',
    'strict_file_score',
    'file_macro_auc',
    'file_macro_auc_delta',
    'file_texture_auc',
    'file_texture_auc_delta',
    'file_event_auc',
    'file_event_auc_delta',
    'min_taxon_file_auc',
    'min_taxon_file_auc_delta',
    'row_macro_auc',
    'row_macro_auc_delta',
    'top_k',
    'rank_aware',
    'rank_power',
    'delta_alpha',
    'use_thresholds',
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
    'experiment_id': 'exp_040',
    'experiment_name': 'v18_strict_filelevel_proxy_audit',
    'artifacts_dir': str(ARTIFACTS_DIR),
    'cache_meta': str(CACHE_META),
    'device': str(DEVICE),
    'n_variants': int(len(strict_results)),
    'baseline_variant': baseline['variant'],
    'baseline_file_macro_auc': float(baseline['file_macro_auc']),
    'baseline_row_macro_auc': float(baseline['row_macro_auc']),
    'best_by_strict_file_score': best_file['variant'],
    'best_strict_file_score': float(best_file['strict_file_score']),
    'best_file_macro_auc': float(best_file['file_macro_auc']),
    'best_file_macro_auc_delta': float(best_file['file_macro_auc_delta']),
    'strict_public_candidate': None if best_candidate is None else best_candidate['variant'],
    'strict_public_candidate_file_macro_auc_delta': None if best_candidate is None else float(best_candidate['file_macro_auc_delta']),
    'strict_public_candidate_worst_file_regime_delta': None if best_candidate is None else float(best_candidate['worst_file_regime_delta']),
    'known_public_negatives': {
        'no_rank_aware': 0.928,
        'no_file_scale': 0.922,
    },
    'selection_note': 'Primary selection is file/regime stability, not row macro AUC.',
}
print(json.dumps(report_snapshot, indent=2, default=str))

if SAVE_RESULTS:
    strict_results.to_csv(OUTPUT_DIR / 'strict_variant_results.csv', index=False)
    taxon_results.to_csv(OUTPUT_DIR / 'taxon_file_auc.csv', index=False)
    if len(taxon_pivot):
        taxon_pivot.to_csv(OUTPUT_DIR / 'taxon_file_auc_pivot.csv', index=False)
    (OUTPUT_DIR / 'report_snapshot.json').write_text(json.dumps(report_snapshot, indent=2, default=str))
    print('Saved strict audit results to:', OUTPUT_DIR)
"""
)

for cell in cells:
    if cell.get("cell_type") == "code":
        cell["execution_count"] = None
        cell["outputs"] = []

DST.parent.mkdir(parents=True, exist_ok=True)
DST.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
print(DST)
