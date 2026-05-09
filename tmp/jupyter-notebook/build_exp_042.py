import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "notebooks/kaggle_submission_exp_038_pantanal_onnx_fast_noalign.ipynb"
DST = ROOT / "notebooks/kaggle_submission_exp_042_v18_texture_graft_no_file_scale.ipynb"


def src_of(cell):
    return "".join(cell.get("source", []))


def to_source(text):
    if not text.endswith("\n"):
        text += "\n"
    return text.splitlines(True)


nb = json.loads(SRC.read_text())
cells = nb["cells"]
code_cells = [c for c in cells if c.get("cell_type") == "code"]

# Markdown: document the final surgical graft in the notebook story.
for cell in cells:
    if cell.get("cell_type") != "markdown":
        continue
    text = src_of(cell)
    if "Pipeline:" in text and "file-level scale" in text:
        text = text.replace(
            "per-taxon temp -> file-level scale -> rank-aware -> delta smooth -> per-class threshold -> final",
            "per-taxon temp -> baseline file-level scale -> rank-aware -> delta smooth -> per-class threshold -> texture-only no-file-scale graft -> final",
        )
        text += (
            "\n\n## exp_042 texture graft\n"
            "This submit keeps the stable exp_038 V18/ONNX-fast stack intact, then replaces only "
            "`Amphibia` and `Insecta` columns with the `no_file_scale` donor postprocess found by "
            "`exp_041_v18_texture_targeted_graft_audit`.\n"
        )
        cell["source"] = to_source(text)
        break

# Cell 3/code index 3 in the existing build scripts: add a small explicit
# submit-time graft config after the V18 postprocess parameters.
cell3 = src_of(code_cells[3])
needle = 'CFG["delta_shift_alpha"] = 0.20\n'
patch = '''CFG["delta_shift_alpha"] = 0.20

# exp_042: surgical texture-only graft selected by exp_041.
# Baseline stays unchanged for all non-texture classes. The donor differs only
# by disabling file-level confidence scaling before the shared rank/delta/threshold steps.
CFG["texture_graft_enabled"] = True
CFG["texture_graft_taxa"] = ["Amphibia", "Insecta"]
CFG["texture_graft_weight"] = 1.0
CFG["texture_graft_donor_file_level_top_k"] = 0
CFG["texture_graft_source"] = "exp_041:texture_graft_no_file_scale_w100"
'''
if needle not in cell3:
    raise RuntimeError("Could not find delta_shift_alpha anchor in CFG upgrade cell")
cell3 = cell3.replace(needle, patch, 1)
code_cells[3]["source"] = to_source(cell3)

# Cell 30/code index 30: replace the final postprocess block with an equivalent
# baseline builder plus a donor builder. The expensive model inference still runs once.
code_cells[30]["source"] = to_source(
    '''# Cell 18 — V18 post-processing pipeline + exp_042 texture-only graft

# V17: Optimize per-class thresholds from OOF (train mode only)
PER_CLASS_THRESHOLDS = np.full(N_CLASSES, 0.5, dtype=np.float32)
if MODE == "train" and oof_proto_flat is not None:
    print("Optimizing per-class thresholds from OOF...")
    best_thresholds, best_scores = optimize_per_class_thresholds(
        oof_proto_flat, Y_FULL, n_windows=N_WINDOWS, thresholds=CFG["threshold_grid"]
    )
    PER_CLASS_THRESHOLDS = best_thresholds.astype(np.float32)
    print(f"  Mean threshold: {best_thresholds.mean():.3f}")
    print(f"  Threshold range: [{best_thresholds.min():.2f}, {best_thresholds.max():.2f}]")
    print(f"  Mean F1 (proxy): {best_scores.mean():.3f}")
    
    high_t = np.where(best_thresholds > 0.6)[0]
    low_t = np.where(best_thresholds < 0.4)[0]
    if len(high_t) > 0:
        print(f"  High threshold classes (>0.6): {len(high_t)}")
    if len(low_t) > 0:
        print(f"  Low threshold classes (<0.4): {len(low_t)}")
else:
    print("Using default per-class thresholds (0.5) for submit mode")


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def adaptive_delta_smooth(probs, n_windows, base_alpha=0.20):
    n_files = probs.shape[0] // n_windows
    result = probs.copy()
    view = result.reshape(n_files, n_windows, -1)
    p_view = probs.reshape(n_files, n_windows, -1)
    for i in range(1, n_windows - 1):
        conf = p_view[:, i, :].max(axis=-1, keepdims=True)
        a = base_alpha * (1.0 - conf)
        neighbor_avg = (p_view[:, i-1, :] + p_view[:, i+1, :]) / 2.0
        view[:, i, :] = (1.0 - a) * p_view[:, i, :] + a * neighbor_avg
    return result.reshape(probs.shape)


# --- Step 1: Per-taxon temperature scaling ---
temp_cfg = CFG["temperature"]
T_AVES = temp_cfg["aves"]
T_TEXTURE = temp_cfg["texture"]

class_temperatures = np.ones(N_CLASSES, dtype=np.float32) * T_AVES
for ci, label in enumerate(PRIMARY_LABELS):
    cn = CLASS_NAME_MAP.get(label, "Aves")
    if cn in TEXTURE_TAXA:
        class_temperatures[ci] = T_TEXTURE

print(f"\\nPer-taxon temperature: Aves={T_AVES}, Texture={T_TEXTURE}")


def build_submit_probs(file_level_top_k, variant_name):
    """Apply the submit postprocess stack without rerunning model inference."""
    scaled_scores = final_test_scores / class_temperatures[None, :]
    out = sigmoid(scaled_scores)

    # File-level confidence scaling is the only knob that differs for the donor.
    top_k = int(file_level_top_k)
    if top_k > 0:
        print(f"[{variant_name}] Applying file-level confidence scaling (top_k={top_k})")
        out = file_level_confidence_scale(out, n_windows=N_WINDOWS, top_k=top_k)
        out = np.clip(out, 0.0, 1.0)
    else:
        print(f"[{variant_name}] Skipping file-level confidence scaling")

    if CFG.get("rank_aware_scale", False):
        power = CFG.get("rank_aware_power", 0.5)
        print(f"[{variant_name}] Applying rank-aware scaling (power={power})")
        out = rank_aware_scaling(out, n_windows=N_WINDOWS, power=power)
        out = np.clip(out, 0.0, 1.0)

    alpha = CFG.get("delta_shift_alpha", 0.0)
    if alpha > 0:
        print(f"[{variant_name}] Applying delta shift smoothing (alpha={alpha})")
        out = adaptive_delta_smooth(out, n_windows=N_WINDOWS, base_alpha=alpha)
        out = np.clip(out, 0.0, 1.0)

    print(f"[{variant_name}] Applying per-class threshold sharpening")
    out = apply_per_class_thresholds(out, PER_CLASS_THRESHOLDS, n_windows=N_WINDOWS)
    return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)


baseline_top_k = int(CFG.get("file_level_top_k", 0))
probs_baseline = build_submit_probs(baseline_top_k, "baseline")
probs = probs_baseline.copy()

texture_graft_enabled = bool(CFG.get("texture_graft_enabled", False))
texture_graft_taxa = set(CFG.get("texture_graft_taxa", []))
texture_graft_idx = np.array(
    [ci for ci, label in enumerate(PRIMARY_LABELS) if CLASS_NAME_MAP.get(label, "Aves") in texture_graft_taxa],
    dtype=np.int32,
)

if texture_graft_enabled:
    if len(texture_graft_idx) == 0:
        raise RuntimeError("Texture graft is enabled but no graft columns were resolved.")
    donor_top_k = int(CFG.get("texture_graft_donor_file_level_top_k", 0))
    graft_weight = float(CFG.get("texture_graft_weight", 1.0))
    if not (0.0 <= graft_weight <= 1.0):
        raise ValueError(f"texture_graft_weight must be in [0, 1], got {graft_weight}")

    donor_probs = build_submit_probs(donor_top_k, "texture_donor_no_file_scale")
    graft_before = probs_baseline[:, texture_graft_idx].copy()
    graft_after = (1.0 - graft_weight) * graft_before + graft_weight * donor_probs[:, texture_graft_idx]
    probs[:, texture_graft_idx] = graft_after
    probs = np.clip(probs, 0.0, 1.0).astype(np.float32, copy=False)

    non_graft_idx = np.setdiff1d(np.arange(N_CLASSES, dtype=np.int32), texture_graft_idx)
    max_non_graft_delta = float(np.max(np.abs(probs[:, non_graft_idx] - probs_baseline[:, non_graft_idx]))) if len(non_graft_idx) else 0.0
    mean_graft_abs_delta = float(np.mean(np.abs(probs[:, texture_graft_idx] - graft_before)))
    max_graft_abs_delta = float(np.max(np.abs(probs[:, texture_graft_idx] - graft_before)))
    if max_non_graft_delta != 0.0:
        raise RuntimeError(f"Non-graft columns changed unexpectedly: {max_non_graft_delta}")

    LOGS["texture_graft"] = {
        "enabled": True,
        "source": CFG.get("texture_graft_source"),
        "taxa": sorted(texture_graft_taxa),
        "n_columns": int(len(texture_graft_idx)),
        "labels": [PRIMARY_LABELS[i] for i in texture_graft_idx],
        "baseline_file_level_top_k": baseline_top_k,
        "donor_file_level_top_k": donor_top_k,
        "weight": graft_weight,
        "mean_abs_delta": mean_graft_abs_delta,
        "max_abs_delta": max_graft_abs_delta,
        "max_non_graft_delta": max_non_graft_delta,
    }
    print(
        f"Applied exp_042 texture graft: {len(texture_graft_idx)} columns, "
        f"weight={graft_weight}, donor_top_k={donor_top_k}, "
        f"mean_abs_delta={mean_graft_abs_delta:.8f}, max_abs_delta={max_graft_abs_delta:.8f}"
    )
else:
    LOGS["texture_graft"] = {"enabled": False}
    print("Texture graft disabled; using pure baseline probabilities")

# --- Build submission ---
submission = pd.DataFrame(probs, columns=PRIMARY_LABELS)
submission.insert(0, "row_id", meta_test["row_id"].values)
submission[PRIMARY_LABELS] = submission[PRIMARY_LABELS].astype(np.float32)

expected_rows = len(test_paths) * N_WINDOWS
assert len(submission) == expected_rows, f"Expected {expected_rows}, got {len(submission)}"
assert submission.columns.tolist() == ["row_id"] + PRIMARY_LABELS
assert not submission.isna().any().any()

submission.to_csv("submission.csv", index=False)

print("\\nSaved submission.csv")
print("Submission shape:", submission.shape)
print(f"Final score range: {probs.min():.6f} to {probs.max():.6f}")
print(f"Final mean: {probs.mean():.4f}")
print(submission.iloc[:3, :8])
'''
)

# Cell 31/code index 31: update identity and log filename.
cell31 = src_of(code_cells[31])
cell31 = cell31.replace(
    'LOGS["experiment_id"] = "exp_038"\nLOGS["experiment_name"] = "pantanal_onnx_fast_noalign"\n',
    'LOGS["experiment_id"] = "exp_042"\nLOGS["experiment_name"] = "v18_texture_graft_no_file_scale"\n',
)
cell31 = cell31.replace(
    'LOGS["temperature"] = CFG["temperature"]\n',
    'LOGS["temperature"] = CFG["temperature"]\nLOGS["file_level_top_k"] = int(CFG.get("file_level_top_k", 0))\n',
)
cell31 = cell31.replace(
    'with open("/kaggle/working/exp_038_pantanal_onnx_fast_noalign_logs.json", "w") as f:\n        json.dump(LOGS, f, indent=2, default=str)\n    print("Saved /kaggle/working/exp_038_pantanal_onnx_fast_noalign_logs.json")\n',
    'with open("/kaggle/working/exp_042_v18_texture_graft_no_file_scale_logs.json", "w") as f:\n        json.dump(LOGS, f, indent=2, default=str)\n    print("Saved /kaggle/working/exp_042_v18_texture_graft_no_file_scale_logs.json")\n',
)
cell31 = cell31.replace(
    "    print(f\"ONNX->TF alignment: {LOGS['onnx_tf_alignment']}\")\n",
    "    print(f\"ONNX->TF alignment: {LOGS['onnx_tf_alignment']}\")\n    print(f\"Texture graft: {LOGS.get('texture_graft', {})}\")\n",
)
code_cells[31]["source"] = to_source(cell31)

for cell in cells:
    if cell.get("cell_type") == "code":
        cell["execution_count"] = None
        cell["outputs"] = []

DST.parent.mkdir(parents=True, exist_ok=True)
DST.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
print(DST)
