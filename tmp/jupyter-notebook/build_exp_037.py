import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "references/private-notebooks/pantanal-distill-birdclef2026-onnx-0.93.ipynb"
DST = ROOT / "notebooks/kaggle_submission_exp_037_pantanal_onnx_tf_aligned.ipynb"


def src_of(cell):
    return "".join(cell.get("source", []))


def to_source(text):
    if not text.endswith("\n"):
        text += "\n"
    return text.splitlines(True)


nb = json.loads(SRC.read_text())
cells = nb["cells"]
code_cells = [c for c in cells if c.get("cell_type") == "code"]

# Cell 0: keep the original TF wheels, add a generic ONNX Runtime wheel install.
code_cells[0]["source"] = to_source(
    """# Cell 0 — Install ONNX Runtime + TF 2.20
from pathlib import Path
import subprocess
import sys

INPUT_ROOT = Path("/kaggle/input")
PERCH_ONNX_HINT = "perch"  # set to None if Kaggle input auto-resolution is too strict
REQUIRE_ONNX_PERCH = True
ONNX_INTRA_OP_THREADS = 4
ENABLE_ONNX_TF_ALIGNMENT = True

def _matches_hint(path, hint):
    return hint is None or hint.lower() in str(path).lower()

def _find_input_file(pattern, hint=None, required=False):
    candidates = []
    if INPUT_ROOT.exists():
        candidates = [p for p in INPUT_ROOT.rglob(pattern) if _matches_hint(p, hint)]
        if not candidates and hint is not None:
            candidates = list(INPUT_ROOT.rglob(pattern))
    candidates = sorted(set(candidates))
    if not candidates:
        if required:
            raise FileNotFoundError(f"Could not find {pattern!r} under /kaggle/input")
        return None
    return candidates[0]

onnxruntime_wheel = _find_input_file("onnxruntime-*.whl", PERCH_ONNX_HINT, required=REQUIRE_ONNX_PERCH)
if onnxruntime_wheel is not None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", str(onnxruntime_wheel)], check=True)
    print("Installed ONNX Runtime:", onnxruntime_wheel)

# Original TensorFlow wheels required by the reference notebook.
!pip install -q --no-deps /kaggle/input/notebooks/ashok205/tf-wheels/tf_wheels/tensorboard-2.20.0-py3-none-any.whl
!pip install -q --no-deps /kaggle/input/notebooks/ashok205/tf-wheels/tf_wheels/tensorflow-2.20.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
"""
)

# Cell 2: import ONNX Runtime, preserving all original config and no added seed.
cell2 = src_of(code_cells[2])
cell2 = cell2.replace(
    "import tensorflow as tf\n",
    "import tensorflow as tf\ntry:\n    import onnxruntime as ort\n    _ONNX_AVAILABLE = True\nexcept ImportError:\n    ort = None\n    _ONNX_AVAILABLE = False\n",
)
cell2 = cell2.replace(
    'print("TensorFlow:", tf.__version__)\nprint("PyTorch:", torch.__version__)\n',
    'print("TensorFlow:", tf.__version__)\nprint("PyTorch:", torch.__version__)\nprint("onnxruntime available:", _ONNX_AVAILABLE)\n',
)
code_cells[2]["source"] = to_source(cell2)

# Cell 11: initialize ONNX session, keep original TF labels/mapping, but avoid
# loading the heavy TF SavedModel when ONNX is required and active.
cell11 = src_of(code_cells[11])
cell11 = cell11.replace(
    '''# Cell 3 — Load Perch, mapping, and selective frog proxies
BEST = CFG["best_fusion"]
birdclassifier = tf.saved_model.load(str(MODEL_DIR))
infer_fn = birdclassifier.signatures["serving_default"]
''',
    '''# Cell 3 — Load Perch, mapping, and selective frog proxies
BEST = CFG["best_fusion"]

def resolve_perch_onnx_path():
    candidates = []
    if INPUT_ROOT.exists():
        for p in INPUT_ROOT.rglob("*.onnx"):
            if "perch" not in str(p).lower():
                continue
            if _matches_hint(p, PERCH_ONNX_HINT):
                candidates.append(p)
        if not candidates and PERCH_ONNX_HINT is not None:
            candidates = [p for p in INPUT_ROOT.rglob("*.onnx") if "perch" in str(p).lower()]
    return sorted(set(candidates))[0] if candidates else None

PERCH_ONNX_PATH = resolve_perch_onnx_path()
USE_ONNX_PERCH = bool(_ONNX_AVAILABLE and PERCH_ONNX_PATH is not None and PERCH_ONNX_PATH.exists())
if REQUIRE_ONNX_PERCH and not USE_ONNX_PERCH:
    raise RuntimeError("ONNX Perch is required but `perch_v2.onnx` or `onnxruntime` was not resolved.")

if USE_ONNX_PERCH:
    print("Using ONNX Perch for test inference:", PERCH_ONNX_PATH)
    _so = ort.SessionOptions()
    _so.intra_op_num_threads = int(ONNX_INTRA_OP_THREADS)
    ONNX_SESSION = ort.InferenceSession(str(PERCH_ONNX_PATH), sess_options=_so, providers=["CPUExecutionProvider"])
    ONNX_INPUT_NAME = ONNX_SESSION.get_inputs()[0].name
    ONNX_OUTPUT_MAP = {o.name: i for i, o in enumerate(ONNX_SESSION.get_outputs())}
else:
    ONNX_SESSION = None
    ONNX_INPUT_NAME = None
    ONNX_OUTPUT_MAP = {}

# Keep TensorFlow fallback available only if ONNX is explicitly disabled.
infer_fn = None
if not USE_ONNX_PERCH:
    birdclassifier = tf.saved_model.load(str(MODEL_DIR))
    infer_fn = birdclassifier.signatures["serving_default"]
''',
)
code_cells[11]["source"] = to_source(cell11)

# Cell 14: switch Perch inference to ONNX, but preserve original loop/order.
cell14 = src_of(code_cells[14])
cell14 = cell14.replace(
    '''        outputs = infer_fn(inputs=tf.convert_to_tensor(x))
        logits = outputs["label"].numpy().astype(np.float32, copy=False)
        emb = outputs["embedding"].numpy().astype(np.float32, copy=False)
''',
    '''        if USE_ONNX_PERCH:
            onnx_outs = ONNX_SESSION.run(None, {ONNX_INPUT_NAME: x})
            logits = onnx_outs[ONNX_OUTPUT_MAP["label"]].astype(np.float32, copy=False)
            emb = onnx_outs[ONNX_OUTPUT_MAP["embedding"]].astype(np.float32, copy=False)
        else:
            outputs = infer_fn(inputs=tf.convert_to_tensor(x))
            logits = outputs["label"].numpy().astype(np.float32, copy=False)
            emb = outputs["embedding"].numpy().astype(np.float32, copy=False)
''',
)
cell14 = cell14.replace(
    "        del x, outputs, logits, emb\n",
    "        if USE_ONNX_PERCH:\n            del x, onnx_outs, logits, emb\n        else:\n            del x, outputs, logits, emb\n",
)
code_cells[14]["source"] = to_source(cell14)

# Insert ONNX->TF feature alignment after the full TF perch cache is loaded.
alignment_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": to_source(
        """# Cell 6b — ONNX -> TF alignment on cached full-file Perch rows
ONNX_TF_ALIGNMENT = None

def fit_affine_alignment(tf_arr, onnx_arr, scale_clip=(0.90, 1.10), eps=1e-6):
    tf_arr = tf_arr.astype(np.float32, copy=False)
    onnx_arr = onnx_arr.astype(np.float32, copy=False)
    tf_mean = tf_arr.mean(axis=0)
    onnx_mean = onnx_arr.mean(axis=0)
    tf_std = tf_arr.std(axis=0)
    onnx_std = onnx_arr.std(axis=0)
    scale = tf_std / np.maximum(onnx_std, eps)
    scale = np.where(np.isfinite(scale), scale, 1.0)
    scale = np.clip(scale, scale_clip[0], scale_clip[1]).astype(np.float32)
    bias = (tf_mean - scale * onnx_mean).astype(np.float32)
    return scale, bias

def apply_onnx_tf_alignment(scores, emb):
    if ONNX_TF_ALIGNMENT is None:
        return scores, emb
    scores = scores * ONNX_TF_ALIGNMENT["score_scale"][None, :] + ONNX_TF_ALIGNMENT["score_bias"][None, :]
    emb = emb * ONNX_TF_ALIGNMENT["emb_scale"][None, :] + ONNX_TF_ALIGNMENT["emb_bias"][None, :]
    return scores.astype(np.float32, copy=False), emb.astype(np.float32, copy=False)

if USE_ONNX_PERCH and ENABLE_ONNX_TF_ALIGNMENT:
    align_files = list(dict.fromkeys(meta_full["filename"].astype(str).tolist()))
    align_paths = [BASE / "train_soundscapes" / f for f in align_files]
    missing_align = [str(p) for p in align_paths if not p.exists()]
    if missing_align:
        print("WARNING: missing alignment files, disabling ONNX->TF alignment:", missing_align[:5])
    else:
        print(f"Fitting ONNX->TF alignment on {len(align_paths)} cached full files...")
        meta_on, scores_on, emb_on = infer_perch_with_embeddings(
            align_paths,
            batch_files=CFG["batch_files"],
            verbose=False,
            proxy_reduce=CFG["proxy_reduce"],
        )
        on_pos = pd.Series(np.arange(len(meta_on)), index=meta_on["row_id"].astype(str))
        mapped = meta_full["row_id"].astype(str).map(on_pos)
        keep = mapped.notna().to_numpy()
        on_idx = mapped[keep].astype(int).to_numpy()

        if keep.sum() < max(24, int(0.8 * len(meta_full))):
            print(f"WARNING: only aligned {keep.sum()} / {len(meta_full)} rows; disabling alignment.")
        else:
            score_scale, score_bias = fit_affine_alignment(
                scores_full_raw[keep], scores_on[on_idx], scale_clip=(0.90, 1.10)
            )
            emb_scale, emb_bias = fit_affine_alignment(
                emb_full[keep], emb_on[on_idx], scale_clip=(0.95, 1.05)
            )
            ONNX_TF_ALIGNMENT = {
                "rows": int(keep.sum()),
                "files": int(len(align_paths)),
                "score_scale": score_scale,
                "score_bias": score_bias,
                "emb_scale": emb_scale,
                "emb_bias": emb_bias,
                "score_mae_before": float(np.abs(scores_full_raw[keep] - scores_on[on_idx]).mean()),
                "emb_mae_before": float(np.abs(emb_full[keep] - emb_on[on_idx]).mean()),
            }
            scores_cal, emb_cal = apply_onnx_tf_alignment(scores_on[on_idx], emb_on[on_idx])
            ONNX_TF_ALIGNMENT["score_mae_after"] = float(np.abs(scores_full_raw[keep] - scores_cal).mean())
            ONNX_TF_ALIGNMENT["emb_mae_after"] = float(np.abs(emb_full[keep] - emb_cal).mean())
            print({
                "onnx_tf_alignment_rows": ONNX_TF_ALIGNMENT["rows"],
                "score_mae_before": ONNX_TF_ALIGNMENT["score_mae_before"],
                "score_mae_after": ONNX_TF_ALIGNMENT["score_mae_after"],
                "emb_mae_before": ONNX_TF_ALIGNMENT["emb_mae_before"],
                "emb_mae_after": ONNX_TF_ALIGNMENT["emb_mae_after"],
            })
else:
    print("ONNX->TF alignment disabled or ONNX inactive.")
"""
    ),
}

# Insert after code cell 15 (full cache load/compute).
insert_after_global_idx = cells.index(code_cells[15])
cells.insert(insert_after_global_idx + 1, alignment_cell)

# Apply alignment to hidden-test ONNX features immediately after test inference.
cell28 = src_of(code_cells[28])
cell28 = cell28.replace(
    '''print("meta_test:", meta_test.shape)
print("scores_test_raw:", scores_test_raw.shape)
print("emb_test:", emb_test.shape)
''',
    '''if USE_ONNX_PERCH and ONNX_TF_ALIGNMENT is not None:
    print("Applying ONNX->TF alignment to test Perch outputs")
    scores_test_raw, emb_test = apply_onnx_tf_alignment(scores_test_raw, emb_test)

print("meta_test:", meta_test.shape)
print("scores_test_raw:", scores_test_raw.shape)
print("emb_test:", emb_test.shape)
''',
)
code_cells[28]["source"] = to_source(cell28)

# Add explicit logging of the ONNX/alignment path.
cell31 = src_of(code_cells[31])
cell31 = cell31.replace(
    'LOGS["temperature"] = CFG["temperature"]\n',
    '''LOGS["experiment_id"] = "exp_037"
LOGS["experiment_name"] = "pantanal_onnx_tf_aligned"
LOGS["source_reference"] = "pantanal-distill-birdclef2026-onnx-0.93.ipynb"
LOGS["onnx_backend_active"] = bool(USE_ONNX_PERCH)
LOGS["onnx_path"] = str(PERCH_ONNX_PATH) if PERCH_ONNX_PATH is not None else None
LOGS["onnx_tf_alignment"] = {
    k: v for k, v in (ONNX_TF_ALIGNMENT or {}).items()
    if not isinstance(v, np.ndarray)
}
LOGS["temperature"] = CFG["temperature"]
''',
)
cell31 = cell31.replace(
    'with open("/kaggle/working/v17_logs.json", "w") as f:\n        json.dump(LOGS, f, indent=2, default=str)\n    print("Saved /kaggle/working/v17_logs.json")\n',
    'with open("/kaggle/working/exp_037_pantanal_onnx_tf_aligned_logs.json", "w") as f:\n        json.dump(LOGS, f, indent=2, default=str)\n    print("Saved /kaggle/working/exp_037_pantanal_onnx_tf_aligned_logs.json")\n',
)
cell31 = cell31.replace(
    '''    print(f"V17 improvements: {LOGS['v17_improvements']}")''',
    '''    print(f"ONNX backend active: {LOGS['onnx_backend_active']}")
    print(f"ONNX->TF alignment rows: {LOGS['onnx_tf_alignment'].get('rows')}")
    print(f"V17 improvements: {LOGS['v17_improvements']}")''',
)
code_cells[31]["source"] = to_source(cell31)

# Clear stale outputs.
for cell in cells:
    if cell.get("cell_type") == "code":
        cell["execution_count"] = None
        cell["outputs"] = []

DST.parent.mkdir(parents=True, exist_ok=True)
DST.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
print(DST)
