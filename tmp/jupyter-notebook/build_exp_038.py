import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "references/private-notebooks/pantanal-distill-birdclef2026-onnx-0.93.ipynb"
DST = ROOT / "notebooks/kaggle_submission_exp_038_pantanal_onnx_fast_noalign.ipynb"


def src_of(cell):
    return "".join(cell.get("source", []))


def to_source(text):
    if not text.endswith("\n"):
        text += "\n"
    return text.splitlines(True)


def replace_between(text, start_marker, end_marker, replacement):
    start = text.index(start_marker)
    end = text.index(end_marker, start)
    return text[:start] + replacement.rstrip() + "\n\n" + text[end:]


nb = json.loads(SRC.read_text())
cells = nb["cells"]
code_cells = [c for c in cells if c.get("cell_type") == "code"]

# Cell 0: add ONNX Runtime while preserving the reference TF wheel setup.
code_cells[0]["source"] = to_source(
    """# Cell 0 — Install ONNX Runtime + TF 2.20
from pathlib import Path
import subprocess
import sys

INPUT_ROOT = Path("/kaggle/input")

# Keep this broad: the public ONNX dataset usually contains both perch_v2.onnx
# and onnxruntime-*.whl in a path with "perch" in its name.
PERCH_ONNX_HINT = "perch"
REQUIRE_ONNX_PERCH = True
ONNX_INTRA_OP_THREADS = 4
IO_PREFETCH_WORKERS = 4
FAST_TTA_MAX_BATCH_SIZE = 512

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

# Cell 2: import ONNX Runtime and print backend availability.
cell2 = src_of(code_cells[2])
cell2 = cell2.replace(
    "import tensorflow as tf\n",
    "import tensorflow as tf\ntry:\n    import onnxruntime as ort\n    _ONNX_AVAILABLE = True\nexcept ImportError:\n    ort = None\n    _ONNX_AVAILABLE = False\n",
)
cell2 = cell2.replace(
    'print("TensorFlow:", tf.__version__)\nprint("PyTorch:", torch.__version__)\n',
    'print("TensorFlow:", tf.__version__)\nprint("PyTorch:", torch.__version__)\nprint("onnxruntime available:", _ONNX_AVAILABLE)\n',
)
cell2 = cell2.replace(
    'CFG["full_cache_work_dir"].mkdir(parents=True, exist_ok=True)\n',
    '''def resolve_full_cache_input_dir(default_dir):
    default_dir = Path(default_dir)
    required = ("full_perch_meta.parquet", "full_perch_arrays.npz")
    if all((default_dir / name).exists() for name in required):
        return default_dir
    if INPUT_ROOT.exists():
        for meta_path in sorted(INPUT_ROOT.rglob("full_perch_meta.parquet")):
            candidate = meta_path.parent
            if all((candidate / name).exists() for name in required):
                return candidate
    return default_dir

CFG["full_cache_input_dir"] = resolve_full_cache_input_dir(CFG["full_cache_input_dir"])
CFG["full_cache_work_dir"].mkdir(parents=True, exist_ok=True)
print("Resolved full_cache_input_dir:", CFG["full_cache_input_dir"])
''',
)
code_cells[2]["source"] = to_source(cell2)

# Cell 11: initialize ONNX Perch, keep the TF model dataset only for labels/fallback.
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
    ONNX_INPUT_SHAPE = list(ONNX_SESSION.get_inputs()[0].shape)
    ONNX_INPUT_RANK = len(ONNX_INPUT_SHAPE)
    ONNX_OUTPUT_MAP = {o.name: i for i, o in enumerate(ONNX_SESSION.get_outputs())}
    print("ONNX input:", ONNX_INPUT_NAME, ONNX_INPUT_SHAPE)
    print("ONNX outputs:", ONNX_OUTPUT_MAP)
else:
    ONNX_SESSION = None
    ONNX_INPUT_NAME = None
    ONNX_INPUT_SHAPE = None
    ONNX_INPUT_RANK = None
    ONNX_OUTPUT_MAP = {}

# TensorFlow SavedModel is heavy; do not load it when ONNX is active.
infer_fn = None
if not USE_ONNX_PERCH:
    birdclassifier = tf.saved_model.load(str(MODEL_DIR))
    infer_fn = birdclassifier.signatures["serving_default"]
''',
)
code_cells[11]["source"] = to_source(cell11)

# Cell 13: batch all temporal-shift TTA passes. This is numerically equivalent
# for eval-mode PyTorch modules, but avoids several separate model calls.
cell13 = src_of(code_cells[13])
fast_tta = '''def temporal_shift_tta(emb_files, logits_files, model, site_ids, hours, shifts=[0, 1, -1], max_batch_size=None):
    """TTA by circular-shifting the 12-window sequence, batched for speed."""
    n_files = emb_files.shape[0]
    n_shifts = len(shifts)
    if n_shifts == 0:
        return np.zeros((n_files, emb_files.shape[1], logits_files.shape[2]), dtype=np.float32)

    e_list, l_list = [], []
    for shift in shifts:
        if shift == 0:
            e_list.append(emb_files)
            l_list.append(logits_files)
        else:
            e_list.append(np.roll(emb_files, shift, axis=1))
            l_list.append(np.roll(logits_files, shift, axis=1))

    e_batch = np.concatenate(e_list, axis=0)
    l_batch = np.concatenate(l_list, axis=0)
    site_batch = np.tile(site_ids, n_shifts)
    hour_batch = np.tile(hours, n_shifts)

    if max_batch_size is None:
        max_batch_size = int(FAST_TTA_MAX_BATCH_SIZE)

    model.eval()
    pred_chunks = []
    with torch.no_grad():
        total = e_batch.shape[0]
        for start_idx in range(0, total, max_batch_size):
            end_idx = min(start_idx + max_batch_size, total)
            out, _, _ = model(
                torch.tensor(e_batch[start_idx:end_idx], dtype=torch.float32),
                torch.tensor(l_batch[start_idx:end_idx], dtype=torch.float32),
                site_ids=torch.tensor(site_batch[start_idx:end_idx], dtype=torch.long),
                hours=torch.tensor(hour_batch[start_idx:end_idx], dtype=torch.long),
            )
            pred_chunks.append(out.numpy())

    pred_batch = np.concatenate(pred_chunks, axis=0)
    pred_batch = pred_batch.reshape(n_shifts, n_files, pred_batch.shape[1], pred_batch.shape[2])

    all_preds = []
    for i, shift in enumerate(shifts):
        pred_i = pred_batch[i]
        if shift != 0:
            pred_i = np.roll(pred_i, -shift, axis=1)
        all_preds.append(pred_i)
    return np.mean(all_preds, axis=0).astype(np.float32, copy=False)
'''
cell13 = replace_between(cell13, "def temporal_shift_tta", "# V17: Post-processing utilities", fast_tta)
code_cells[13]["source"] = to_source(cell13)

# Cell 14: add audio prefetch and ONNX Perch inference. This mirrors the fast
# public baseline but keeps the Pantanal scoring/postprocess recipe unchanged.
code_cells[14]["source"] = to_source(
    """# Cell 5 — Perch inference with embeddings + selective proxies
import concurrent.futures

def read_soundscape_60s(path):
    y, sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if sr != SR:
        raise ValueError(f"Unexpected sample rate {sr} in {path}; expected {SR}")
    if len(y) < FILE_SAMPLES:
        y = np.pad(y, (0, FILE_SAMPLES - len(y)))
    elif len(y) > FILE_SAMPLES:
        y = y[:FILE_SAMPLES]
    return y

def _run_perch_batch(x):
    if USE_ONNX_PERCH:
        onnx_outs = ONNX_SESSION.run(None, {ONNX_INPUT_NAME: _format_onnx_input(x)})
        logits = onnx_outs[ONNX_OUTPUT_MAP["label"]].astype(np.float32, copy=False)
        emb = onnx_outs[ONNX_OUTPUT_MAP["embedding"]].astype(np.float32, copy=False)
        return logits, emb
    outputs = infer_fn(inputs=tf.convert_to_tensor(x))
    logits = outputs["label"].numpy().astype(np.float32, copy=False)
    emb = outputs["embedding"].numpy().astype(np.float32, copy=False)
    return logits, emb

def _dim_matches(dim, value):
    try:
        return int(dim) == int(value)
    except Exception:
        return False

def _format_onnx_input(x):
    # Adapt waveform batches to ONNX exports that require rank-2/3/4 input.
    x = x.astype(np.float32, copy=False)
    if ONNX_INPUT_RANK == 2:
        return x
    if ONNX_INPUT_RANK == 3:
        shape = list(ONNX_INPUT_SHAPE or [])
        tail = shape[1:] if len(shape) == 3 else []
        if len(tail) == 2 and _dim_matches(tail[0], WINDOW_SAMPLES):
            return x[:, :, None]
        if len(tail) == 2 and _dim_matches(tail[1], WINDOW_SAMPLES):
            return x[:, None, :]
        return x[:, :, None]
    if ONNX_INPUT_RANK == 4:
        shape = list(ONNX_INPUT_SHAPE or [])
        tail = shape[1:] if len(shape) == 4 else []
        if len(tail) == 3 and _dim_matches(tail[0], WINDOW_SAMPLES):
            return x[:, :, None, None]
        if len(tail) == 3 and _dim_matches(tail[1], WINDOW_SAMPLES):
            return x[:, None, :, None]
        if len(tail) == 3 and _dim_matches(tail[2], WINDOW_SAMPLES):
            return x[:, None, None, :]
        # The public Perch ONNX often reports a dynamic 4D input but starts by
        # reshaping back to waveform. Preserve row-major waveform order.
        return x[:, :, None, None]
    raise ValueError(f"Unsupported ONNX input rank: {ONNX_INPUT_RANK}, shape={ONNX_INPUT_SHAPE}")

def infer_perch_with_embeddings(paths, batch_files=16, verbose=True, proxy_reduce="max"):
    paths = [Path(p) for p in paths]
    n_files = len(paths)
    n_rows = n_files * N_WINDOWS

    row_ids = np.empty(n_rows, dtype=object)
    filenames = np.empty(n_rows, dtype=object)
    sites = np.empty(n_rows, dtype=object)
    hours = np.empty(n_rows, dtype=np.int16)

    scores = np.zeros((n_rows, N_CLASSES), dtype=np.float32)
    embeddings = np.zeros((n_rows, 1536), dtype=np.float32)

    write_row = 0
    iterator = range(0, n_files, batch_files)
    if verbose:
        iterator = tqdm(iterator, total=(n_files + batch_files - 1) // batch_files, desc="Perch batches")

    workers = max(1, int(IO_PREFETCH_WORKERS))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as io_executor:
        next_paths = paths[:batch_files]
        future_audio = [io_executor.submit(read_soundscape_60s, p) for p in next_paths]

        for start in iterator:
            batch_paths = next_paths
            batch_n = len(batch_paths)
            batch_audio = [f.result() for f in future_audio]

            next_start = start + batch_files
            if next_start < n_files:
                next_paths = paths[next_start:next_start + batch_files]
                future_audio = [io_executor.submit(read_soundscape_60s, p) for p in next_paths]

            x = np.empty((batch_n * N_WINDOWS, WINDOW_SAMPLES), dtype=np.float32)
            batch_row_start = write_row
            x_pos = 0

            for i, path in enumerate(batch_paths):
                y = batch_audio[i]
                x[x_pos:x_pos + N_WINDOWS] = y.reshape(N_WINDOWS, WINDOW_SAMPLES)

                meta = parse_soundscape_filename(path.name)
                stem = path.stem

                row_ids[write_row:write_row + N_WINDOWS] = [f"{stem}_{t}" for t in range(5, 65, 5)]
                filenames[write_row:write_row + N_WINDOWS] = path.name
                sites[write_row:write_row + N_WINDOWS] = meta["site"]
                hours[write_row:write_row + N_WINDOWS] = int(meta["hour_utc"])

                x_pos += N_WINDOWS
                write_row += N_WINDOWS

            logits, emb = _run_perch_batch(x)

            scores[batch_row_start:write_row, MAPPED_POS] = logits[:, MAPPED_BC_INDICES]
            embeddings[batch_row_start:write_row] = emb

            for pos, bc_idx_arr in selected_proxy_pos_to_bc.items():
                sub = logits[:, bc_idx_arr]
                if proxy_reduce == "max":
                    proxy_score = sub.max(axis=1)
                elif proxy_reduce == "mean":
                    proxy_score = sub.mean(axis=1)
                else:
                    raise ValueError("proxy_reduce must be 'max' or 'mean'")
                scores[batch_row_start:write_row, pos] = proxy_score.astype(np.float32)

            del x, logits, emb, batch_audio
            gc.collect()

    meta_df = pd.DataFrame({
        "row_id": row_ids,
        "filename": filenames,
        "site": sites,
        "hour_utc": hours,
    })

    return meta_df, scores, embeddings
"""
)

# Cell 31: log the exact fast path.
cell31 = src_of(code_cells[31])
cell31 = cell31.replace(
    'LOGS["temperature"] = CFG["temperature"]\n',
    '''LOGS["experiment_id"] = "exp_038"
LOGS["experiment_name"] = "pantanal_onnx_fast_noalign"
LOGS["source_reference"] = "pantanal-distill-birdclef2026-onnx-0.93.ipynb"
LOGS["onnx_backend_active"] = bool(USE_ONNX_PERCH)
LOGS["onnx_path"] = str(PERCH_ONNX_PATH) if PERCH_ONNX_PATH is not None else None
LOGS["onnx_intra_op_threads"] = int(ONNX_INTRA_OP_THREADS)
LOGS["io_prefetch_workers"] = int(IO_PREFETCH_WORKERS)
LOGS["fast_tta_max_batch_size"] = int(FAST_TTA_MAX_BATCH_SIZE)
LOGS["onnx_tf_alignment"] = "disabled_for_submit_runtime"
LOGS["temperature"] = CFG["temperature"]
''',
)
cell31 = cell31.replace(
    'with open("/kaggle/working/v17_logs.json", "w") as f:\n        json.dump(LOGS, f, indent=2, default=str)\n    print("Saved /kaggle/working/v17_logs.json")\n',
    'with open("/kaggle/working/exp_038_pantanal_onnx_fast_noalign_logs.json", "w") as f:\n        json.dump(LOGS, f, indent=2, default=str)\n    print("Saved /kaggle/working/exp_038_pantanal_onnx_fast_noalign_logs.json")\n',
)
cell31 = cell31.replace(
    '''    print(f"V17 improvements: {LOGS['v17_improvements']}")''',
    '''    print(f"ONNX backend active: {LOGS['onnx_backend_active']}")
    print(f"ONNX->TF alignment: {LOGS['onnx_tf_alignment']}")
    print(f"V17 improvements: {LOGS['v17_improvements']}")''',
)
code_cells[31]["source"] = to_source(cell31)

for cell in cells:
    if cell.get("cell_type") == "code":
        cell["execution_count"] = None
        cell["outputs"] = []

DST.parent.mkdir(parents=True, exist_ok=True)
DST.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
print(DST)
