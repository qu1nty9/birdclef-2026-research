import json
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "references/private-notebooks/pantanal-distill-birdclef2026-onnx-0.93.ipynb"
DST = ROOT / "notebooks/kaggle_submission_exp_035_pantanal_onnx093_replay.ipynb"


def code_cell(source):
    if not source.endswith("\n"):
        source += "\n"
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(True),
    }


def markdown_cell(source):
    if not source.endswith("\n"):
        source += "\n"
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(True),
    }


def src_of(cell):
    return "".join(cell.get("source", []))


nb = json.loads(SRC.read_text())
cells = [deepcopy(c) for c in nb["cells"]]
code_cells = [c for c in cells if c.get("cell_type") == "code"]

intro = markdown_cell(
    "# exp_035: Pantanal ONNX 0.93 Replay\n\n"
    "This notebook is a controlled replay of `pantanal-distill-birdclef2026-onnx-0.93.ipynb`.\n\n"
    "Key changes versus the reference:\n\n"
    "- The original reference name says ONNX, but its code does not use `onnxruntime`.\n"
    "- This replay activates Perch ONNX strictly by default and logs whether it was actually used.\n"
    "- The modeling recipe remains close to the `0.93` reference: submit-time ProtoSSM/ResidualSSM training, "
    "default `0.5` thresholds, `ENSEMBLE_WEIGHT_PROTO = 0.5`, and V18 fusion/postprocess settings.\n"
    "- Audio loading is prefetched while ONNX runs on CPU to reduce timeout risk.\n"
)

hint_cell = code_cell(
    """# Cell 0a - Kaggle input hints and strict runtime controls
from pathlib import Path
import os
import subprocess
import sys

INPUT_ROOT = Path("/kaggle/input")

# Optional substring filters. Leave as None unless auto-resolution picks a wrong dataset.
COMPETITION_HINT = None          # e.g. "birdclef-2026"
PERCH_ONNX_HINT = None           # e.g. "perch-onnx"
PERCH_MODEL_HINT = None          # only needed for TensorFlow fallback / labels fallback
TF_WHEELS_HINT = None            # e.g. "tf-wheels"

REQUIRE_ONNX_PERCH = True
ONNX_INTRA_OP_THREADS = 4
IO_PREFETCH_WORKERS = 4
SEED = 7177

def _matches_hint(path, hint):
    return hint is None or hint.lower() in str(path).lower()

def _find_input_files(pattern, hint=None):
    if not INPUT_ROOT.exists():
        return []
    hits = [p for p in INPUT_ROOT.rglob(pattern) if _matches_hint(p, hint)]
    if not hits and hint is not None:
        hits = list(INPUT_ROOT.rglob(pattern))
    return sorted(set(hits))

def _install_wheel(pattern, hint=None, required=False):
    candidates = _find_input_files(pattern, hint)
    if not candidates:
        if required:
            raise FileNotFoundError(f"Could not find {pattern!r} under /kaggle/input")
        print(f"Wheel not found for pattern {pattern!r}; assuming package is already available or fallback is disabled.")
        return None
    wheel = candidates[0]
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", str(wheel)], check=True)
    print("Installed wheel:", wheel)
    return wheel

# ONNX Runtime is the intended fast Perch backend.
_install_wheel("onnxruntime-*.whl", PERCH_ONNX_HINT, required=REQUIRE_ONNX_PERCH)

# TensorFlow is kept for compatibility with the reference and fallback path.
for pattern in ("tensorboard-*.whl", "tensorflow-*.whl"):
    _install_wheel(pattern, TF_WHEELS_HINT, required=False)

print({
    "COMPETITION_HINT": COMPETITION_HINT,
    "PERCH_ONNX_HINT": PERCH_ONNX_HINT,
    "PERCH_MODEL_HINT": PERCH_MODEL_HINT,
    "TF_WHEELS_HINT": TF_WHEELS_HINT,
    "REQUIRE_ONNX_PERCH": REQUIRE_ONNX_PERCH,
    "ONNX_INTRA_OP_THREADS": ONNX_INTRA_OP_THREADS,
    "IO_PREFETCH_WORKERS": IO_PREFETCH_WORKERS,
    "SEED": SEED,
})
"""
)

# Replace the original hardcoded pip-install cell with our robust install/hints cell.
code_cells[0]["source"] = hint_cell["source"]

# Patch imports/config cell: robust path resolution, optional ONNX import, reproducibility.
cell2 = src_of(code_cells[2])
cell2 = cell2.replace(
    'from pathlib import Path\n',
    'from pathlib import Path\nimport random\n',
)
cell2 = cell2.replace(
    'import tensorflow as tf\n',
    'try:\n    import tensorflow as tf\n    _TF_AVAILABLE = True\nexcept ImportError:\n    tf = None\n    _TF_AVAILABLE = False\ntry:\n    import onnxruntime as ort\n    _ONNX_AVAILABLE = True\nexcept ImportError:\n    ort = None\n    _ONNX_AVAILABLE = False\n',
)
cell2 = cell2.replace(
    'warnings.filterwarnings("ignore")\ntf.experimental.numpy.experimental_enable_numpy_behavior()\n\n_WALL_START = time.time()\n\nBASE = Path("/kaggle/input/competitions/birdclef-2026")\nMODEL_DIR = Path("/kaggle/input/models/google/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1")\n',
    '''warnings.filterwarnings("ignore")
if _TF_AVAILABLE:
    tf.experimental.numpy.experimental_enable_numpy_behavior()

def seed_everything(seed):
    if seed is None:
        return
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    try:
        if _TF_AVAILABLE:
            tf.random.set_seed(seed)
    except Exception:
        pass

seed_everything(SEED)

_WALL_START = time.time()

def resolve_competition_dir():
    candidates = []
    for p in INPUT_ROOT.rglob("sample_submission.csv"):
        parent = p.parent
        if (parent / "taxonomy.csv").exists():
            if _matches_hint(parent, COMPETITION_HINT):
                candidates.append(parent)
    if not candidates and COMPETITION_HINT is not None:
        for p in INPUT_ROOT.rglob("sample_submission.csv"):
            parent = p.parent
            if (parent / "taxonomy.csv").exists():
                candidates.append(parent)
    if not candidates:
        raise FileNotFoundError("Could not resolve BirdCLEF competition directory under /kaggle/input")
    return sorted(candidates)[0]

def resolve_perch_model_dir(allow_missing=True):
    candidates = []
    for p in INPUT_ROOT.rglob("saved_model.pb"):
        parent = p.parent
        if (parent / "assets" / "labels.csv").exists() and _matches_hint(parent, PERCH_MODEL_HINT):
            candidates.append(parent)
    if not candidates and PERCH_MODEL_HINT is not None:
        for p in INPUT_ROOT.rglob("saved_model.pb"):
            parent = p.parent
            if (parent / "assets" / "labels.csv").exists():
                candidates.append(parent)
    if not candidates:
        if allow_missing:
            return None
        raise FileNotFoundError("Could not find Perch SavedModel under /kaggle/input")
    return sorted(candidates)[0]

def resolve_perch_onnx_path():
    candidates = []
    for p in INPUT_ROOT.rglob("*.onnx"):
        sp = str(p).lower()
        if "perch" not in sp:
            continue
        if _matches_hint(p, PERCH_ONNX_HINT):
            candidates.append(p)
    if not candidates and PERCH_ONNX_HINT is not None:
        for p in INPUT_ROOT.rglob("*.onnx"):
            if "perch" in str(p).lower():
                candidates.append(p)
    return sorted(candidates)[0] if candidates else None

BASE = resolve_competition_dir()
MODEL_DIR = resolve_perch_model_dir(allow_missing=True)
PERCH_ONNX_PATH = resolve_perch_onnx_path()
''',
)
cell2 = cell2.replace(
    'print("TensorFlow:", tf.__version__)\nprint("PyTorch:", torch.__version__)\nprint("Competition dir exists:", BASE.exists())\nprint("Model dir exists:", MODEL_DIR.exists())\n',
    'print("TensorFlow available:", _TF_AVAILABLE, getattr(tf, "__version__", None))\nprint("PyTorch:", torch.__version__)\nprint("onnxruntime available:", _ONNX_AVAILABLE)\nprint("Competition dir:", BASE)\nprint("Model dir:", MODEL_DIR)\nprint("Perch ONNX path:", PERCH_ONNX_PATH)\n',
)
code_cells[2]["source"] = cell2.splitlines(True)

# Patch Perch load/mapping cell to activate ONNX strictly and avoid TF SavedModel when ONNX is active.
cell11 = src_of(code_cells[11])
cell11 = cell11.replace(
    '''# Cell 3 — Load Perch, mapping, and selective frog proxies
BEST = CFG["best_fusion"]
birdclassifier = tf.saved_model.load(str(MODEL_DIR))
infer_fn = birdclassifier.signatures["serving_default"]

bc_labels = (
    pd.read_csv(MODEL_DIR / "assets" / "labels.csv")
    .reset_index()
    .rename(columns={"index": "bc_index", "inat2024_fsd50k": "scientific_name"})
)
''',
    '''# Cell 3 — Load Perch, mapping, and selective frog proxies
BEST = CFG["best_fusion"]

USE_ONNX_PERCH = bool(_ONNX_AVAILABLE and PERCH_ONNX_PATH is not None and PERCH_ONNX_PATH.exists())
ONNX_SESSION = None
ONNX_INPUT_NAME = None
ONNX_OUTPUT_MAP = {}

if USE_ONNX_PERCH:
    print("Using ONNX Perch:", PERCH_ONNX_PATH)
    _so = ort.SessionOptions()
    _so.intra_op_num_threads = int(ONNX_INTRA_OP_THREADS)
    ONNX_SESSION = ort.InferenceSession(str(PERCH_ONNX_PATH), sess_options=_so, providers=["CPUExecutionProvider"])
    ONNX_INPUT_NAME = ONNX_SESSION.get_inputs()[0].name
    ONNX_OUTPUT_MAP = {o.name: i for i, o in enumerate(ONNX_SESSION.get_outputs())}
else:
    if REQUIRE_ONNX_PERCH:
        raise RuntimeError("ONNX Perch is required but was not resolved/importable.")
    if not _TF_AVAILABLE:
        raise ImportError("TensorFlow fallback requested, but TensorFlow is not importable.")
    if MODEL_DIR is None:
        raise FileNotFoundError("TensorFlow fallback requires a Perch SavedModel dataset.")
    print("Using TensorFlow Perch fallback:", MODEL_DIR)

infer_fn = None
if not USE_ONNX_PERCH:
    birdclassifier = tf.saved_model.load(str(MODEL_DIR))
    infer_fn = birdclassifier.signatures["serving_default"]

labels_path = None
if PERCH_ONNX_PATH is not None and (PERCH_ONNX_PATH.with_name("labels.csv")).exists():
    labels_path = PERCH_ONNX_PATH.with_name("labels.csv")
elif MODEL_DIR is not None and (MODEL_DIR / "assets" / "labels.csv").exists():
    labels_path = MODEL_DIR / "assets" / "labels.csv"
else:
    raise FileNotFoundError("Could not resolve Perch labels.csv from ONNX or TensorFlow dataset.")

bc_labels = (
    pd.read_csv(labels_path)
    .reset_index()
    .rename(columns={"index": "bc_index", "inat2024_fsd50k": "scientific_name"})
)
print("Perch labels:", labels_path, bc_labels.shape)
''',
)
code_cells[11]["source"] = cell11.splitlines(True)

# Replace Perch inference cell with ONNX + audio prefetch implementation.
code_cells[14]["source"] = """# Cell 5 — Perch inference with embeddings + selective proxies
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
        onnx_outs = ONNX_SESSION.run(None, {ONNX_INPUT_NAME: x})
        logits = onnx_outs[ONNX_OUTPUT_MAP["label"]].astype(np.float32, copy=False)
        emb = onnx_outs[ONNX_OUTPUT_MAP["embedding"]].astype(np.float32, copy=False)
        return logits, emb
    outputs = infer_fn(inputs=tf.convert_to_tensor(x))
    logits = outputs["label"].numpy().astype(np.float32, copy=False)
    emb = outputs["embedding"].numpy().astype(np.float32, copy=False)
    return logits, emb

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
""".splitlines(True)

# Patch final logging.
cell31 = src_of(code_cells[31])
cell31 = cell31.replace(
    'LOGS["temperature"] = CFG["temperature"]\n',
    'LOGS["experiment_id"] = "exp_035"\nLOGS["experiment_name"] = "pantanal_onnx093_replay"\nLOGS["source_reference"] = "pantanal-distill-birdclef2026-onnx-0.93.ipynb"\nLOGS["onnx_backend_active"] = bool(USE_ONNX_PERCH)\nLOGS["onnx_path"] = str(PERCH_ONNX_PATH) if PERCH_ONNX_PATH is not None else None\nLOGS["model_dir"] = str(MODEL_DIR) if MODEL_DIR is not None else None\nLOGS["seed"] = SEED\nLOGS["io_prefetch_workers"] = int(IO_PREFETCH_WORKERS)\nLOGS["temperature"] = CFG["temperature"]\n',
)
cell31 = cell31.replace(
    'with open("/kaggle/working/v17_logs.json", "w") as f:\n        json.dump(LOGS, f, indent=2, default=str)\n    print("Saved /kaggle/working/v17_logs.json")\n',
    'with open("/kaggle/working/exp_035_pantanal_onnx093_replay_logs.json", "w") as f:\n        json.dump(LOGS, f, indent=2, default=str)\n    print("Saved /kaggle/working/exp_035_pantanal_onnx093_replay_logs.json")\n',
)
cell31 = cell31.replace(
    '''    print(f"V17 improvements: {LOGS['v17_improvements']}")''',
    '''    print(f"ONNX backend active: {LOGS['onnx_backend_active']}")
    print(f"V17 improvements: {LOGS['v17_improvements']}")''',
)
code_cells[31]["source"] = cell31.splitlines(True)

# Add a small note before the mode switch.
cells.insert(0, intro)

nb["cells"] = cells
nb["metadata"].setdefault("language_info", {})["name"] = "python"
for cell in nb["cells"]:
    if cell.get("cell_type") == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
DST.parent.mkdir(parents=True, exist_ok=True)
DST.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
print(DST)
