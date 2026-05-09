from __future__ import annotations

import json
import textwrap
from pathlib import Path


ROOT = Path("/Users/yaroslav/Documents/Kaggle/BirdCLEF_2026")
NOTEBOOKS = ROOT / "notebooks"


def lines(text: str) -> list[str]:
    text = textwrap.dedent(text).strip("\n")
    return [f"{line}\n" for line in text.splitlines()]


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise RuntimeError(f"Could not find target for {label}")
    return text.replace(old, new, 1)


src_path = NOTEBOOKS / "kaggle_submission_exp_029b_exp015d_runtime_port.ipynb"
dst_path = NOTEBOOKS / "kaggle_submission_exp_029c_exp015d_onnx_first_runtime_port.ipynb"

nb = json.loads(src_path.read_text())

nb["cells"][3]["source"] = lines(
    """
    # BirdCLEF+ 2026 -- ProtoSSM v5: V18 ONNX-First Runtime Port (Exp 029c)
    """
)

nb["cells"][1]["source"] = lines(
    """
    # Cell 0b — Kaggle input hints
    TF_WHEELS_HINT = None  # only needed for TensorFlow fallback
    COMPETITION_HINT = None  # e.g. "birdclef-2026"
    PERCH_MODEL_HINT = None  # optional, only needed for TensorFlow fallback
    PERCH_CACHE_HINT = None  # not required for thin-submit; can stay None
    PERCH_ONNX_HINT = None  # e.g. "perch-onnx"
    ARTIFACTS_HINT = None  # e.g. "birdclef-exp015c-v18-artifacts"

    ENABLE_ONNX_PERCH = True
    REQUIRE_ONNX_PERCH = True
    ONNX_INTRA_OP_THREADS = 4
    IO_PREFETCH_WORKERS = 4
    TORCH_INTRA_OP_THREADS = 4
    ENABLE_VECTORIZED_MLP_PROBES = True
    ENABLE_BATCHED_TTA = True
    TTA_MAX_BATCH_FILES = 512

    print({
        "TF_WHEELS_HINT": TF_WHEELS_HINT,
        "COMPETITION_HINT": COMPETITION_HINT,
        "PERCH_MODEL_HINT": PERCH_MODEL_HINT,
        "PERCH_CACHE_HINT": PERCH_CACHE_HINT,
        "PERCH_ONNX_HINT": PERCH_ONNX_HINT,
        "ARTIFACTS_HINT": ARTIFACTS_HINT,
        "ENABLE_ONNX_PERCH": ENABLE_ONNX_PERCH,
        "REQUIRE_ONNX_PERCH": REQUIRE_ONNX_PERCH,
        "ONNX_INTRA_OP_THREADS": ONNX_INTRA_OP_THREADS,
        "IO_PREFETCH_WORKERS": IO_PREFETCH_WORKERS,
        "TORCH_INTRA_OP_THREADS": TORCH_INTRA_OP_THREADS,
        "ENABLE_VECTORIZED_MLP_PROBES": ENABLE_VECTORIZED_MLP_PROBES,
        "ENABLE_BATCHED_TTA": ENABLE_BATCHED_TTA,
        "TTA_MAX_BATCH_FILES": TTA_MAX_BATCH_FILES,
    })
    """
)

nb["cells"][2]["source"] = lines(
    """
    # Cell 0 — Install ONNX Runtime or TF fallback wheels
    from pathlib import Path
    import subprocess
    import sys

    INPUT_ROOT = Path("/kaggle/input")

    def find_single_file(pattern, hint=None, required=True):
        candidates = []
        for p in INPUT_ROOT.rglob(pattern):
            sp = str(p).lower()
            if hint is None or hint.lower() in sp:
                candidates.append(p)
        if not candidates and hint is not None:
            for p in INPUT_ROOT.rglob(pattern):
                candidates.append(p)
        candidates = sorted(set(candidates))
        if not candidates:
            if required:
                raise FileNotFoundError(f"Could not find {pattern!r} under /kaggle/input")
            return None
        return candidates[0]

    onnx_whl = find_single_file("onnxruntime-*.whl", PERCH_ONNX_HINT, required=False)
    onnx_model = find_single_file("*.onnx", PERCH_ONNX_HINT, required=False)
    use_onnx_bootstrap = bool(ENABLE_ONNX_PERCH and onnx_whl is not None and onnx_model is not None)

    print({
        "onnxruntime_wheel": str(onnx_whl) if onnx_whl is not None else None,
        "onnx_model": str(onnx_model) if onnx_model is not None else None,
        "use_onnx_bootstrap": use_onnx_bootstrap,
    })

    if use_onnx_bootstrap:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", str(onnx_whl)], check=True)
        print("Installed ONNX Runtime wheel:", onnx_whl)
    else:
        if REQUIRE_ONNX_PERCH:
            raise FileNotFoundError(
                "ONNX Perch was required, but the notebook could not resolve both "
                "`onnxruntime-*.whl` and `*.onnx` under /kaggle/input. "
                "Attach the ONNX dataset and set `PERCH_ONNX_HINT` to match it."
            )

        tb_whl = find_single_file("tensorboard-2.20.0-*.whl", TF_WHEELS_HINT)
        tf_whl = find_single_file("tensorflow-2.20.0-*.whl", TF_WHEELS_HINT)
        print("TensorBoard wheel:", tb_whl)
        print("TensorFlow wheel:", tf_whl)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", str(tb_whl)], check=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", str(tf_whl)], check=True)
    """
)

cell6 = "".join(nb["cells"][6]["source"])
cell6 = replace_once(
    cell6,
    "import numpy as np\nimport pandas as pd\nimport soundfile as sf\nimport tensorflow as tf\n\nimport torch\n",
    """import numpy as np\nimport pandas as pd\nimport soundfile as sf\n\ntry:\n    import tensorflow as tf\n    _TF_AVAILABLE = True\nexcept ImportError:\n    tf = None\n    _TF_AVAILABLE = False\n\nimport torch\n""",
    "cell6 optional tf import",
)
cell6 = replace_once(
    cell6,
    'warnings.filterwarnings("ignore")\ntf.experimental.numpy.experimental_enable_numpy_behavior()\n',
    """warnings.filterwarnings("ignore")\nif _TF_AVAILABLE:\n    tf.experimental.numpy.experimental_enable_numpy_behavior()\n""",
    "cell6 tf guard",
)
cell6 = replace_once(
    cell6,
    """def resolve_perch_model_dir():
    found = []
    for p in INPUT_ROOT.rglob("saved_model.pb"):
        parent = p.parent
        if (parent / "assets" / "labels.csv").exists():
            if PERCH_MODEL_HINT is None or PERCH_MODEL_HINT.lower() in str(parent).lower():
                found.append(parent)
    if not found:
        raise FileNotFoundError("Could not find perch_v2_cpu SavedModel under /kaggle/input")
    return sorted(found)[0]

def resolve_perch_cache_dir():
    found = []
    for p in INPUT_ROOT.rglob("full_perch_meta.parquet"):
        parent = p.parent
        if (parent / "full_perch_arrays.npz").exists():
            if PERCH_CACHE_HINT is None or PERCH_CACHE_HINT.lower() in str(parent).lower():
                found.append(parent)
    return sorted(found)[0] if found else None

def resolve_perch_onnx_path():
    found = []
    for p in INPUT_ROOT.rglob("*.onnx"):
        if "perch" not in p.name.lower() and "perch" not in str(p.parent).lower():
            continue
        if PERCH_ONNX_HINT is None or PERCH_ONNX_HINT.lower() in str(p).lower():
            found.append(p)
    return sorted(found)[0] if found else None

BASE = resolve_competition_dir()
MODEL_DIR = resolve_perch_model_dir()
PERCH_CACHE_DIR = resolve_perch_cache_dir()
PERCH_ONNX_PATH = resolve_perch_onnx_path()
print("Competition dir:", BASE)
print("Perch model dir:", MODEL_DIR)
print("Perch cache dir:", PERCH_CACHE_DIR)
print("Perch ONNX path:", PERCH_ONNX_PATH)
print("onnxruntime available:", _ORT_AVAILABLE)
""",
    """def resolve_perch_model_dir(allow_missing=False):
    found = []
    for p in INPUT_ROOT.rglob("saved_model.pb"):
        parent = p.parent
        if (parent / "assets" / "labels.csv").exists():
            if PERCH_MODEL_HINT is None or PERCH_MODEL_HINT.lower() in str(parent).lower():
                found.append(parent)
    if not found:
        if allow_missing:
            return None
        raise FileNotFoundError("Could not find perch_v2_cpu SavedModel under /kaggle/input")
    return sorted(found)[0]

def resolve_perch_cache_dir():
    found = []
    for p in INPUT_ROOT.rglob("full_perch_meta.parquet"):
        parent = p.parent
        if (parent / "full_perch_arrays.npz").exists():
            if PERCH_CACHE_HINT is None or PERCH_CACHE_HINT.lower() in str(parent).lower():
                found.append(parent)
    return sorted(found)[0] if found else None

def resolve_perch_onnx_path():
    found = []
    for p in INPUT_ROOT.rglob("*.onnx"):
        if "perch" not in p.name.lower() and "perch" not in str(p.parent).lower():
            continue
        if PERCH_ONNX_HINT is None or PERCH_ONNX_HINT.lower() in str(p).lower():
            found.append(p)
    return sorted(found)[0] if found else None

def resolve_perch_labels_path(perch_onnx_path, model_dir):
    if perch_onnx_path is not None:
        onnx_labels = Path(perch_onnx_path).with_name("labels.csv")
        if onnx_labels.exists():
            return onnx_labels
    if model_dir is not None:
        tf_labels = Path(model_dir) / "assets" / "labels.csv"
        if tf_labels.exists():
            return tf_labels
    raise FileNotFoundError("Could not resolve Perch labels.csv from either ONNX or TensorFlow inputs")

BASE = resolve_competition_dir()
PERCH_CACHE_DIR = resolve_perch_cache_dir()
PERCH_ONNX_PATH = resolve_perch_onnx_path()
MODEL_DIR = None
if not (bool(ENABLE_ONNX_PERCH) and PERCH_ONNX_PATH is not None):
    MODEL_DIR = resolve_perch_model_dir(allow_missing=bool(ENABLE_ONNX_PERCH))
PERCH_LABELS_PATH = resolve_perch_labels_path(PERCH_ONNX_PATH, MODEL_DIR)
print("Competition dir:", BASE)
print("Perch model dir:", MODEL_DIR)
print("Perch cache dir:", PERCH_CACHE_DIR)
print("Perch ONNX path:", PERCH_ONNX_PATH)
print("Perch labels path:", PERCH_LABELS_PATH)
print("onnxruntime available:", _ORT_AVAILABLE)
print("tensorflow available:", _TF_AVAILABLE)
""",
    "cell6 path resolution",
)
cell6 = replace_once(
    cell6,
    '    "enable_onnx_perch": bool(ENABLE_ONNX_PERCH),\n',
    '    "enable_onnx_perch": bool(ENABLE_ONNX_PERCH),\n    "require_onnx_perch": bool(REQUIRE_ONNX_PERCH),\n',
    "cell6 require onnx cfg",
)
nb["cells"][6]["source"] = lines(cell6)

cell10 = "".join(nb["cells"][10]["source"])
cell10 = replace_once(
    cell10,
    """USE_ONNX_PERCH = bool(
    CFG.get("enable_onnx_perch", True)
    and _ORT_AVAILABLE
    and PERCH_ONNX_PATH is not None
    and Path(PERCH_ONNX_PATH).exists()
)

if USE_ONNX_PERCH:
    _so = ort.SessionOptions()
    _so.intra_op_num_threads = int(CFG.get("onnx_intra_op_threads", 4))
    ONNX_SESSION = ort.InferenceSession(
        str(PERCH_ONNX_PATH),
        sess_options=_so,
        providers=["CPUExecutionProvider"],
    )
    ONNX_INPUT_NAME = ONNX_SESSION.get_inputs()[0].name
    ONNX_LABEL_INDEX = resolve_onnx_output_index(ONNX_SESSION, ["label", "logits", "scores"])
    ONNX_EMB_INDEX = resolve_onnx_output_index(ONNX_SESSION, ["embedding", "emb", "features"])
    print(f"Using ONNX Perch: {PERCH_ONNX_PATH}")
else:
    birdclassifier = tf.saved_model.load(str(MODEL_DIR))
    infer_fn = birdclassifier.signatures["serving_default"]
    print("Using TensorFlow Perch SavedModel")
""",
    """USE_ONNX_PERCH = bool(
    CFG.get("enable_onnx_perch", True)
    and PERCH_ONNX_PATH is not None
    and Path(PERCH_ONNX_PATH).exists()
)

if USE_ONNX_PERCH:
    if not _ORT_AVAILABLE:
        raise ImportError(
            "ONNX Perch path was resolved, but `onnxruntime` is not importable. "
            "Attach the ONNX wheel dataset or disable `REQUIRE_ONNX_PERCH`."
        )
    _so = ort.SessionOptions()
    _so.intra_op_num_threads = int(CFG.get("onnx_intra_op_threads", 4))
    ONNX_SESSION = ort.InferenceSession(
        str(PERCH_ONNX_PATH),
        sess_options=_so,
        providers=["CPUExecutionProvider"],
    )
    ONNX_INPUT_NAME = ONNX_SESSION.get_inputs()[0].name
    ONNX_LABEL_INDEX = resolve_onnx_output_index(ONNX_SESSION, ["label", "logits", "scores"])
    ONNX_EMB_INDEX = resolve_onnx_output_index(ONNX_SESSION, ["embedding", "emb", "features"])
    print(f"Using ONNX Perch: {PERCH_ONNX_PATH}")
else:
    if CFG.get("require_onnx_perch", False):
        raise RuntimeError(
            "ONNX Perch was required but could not be activated. "
            "Check `PERCH_ONNX_HINT` and the attached ONNX dataset."
        )
    if not _TF_AVAILABLE:
        raise ImportError("TensorFlow fallback is unavailable in this runtime")
    if MODEL_DIR is None:
        raise FileNotFoundError("TensorFlow fallback requested, but SavedModel could not be resolved")
    birdclassifier = tf.saved_model.load(str(MODEL_DIR))
    infer_fn = birdclassifier.signatures["serving_default"]
    print("Using TensorFlow Perch SavedModel")
""",
    "cell10 strict onnx block",
)
cell10 = replace_once(
    cell10,
    """bc_labels = (
    pd.read_csv(MODEL_DIR / "assets" / "labels.csv")
    .reset_index()
    .rename(columns={"index": "bc_index", "inat2024_fsd50k": "scientific_name"})
)
""",
    """bc_labels = (
    pd.read_csv(PERCH_LABELS_PATH)
    .reset_index()
    .rename(columns={"index": "bc_index", "inat2024_fsd50k": "scientific_name"})
)
""",
    "cell10 labels path",
)
nb["cells"][10]["source"] = lines(cell10)

cell26 = "".join(nb["cells"][26]["source"])
cell26 = replace_once(
    cell26,
    """LOGS["runtime_port"] = {
    "onnx_backend_requested": bool(CFG.get("enable_onnx_perch", False)),
    "onnx_backend_active": bool(USE_ONNX_PERCH),
    "onnx_path": str(PERCH_ONNX_PATH) if PERCH_ONNX_PATH is not None else None,
    "onnx_intra_op_threads": int(CFG.get("onnx_intra_op_threads", 4)),
    "io_prefetch_workers": int(CFG.get("io_prefetch_workers", 4)),
    "vectorized_mlp_probes": bool(CFG.get("enable_vectorized_mlp_probes", False)),
    "batched_tta": bool(CFG.get("enable_batched_tta", False)),
    "tta_max_batch_files": int(CFG.get("tta_max_batch_files", 512)),
}
""",
    """LOGS["runtime_port"] = {
    "onnx_backend_requested": bool(CFG.get("enable_onnx_perch", False)),
    "require_onnx_perch": bool(CFG.get("require_onnx_perch", False)),
    "onnx_backend_active": bool(USE_ONNX_PERCH),
    "onnxruntime_available": bool(_ORT_AVAILABLE),
    "tensorflow_available": bool(_TF_AVAILABLE),
    "onnx_path": str(PERCH_ONNX_PATH) if PERCH_ONNX_PATH is not None else None,
    "labels_path": str(PERCH_LABELS_PATH),
    "model_dir": str(MODEL_DIR) if MODEL_DIR is not None else None,
    "onnx_intra_op_threads": int(CFG.get("onnx_intra_op_threads", 4)),
    "io_prefetch_workers": int(CFG.get("io_prefetch_workers", 4)),
    "vectorized_mlp_probes": bool(CFG.get("enable_vectorized_mlp_probes", False)),
    "batched_tta": bool(CFG.get("enable_batched_tta", False)),
    "tta_max_batch_files": int(CFG.get("tta_max_batch_files", 512)),
}
""",
    "cell26 runtime log enrichment",
)
cell26 = replace_once(
    cell26,
    'with open("/kaggle/working/v18_runtime_port_submit_logs.json", "w") as f:\n        json.dump(LOGS, f, indent=2, default=str)\n    print("Saved /kaggle/working/v18_runtime_port_submit_logs.json")\n',
    'with open("/kaggle/working/v18_onnx_first_runtime_port_submit_logs.json", "w") as f:\n        json.dump(LOGS, f, indent=2, default=str)\n    print("Saved /kaggle/working/v18_onnx_first_runtime_port_submit_logs.json")\n',
    "cell26 log path",
)
nb["cells"][26]["source"] = lines(cell26)

dst_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
print(f"Wrote {dst_path}")
