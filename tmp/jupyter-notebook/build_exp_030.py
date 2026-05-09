from __future__ import annotations

import json
import textwrap
from copy import deepcopy
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


base_path = NOTEBOOKS / "kaggle_submission_exp_029c_exp015d_onnx_first_runtime_port.ipynb"
overlay_path = NOTEBOOKS / "kaggle_submission_exp_018e_exp015d_texture_overlay_accel.ipynb"
dst_path = NOTEBOOKS / "kaggle_submission_exp_030_exp029c_texture_overlay_multifold.ipynb"

base_nb = json.loads(base_path.read_text())
overlay_nb = json.loads(overlay_path.read_text())

nb = deepcopy(base_nb)


def find_cell_index_containing(notebook: dict, needle: str) -> int:
    for idx, cell in enumerate(notebook["cells"]):
        src = "".join(cell.get("source", []))
        if needle in src:
            return idx
    raise RuntimeError(f"Could not find cell containing {needle!r}")

nb["cells"][3]["source"] = lines(
    """
    # BirdCLEF+ 2026 -- V18 ONNX-First + Multi-Fold Texture Overlay (Exp 030)
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

    EXP018A_MODEL_DATASET_HINT = None  # e.g. "birdclef-exp018a-texture-specialist-4fold"
    RUN_EXP018A_OVERLAY = True
    EXP018A_BLEND_WEIGHT = 0.35
    EXP018A_FOLD_IDS = (0, 1, 3)  # strongest positive specialist folds from exp_018a
    EXP018A_CHECKPOINT_NAME = "best_model.pt"
    EXP018A_BATCH_FILES = 12
    EXP018A_MAX_START_WALL_SECONDS = 3000
    EXP018A_ABORT_WALL_SECONDS = 4800
    EXP018A_SAVE_BASELINE_FIRST = True
    EXP018A_ACCEL_BACKEND = "torchscript"  # torchscript | auto | eager | openvino
    EXP018A_TRACE_BATCH_WAVES = 32
    EXP018A_OPENVINO_NUM_REQUESTS = 2
    EXP018A_OPENVINO_CHUNK_ROWS = 128

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
        "EXP018A_MODEL_DATASET_HINT": EXP018A_MODEL_DATASET_HINT,
        "RUN_EXP018A_OVERLAY": RUN_EXP018A_OVERLAY,
        "EXP018A_BLEND_WEIGHT": EXP018A_BLEND_WEIGHT,
        "EXP018A_FOLD_IDS": EXP018A_FOLD_IDS,
        "EXP018A_BATCH_FILES": EXP018A_BATCH_FILES,
        "EXP018A_MAX_START_WALL_SECONDS": EXP018A_MAX_START_WALL_SECONDS,
        "EXP018A_ABORT_WALL_SECONDS": EXP018A_ABORT_WALL_SECONDS,
        "EXP018A_ACCEL_BACKEND": EXP018A_ACCEL_BACKEND,
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

cell6 = "".join(nb["cells"][6]["source"])
cell6 = replace_once(
    cell6,
    "import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n",
    "import timm\nimport torchaudio\n\nimport torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n",
    "cell6 timm torchaudio imports",
)
cell6 = replace_once(
    cell6,
    "LOGS = {}\n\nCFG = {\n",
    """LOGS = {}

def overlay_wall_elapsed_seconds() -> float:
    return float(time.time() - _WALL_START)

def save_primary_submission_snapshot(row_ids, probs, tag: str):
    submission = pd.DataFrame(probs, columns=PRIMARY_LABELS)
    submission.insert(0, "row_id", np.asarray(row_ids))
    submission[PRIMARY_LABELS] = submission[PRIMARY_LABELS].astype(np.float32)
    submission.to_csv("submission.csv", index=False)
    out_path = Path('/kaggle/working') / f'submission_{tag}.csv'
    submission.to_csv(out_path, index=False)
    LOGS.setdefault('submission_snapshots', []).append({
        'tag': str(tag),
        'elapsed_seconds': overlay_wall_elapsed_seconds(),
        'path': str(out_path),
        'rows': int(len(submission)),
        'mean': float(np.asarray(probs).mean()),
    })
    print(f"Saved primary submission snapshot: {tag}")
    return submission

CFG = {
""",
    "cell6 overlay helpers",
)
nb["cells"][6]["source"] = lines(cell6)

# Insert overlay runtime cell from exp_018e after config/taxonomy setup and before Perch load.
nb["cells"].insert(9, deepcopy(overlay_nb["cells"][9]))

postprocess_idx = find_cell_index_containing(nb, "# --- Build submission ---")
cell25 = "".join(nb["cells"][postprocess_idx]["source"])
overlay_block = "".join(overlay_nb["cells"][26]["source"])
overlay_start = overlay_block.index("# --- Optional exp_018a texture overlay ---")
overlay_end = overlay_block.index("# --- Build submission ---")
overlay_snippet = overlay_block[overlay_start:overlay_end]
cell25 = replace_once(
    cell25,
    "# --- Build submission ---\n",
    overlay_snippet + "\n# --- Build submission ---\n",
    "cell25 overlay injection",
)
nb["cells"][postprocess_idx]["source"] = lines(cell25)

logging_idx = find_cell_index_containing(nb, 'LOGS["per_class_thresholds"] = PER_CLASS_THRESHOLDS.tolist()')
cell26 = "".join(nb["cells"][logging_idx]["source"])
cell26 = replace_once(
    cell26,
    'LOGS["per_class_thresholds"] = PER_CLASS_THRESHOLDS.tolist()\n',
    """LOGS["per_class_thresholds"] = PER_CLASS_THRESHOLDS.tolist()
LOGS["overlay_profile"] = "multifold_accel_texture_overlay_onnx_first"
""",
    "cell26 overlay profile",
)
cell26 = replace_once(
    cell26,
    'with open("/kaggle/working/v18_onnx_first_runtime_port_submit_logs.json", "w") as f:\n        json.dump(LOGS, f, indent=2, default=str)\n    print("Saved /kaggle/working/v18_onnx_first_runtime_port_submit_logs.json")\n',
    'with open("/kaggle/working/exp_030_texture_overlay_multifold_logs.json", "w") as f:\n        json.dump(LOGS, f, indent=2, default=str)\n    print("Saved /kaggle/working/exp_030_texture_overlay_multifold_logs.json")\n',
    "cell26 log path",
)
cell26 = replace_once(
    cell26,
    'print("Artifactized runtime-port submit mode completed.")\n',
    'print("ONNX-first multi-fold texture-overlay submit mode completed.")\n',
    "cell26 final print",
)
nb["cells"][logging_idx]["source"] = lines(cell26)

dst_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
print(f"Wrote {dst_path}")
