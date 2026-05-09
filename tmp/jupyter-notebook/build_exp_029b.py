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


src_path = NOTEBOOKS / "kaggle_submission_exp_015d_v18_artifact_submit.ipynb"
dst_path = NOTEBOOKS / "kaggle_submission_exp_029b_exp015d_runtime_port.ipynb"

nb = json.loads(src_path.read_text())

# Title / hints
nb["cells"][3]["source"] = lines(
    """
    # BirdCLEF+ 2026 -- ProtoSSM v5: V18 Runtime Port (Exp 029b)
    """
)

nb["cells"][1]["source"] = lines(
    """
    # Cell 0b — Kaggle input hints
    TF_WHEELS_HINT = None  # e.g. "bc26-tensorflow-2-20-0"
    COMPETITION_HINT = None  # e.g. "birdclef-2026"
    PERCH_MODEL_HINT = None  # e.g. "perch_v2_cpu"
    PERCH_CACHE_HINT = None  # not required for thin-submit; can stay None
    PERCH_ONNX_HINT = None  # optional dataset/model hint for ONNX Perch, e.g. "perch-onnx"
    ARTIFACTS_HINT = None  # e.g. "birdclef-exp015c-v18-artifacts"

    ENABLE_ONNX_PERCH = True
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
        "ONNX_INTRA_OP_THREADS": ONNX_INTRA_OP_THREADS,
        "IO_PREFETCH_WORKERS": IO_PREFETCH_WORKERS,
        "TORCH_INTRA_OP_THREADS": TORCH_INTRA_OP_THREADS,
        "ENABLE_VECTORIZED_MLP_PROBES": ENABLE_VECTORIZED_MLP_PROBES,
        "ENABLE_BATCHED_TTA": ENABLE_BATCHED_TTA,
        "TTA_MAX_BATCH_FILES": TTA_MAX_BATCH_FILES,
    })
    """
)


# Cell 6: imports and config
cell6 = "".join(nb["cells"][6]["source"])
cell6 = replace_once(
    cell6,
    "import gc\nimport json\n",
    "import concurrent.futures\nimport gc\nimport json\n",
    "cell6 imports",
)
cell6 = replace_once(
    cell6,
    'from tqdm.auto import tqdm\n\nwarnings.filterwarnings("ignore")\n',
    '''from tqdm.auto import tqdm\n\ntry:\n    import onnxruntime as ort\n    _ORT_AVAILABLE = True\nexcept ImportError:\n    ort = None\n    _ORT_AVAILABLE = False\n\nwarnings.filterwarnings("ignore")\n''',
    "cell6 ort import",
)
cell6 = replace_once(
    cell6,
    'INPUT_ROOT = Path("/kaggle/input")\n',
    '''INPUT_ROOT = Path("/kaggle/input")\n\nif TORCH_INTRA_OP_THREADS is not None:\n    try:\n        torch.set_num_threads(int(TORCH_INTRA_OP_THREADS))\n    except Exception as e:\n        print(f"Warning: could not set torch intra-op threads: {e}")\n''',
    "cell6 torch threads",
)
cell6 = replace_once(
    cell6,
    """def resolve_perch_cache_dir():
    found = []
    for p in INPUT_ROOT.rglob("full_perch_meta.parquet"):
        parent = p.parent
        if (parent / "full_perch_arrays.npz").exists():
            if PERCH_CACHE_HINT is None or PERCH_CACHE_HINT.lower() in str(parent).lower():
                found.append(parent)
    return sorted(found)[0] if found else None

BASE = resolve_competition_dir()
MODEL_DIR = resolve_perch_model_dir()
PERCH_CACHE_DIR = resolve_perch_cache_dir()
print("Competition dir:", BASE)
print("Perch model dir:", MODEL_DIR)
print("Perch cache dir:", PERCH_CACHE_DIR)
""",
    """def resolve_perch_cache_dir():
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
    "cell6 onnx resolver",
)
cell6 = replace_once(
    cell6,
    '    "batch_files": 16,\n',
    '''    "batch_files": 16,\n    "enable_onnx_perch": bool(ENABLE_ONNX_PERCH),\n    "onnx_intra_op_threads": int(ONNX_INTRA_OP_THREADS),\n    "io_prefetch_workers": int(IO_PREFETCH_WORKERS),\n    "enable_vectorized_mlp_probes": bool(ENABLE_VECTORIZED_MLP_PROBES),\n    "enable_batched_tta": bool(ENABLE_BATCHED_TTA),\n    "tta_max_batch_files": int(TTA_MAX_BATCH_FILES),\n''',
    "cell6 cfg runtime",
)
nb["cells"][6]["source"] = lines(cell6)


# Cell 10: Perch load / ONNX session
cell10 = "".join(nb["cells"][10]["source"])
cell10 = replace_once(
    cell10,
    'BEST = CFG["best_fusion"]\nbirdclassifier = tf.saved_model.load(str(MODEL_DIR))\ninfer_fn = birdclassifier.signatures["serving_default"]\n',
    '''BEST = CFG["best_fusion"]\n\nONNX_SESSION = None\nONNX_INPUT_NAME = None\nONNX_LABEL_INDEX = None\nONNX_EMB_INDEX = None\nbirdclassifier = None\ninfer_fn = None\n\n\ndef resolve_onnx_output_index(session, preferred_names):\n    outputs = session.get_outputs()\n    for preferred in preferred_names:\n        for idx, out in enumerate(outputs):\n            if out.name.lower() == preferred.lower() or preferred.lower() in out.name.lower():\n                return idx\n    raise KeyError(f"Could not resolve ONNX output among {[o.name for o in outputs]} for {preferred_names}")\n\n\nUSE_ONNX_PERCH = bool(\n    CFG.get("enable_onnx_perch", True)\n    and _ORT_AVAILABLE\n    and PERCH_ONNX_PATH is not None\n    and Path(PERCH_ONNX_PATH).exists()\n)\n\nif USE_ONNX_PERCH:\n    _so = ort.SessionOptions()\n    _so.intra_op_num_threads = int(CFG.get("onnx_intra_op_threads", 4))\n    ONNX_SESSION = ort.InferenceSession(\n        str(PERCH_ONNX_PATH),\n        sess_options=_so,\n        providers=["CPUExecutionProvider"],\n    )\n    ONNX_INPUT_NAME = ONNX_SESSION.get_inputs()[0].name\n    ONNX_LABEL_INDEX = resolve_onnx_output_index(ONNX_SESSION, ["label", "logits", "scores"])\n    ONNX_EMB_INDEX = resolve_onnx_output_index(ONNX_SESSION, ["embedding", "emb", "features"])\n    print(f"Using ONNX Perch: {PERCH_ONNX_PATH}")\nelse:\n    birdclassifier = tf.saved_model.load(str(MODEL_DIR))\n    infer_fn = birdclassifier.signatures["serving_default"]\n    print("Using TensorFlow Perch SavedModel")\n''',
    "cell10 onnx load",
)
nb["cells"][10]["source"] = lines(cell10)


# Cell 13: batched TTA
cell13 = "".join(nb["cells"][13]["source"])
old_tta = """def temporal_shift_tta(emb_files, logits_files, model, site_ids, hours, shifts=[0, 1, -1]):
    \"\"\"TTA by circular-shifting the 12-window embedding sequence.\"\"\"
    all_preds = []
    model.eval()
    
    for shift in shifts:
        if shift == 0:
            e = emb_files
            l = logits_files
        else:
            e = np.roll(emb_files, shift, axis=1)
            l = np.roll(logits_files, shift, axis=1)
        
        with torch.no_grad():
            with autocast_context():
                out, _, _ = model(
                    to_device_tensor(e, torch.float32),
                    to_device_tensor(l, torch.float32),
                    site_ids=to_device_tensor(site_ids, torch.long),
                    hours=to_device_tensor(hours, torch.long),
                )
            pred = tensor_to_numpy(out)
        
        if shift != 0:
            pred = np.roll(pred, -shift, axis=1)
        
        all_preds.append(pred)
    
    return np.mean(all_preds, axis=0)
"""
new_tta = """def temporal_shift_tta(emb_files, logits_files, model, site_ids, hours, shifts=[0, 1, -1], max_batch_files=512):
    \"\"\"Batched TTA by circular-shifting the 12-window embedding sequence.\"\"\"
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

    model.eval()
    pred_chunks = []
    with torch.no_grad():
        total = e_batch.shape[0]
        for start_idx in range(0, total, max_batch_files):
            end_idx = min(start_idx + max_batch_files, total)
            with autocast_context():
                out, _, _ = model(
                    to_device_tensor(e_batch[start_idx:end_idx], torch.float32),
                    to_device_tensor(l_batch[start_idx:end_idx], torch.float32),
                    site_ids=to_device_tensor(site_batch[start_idx:end_idx], torch.long),
                    hours=to_device_tensor(hour_batch[start_idx:end_idx], torch.long),
                )
            pred_chunks.append(tensor_to_numpy(out))

    pred_batch = np.concatenate(pred_chunks, axis=0)
    pred_batch = pred_batch.reshape(n_shifts, n_files, pred_batch.shape[1], pred_batch.shape[2])

    all_preds = []
    for i, shift in enumerate(shifts):
        pred_i = pred_batch[i]
        if shift != 0:
            pred_i = np.roll(pred_i, -shift, axis=1)
        all_preds.append(pred_i)
    return np.mean(all_preds, axis=0)
"""
cell13 = replace_once(cell13, old_tta, new_tta, "cell13 batched tta")
nb["cells"][13]["source"] = lines(cell13)


# Cell 15: ONNX + prefetch Perch inference
nb["cells"][15]["source"] = lines(
    """
    # Cell 5 — Perch inference with embeddings + selective proxies
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

        next_paths = paths[0:batch_files]
        max_workers = max(1, int(CFG.get("io_prefetch_workers", 4)))

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as io_executor:
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

                if USE_ONNX_PERCH:
                    onnx_outs = ONNX_SESSION.run(None, {ONNX_INPUT_NAME: x})
                    logits = onnx_outs[ONNX_LABEL_INDEX].astype(np.float32, copy=False)
                    emb = onnx_outs[ONNX_EMB_INDEX].astype(np.float32, copy=False)
                else:
                    outputs = infer_fn(inputs=tf.convert_to_tensor(x))
                    logits = outputs["label"].numpy().astype(np.float32, copy=False)
                    emb = outputs["embedding"].numpy().astype(np.float32, copy=False)

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
                    scores[batch_row_start:write_row, pos] = proxy_score.astype(np.float32, copy=False)

                del x, logits, emb
                if not USE_ONNX_PERCH:
                    del outputs
                gc.collect()

        meta_df = pd.DataFrame({
            "row_id": row_ids,
            "filename": filenames,
            "site": sites,
            "hour_utc": hours,
        })

        LOGS["perch_runtime"] = {
            "backend": "onnx" if USE_ONNX_PERCH else "tensorflow",
            "io_prefetch_workers": int(CFG.get("io_prefetch_workers", 4)),
            "batch_files": int(batch_files),
        }
        return meta_df, scores, embeddings
    """
)


# Cell 18: append vectorized probe helpers
cell18 = "".join(nb["cells"][18]["source"])
cell18 += """


def build_all_class_features_vectorized(Z, raw_scores, prior_scores, base_scores, valid_classes, n_windows=12):
    N, D = Z.shape
    V = len(valid_classes)
    raw = raw_scores[:, valid_classes].T
    prior = prior_scores[:, valid_classes].T
    base = base_scores[:, valid_classes].T

    n_files = N // n_windows
    base_view = base.reshape(V, n_files, n_windows)
    prev_base = np.concatenate([base_view[:, :, :1], base_view[:, :, :-1]], axis=2).reshape(V, N)
    next_base = np.concatenate([base_view[:, :, 1:], base_view[:, :, -1:]], axis=2).reshape(V, N)
    mean_base = np.repeat(base_view.mean(axis=2), n_windows, axis=1)
    max_base = np.repeat(base_view.max(axis=2), n_windows, axis=1)
    std_base = np.repeat(base_view.std(axis=2), n_windows, axis=1)

    diff_mean = base - mean_base
    diff_prev = base - prev_base
    diff_next = base - next_base
    interact_rp = raw * prior
    interact_rb = raw * base
    interact_pb = prior * base

    scalar_feats = np.stack([
        raw, prior, base, prev_base, next_base,
        mean_base, max_base, std_base,
        diff_mean, diff_prev, diff_next,
        interact_rp, interact_rb, interact_pb,
    ], axis=-1)

    z_expanded = np.broadcast_to(Z, (V, N, D))
    x_all = np.concatenate([z_expanded, scalar_feats], axis=-1)
    return x_all.astype(np.float32, copy=False)


class VectorizedMLPProbes(nn.Module):
    def __init__(self, probe_models, device="cpu"):
        super().__init__()
        self.valid_classes = sorted(list(probe_models.keys()))
        self.V = len(self.valid_classes)
        if self.V == 0:
            return

        sample_clf = probe_models[self.valid_classes[0]]
        self.n_layers = len(sample_clf.coefs_)
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()

        for layer_idx in range(self.n_layers):
            w = np.stack([probe_models[c].coefs_[layer_idx] for c in self.valid_classes], axis=0)
            b = np.stack([probe_models[c].intercepts_[layer_idx] for c in self.valid_classes], axis=0)
            self.weights.append(nn.Parameter(torch.tensor(w, dtype=torch.float32), requires_grad=False))
            self.biases.append(nn.Parameter(torch.tensor(b, dtype=torch.float32), requires_grad=False))
        self.to(device)

    def forward(self, x):
        h = x
        for i in range(self.n_layers):
            h = torch.bmm(h, self.weights[i]) + self.biases[i].unsqueeze(1)
            if i < self.n_layers - 1:
                h = torch.relu(h)
        return h.squeeze(-1)


def get_vectorized_mlp_scores(Z, raw, prior, base, probe_models, alpha_p, n_windows=12, device="cpu"):
    mlp_scores = base.copy()
    if len(probe_models) == 0:
        return mlp_scores

    valid_classes = sorted(list(probe_models.keys()))
    x_all = build_all_class_features_vectorized(Z, raw, prior, base, valid_classes, n_windows=n_windows)
    vec_probe = VectorizedMLPProbes(probe_models, device=device)
    vec_probe.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(x_all, dtype=torch.float32, device=device)
        preds = vec_probe(x_tensor).cpu().numpy()

    preds_t = preds.T
    base_valid = base[:, valid_classes]
    mlp_scores[:, valid_classes] = (1.0 - alpha_p) * base_valid + alpha_p * preds_t
    return mlp_scores
"""
nb["cells"][18]["source"] = lines(cell18)


# Cell 24: batched TTA call + vectorized probe path
cell24 = "".join(nb["cells"][24]["source"])
cell24 = replace_once(
    cell24,
    """if len(tta_shifts) > 1:
    print(f"Running TTA with shifts: {tta_shifts}")
    proto_scores = temporal_shift_tta(
        emb_test_files, logits_test_files, model,
        test_site_ids, test_hours, shifts=tta_shifts
    )
else:
""",
    """if len(tta_shifts) > 1:
    print(f"Running TTA with shifts: {tta_shifts}")
    proto_scores = temporal_shift_tta(
        emb_test_files, logits_test_files, model,
        test_site_ids, test_hours, shifts=tta_shifts,
        max_batch_files=CFG.get("tta_max_batch_files", 512),
    )
else:
""",
    "cell24 tta call",
)
old_probe = """mlp_scores = test_base_scores.copy()

for cls_idx, clf in probe_models.items():
    X_cls_test = build_class_features(
        Z_TEST,
        raw_col=scores_test_raw[:, cls_idx],
        prior_col=test_prior_scores[:, cls_idx],
        base_col=test_base_scores[:, cls_idx],
    )

    if hasattr(clf, "predict_proba"):
        prob = clf.predict_proba(X_cls_test)[:, 1].astype(np.float32)
        pred = np.log(prob + 1e-7) - np.log(1 - prob + 1e-7)
    else:
        pred = clf.decision_function(X_cls_test).astype(np.float32)

    alpha = float(CFG["frozen_best_probe"]["alpha"])
    mlp_scores[:, cls_idx] = (1.0 - alpha) * test_base_scores[:, cls_idx] + alpha * pred
"""
new_probe = """mlp_scores = test_base_scores.copy()

alpha = float(CFG["frozen_best_probe"]["alpha"])
if CFG.get("enable_vectorized_mlp_probes", False):
    mlp_scores = get_vectorized_mlp_scores(
        Z_TEST,
        scores_test_raw,
        test_prior_scores,
        test_base_scores,
        probe_models,
        alpha,
        n_windows=N_WINDOWS,
        device=DEVICE,
    )
else:
    for cls_idx, clf in probe_models.items():
        X_cls_test = build_class_features(
            Z_TEST,
            raw_col=scores_test_raw[:, cls_idx],
            prior_col=test_prior_scores[:, cls_idx],
            base_col=test_base_scores[:, cls_idx],
        )

        if hasattr(clf, "predict_proba"):
            prob = clf.predict_proba(X_cls_test)[:, 1].astype(np.float32)
            pred = np.log(prob + 1e-7) - np.log(1 - prob + 1e-7)
        else:
            pred = clf.decision_function(X_cls_test).astype(np.float32)

        mlp_scores[:, cls_idx] = (1.0 - alpha) * test_base_scores[:, cls_idx] + alpha * pred
"""
cell24 = replace_once(cell24, old_probe, new_probe, "cell24 vectorized probe")
nb["cells"][24]["source"] = lines(cell24)


# Cell 26: runtime logs
cell26 = "".join(nb["cells"][26]["source"])
cell26 = replace_once(
    cell26,
    'LOGS["v18_improvements"] = [\n    "d_model_320", "n_ssm_layers_4", "n_prototypes_2", "meta_dim_24",\n    "cross_attention_heads_8", "per_taxon_temperature", "file_level_scaling",\n    "tta_disabled", "rank_aware_scaling_v18", "adaptive_delta_smooth",\n    "updated_probe_defaults", "updated_fusion_lambdas", "per_class_thresholds"\n]\n',
    '''LOGS["v18_improvements"] = [\n    "d_model_320", "n_ssm_layers_4", "n_prototypes_2", "meta_dim_24",\n    "cross_attention_heads_8", "per_taxon_temperature", "file_level_scaling",\n    "tta_disabled", "rank_aware_scaling_v18", "adaptive_delta_smooth",\n    "updated_probe_defaults", "updated_fusion_lambdas", "per_class_thresholds"\n]\nLOGS["runtime_port"] = {\n    "onnx_backend_requested": bool(CFG.get("enable_onnx_perch", False)),\n    "onnx_backend_active": bool(USE_ONNX_PERCH),\n    "onnx_path": str(PERCH_ONNX_PATH) if PERCH_ONNX_PATH is not None else None,\n    "onnx_intra_op_threads": int(CFG.get("onnx_intra_op_threads", 4)),\n    "io_prefetch_workers": int(CFG.get("io_prefetch_workers", 4)),\n    "vectorized_mlp_probes": bool(CFG.get("enable_vectorized_mlp_probes", False)),\n    "batched_tta": bool(CFG.get("enable_batched_tta", False)),\n    "tta_max_batch_files": int(CFG.get("tta_max_batch_files", 512)),\n}\n''',
    "cell26 runtime log dict",
)
cell26 = replace_once(
    cell26,
    '    with open("/kaggle/working/v18_artifact_submit_logs.json", "w") as f:\n',
    '    with open("/kaggle/working/v18_runtime_port_submit_logs.json", "w") as f:\n',
    "cell26 log filename open",
)
cell26 = replace_once(
    cell26,
    '    print("Saved /kaggle/working/v18_artifact_submit_logs.json")\n',
    '    print("Saved /kaggle/working/v18_runtime_port_submit_logs.json")\n',
    "cell26 log filename print",
)
cell26 = replace_once(
    cell26,
    '    print("Artifactized submit mode completed.")\n',
    '    print("Artifactized runtime-port submit mode completed.")\n',
    "cell26 final print",
)
nb["cells"][26]["source"] = lines(cell26)

dst_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print(dst_path)
