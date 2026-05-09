from __future__ import annotations

import json
import textwrap
from pathlib import Path


ROOT = Path("/Users/yaroslav/Documents/Kaggle/BirdCLEF_2026")
NOTEBOOKS = ROOT / "notebooks"


def load_nb(path: Path) -> dict:
    return json.loads(path.read_text())


def lines(text: str) -> list[str]:
    text = textwrap.dedent(text).strip("\n")
    if not text:
        return []
    return [f"{line}\n" for line in text.splitlines()]


def md_cell(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": lines(text)}


def code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines(text),
    }


base_md = load_nb(NOTEBOOKS / "exp_011_hgnetv2_soundscape_supervised.ipynb")["metadata"]


cells = [
    md_cell(
        """
        # Exp 029a: Perch ONNX Compatibility Benchmark

        Safety-first benchmark for the forum idea "just use Perch v2 in ONNX version".
        """
    ),
    md_cell(
        """
        ## Goal

        We are **not** testing a new score source here. We are testing whether an ONNX-exported Perch cache is close enough to the current TensorFlow/official cache that the fixed `exp_015d` artifact stack still behaves the same.

        The notebook answers three questions:

        1. How much do raw `scores_full_raw` and `emb_full` drift?
        2. Does the fixed `exp_015d` replay stay stable on the ONNX cache?
        3. Is the drift small enough that an ONNX Perch replacement is engineering-safe for future runtime work?
        """
    ),
    code_cell(
        """
        from __future__ import annotations

        import gc
        import json
        import pickle
        import re
        import time
        import typing as tp
        import warnings
        from contextlib import nullcontext
        from dataclasses import dataclass
        from pathlib import Path

        import numpy as np
        import pandas as pd
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from sklearn.metrics import roc_auc_score

        warnings.filterwarnings('ignore')

        try:
            from IPython.display import display
        except Exception:
            def display(obj: object) -> None:
                print(obj)


        def resolve_repo_root(start: Path | None = None) -> Path:
            current = (start or Path.cwd()).resolve()
            for candidate in [current, *current.parents]:
                if (candidate / 'PROJECT_STATE.md').exists() and (candidate / 'data').exists():
                    return candidate
            raise FileNotFoundError('Could not resolve repository root')


        @dataclass
        class Config:
            experiment_id: str = 'exp_029a'
            experiment_name: str = 'perch_onnx_compat_benchmark'
            teacher_dir_override: str | None = None
            official_perch_dir_override: str | None = None
            onnx_perch_dir_override: str | None = None
            artifacts_dir_override: str | None = None
            onnx_dir_hint: str = 'onnx'
            topk_compare: int = 5


        CFG = Config()
        ROOT = resolve_repo_root()
        DATA = ROOT / 'data' / 'birdclef-2026'
        EXPERIMENTS = ROOT / 'experiments'
        OUTPUT_DIR = EXPERIMENTS / 'outputs' / f'{CFG.experiment_id}_{CFG.experiment_name}'
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        INPUT_ROOT = Path('/kaggle/input') if Path('/kaggle/input').exists() else ROOT

        if torch.cuda.is_available():
            DEVICE = torch.device('cuda')
        elif getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
            DEVICE = torch.device('mps')
        else:
            DEVICE = torch.device('cpu')
        AMP_ENABLED = DEVICE.type == 'cuda'


        def autocast_context():
            if AMP_ENABLED:
                return torch.amp.autocast('cuda', enabled=True)
            return nullcontext()


        def to_device_tensor(x, dtype):
            return torch.as_tensor(x, dtype=dtype, device=DEVICE)


        def tensor_to_numpy(x):
            return x.detach().cpu().numpy()


        def save_json(payload: dict[str, tp.Any], path: Path) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, indent=2, sort_keys=False))


        print({
            'root': str(ROOT),
            'output_dir': str(OUTPUT_DIR),
            'device': str(DEVICE),
            'teacher_dir_override': CFG.teacher_dir_override,
            'official_perch_dir_override': CFG.official_perch_dir_override,
            'onnx_perch_dir_override': CFG.onnx_perch_dir_override,
            'artifacts_dir_override': CFG.artifacts_dir_override,
        })
        """
    ),
    code_cell(
        """
        def has_teacher_cache(path: Path) -> bool:
            return (path / 'teacher_meta.parquet').exists() and (path / 'teacher_outputs.npz').exists()


        def resolve_teacher_dir() -> Path:
            candidates = []
            if CFG.teacher_dir_override:
                candidates.append(Path(CFG.teacher_dir_override).expanduser())
            candidates.extend([
                ROOT / 'experiments' / 'outputs' / 'exp_027a_exp015d_teacher_cache',
                Path.home() / 'Downloads' / 'exp_027a_exp015d_teacher_cache',
            ])
            for candidate in candidates:
                if has_teacher_cache(candidate):
                    return candidate
            for search_root in [ROOT / 'experiments' / 'outputs', ROOT / 'data']:
                if not search_root.exists():
                    continue
                for meta_path in search_root.rglob('teacher_meta.parquet'):
                    parent = meta_path.parent
                    if has_teacher_cache(parent) and 'exp_027a' in str(parent):
                        return parent
            raise FileNotFoundError(
                'Could not resolve exp_027a teacher cache. '
                'Run exp_027a first or set CFG.teacher_dir_override.'
            )


        def resolve_artifacts_dir() -> Path:
            candidates = []
            if CFG.artifacts_dir_override:
                candidates.append(Path(CFG.artifacts_dir_override).expanduser())
            for candidate in candidates:
                if (candidate / 'artifacts_manifest.json').exists():
                    return candidate

            search_roots = [
                ROOT / 'submissions',
                ROOT / 'experiments',
                ROOT / 'data',
                INPUT_ROOT,
            ]
            for search_root in search_roots:
                if not search_root.exists():
                    continue
                found = []
                for p in search_root.rglob('artifacts_manifest.json'):
                    parent = p.parent
                    parent_str = str(parent).lower()
                    score = 0
                    if 'exp015' in parent_str:
                        score += 2
                    if 'artifact' in parent_str:
                        score += 1
                    found.append((score, len(parent.parts), parent))
                if found:
                    found.sort(key=lambda x: (-x[0], x[1], str(x[2])))
                    return found[0][2]

            raise FileNotFoundError(
                'Could not resolve exp_015d artifact directory. '
                'Set CFG.artifacts_dir_override to the directory containing artifacts_manifest.json.'
            )


        def resolve_perch_cache_dir(
            override: str | None,
            *,
            require_onnx: bool,
        ) -> tuple[Path, Path]:
            meta_names = (
                'full_perch_meta.parquet',
                'full_perch_onnx_meta.parquet',
                'onnx_full_perch_meta.parquet',
                'perch_onnx_meta.parquet',
            )
            npz_names = (
                'full_perch_arrays.npz',
                'full_perch_onnx_arrays.npz',
                'onnx_full_perch_arrays.npz',
                'perch_onnx_arrays.npz',
            )

            explicit_roots = []
            if override:
                explicit_roots.append(Path(override).expanduser())

            search_roots = [
                ROOT / 'data',
                ROOT / 'experiments',
                ROOT / 'submissions',
                INPUT_ROOT,
            ]

            def match_pair(meta_path: Path, npz_path: Path) -> bool:
                joined = f'{meta_path} {npz_path}'.lower()
                if require_onnx:
                    return CFG.onnx_dir_hint.lower() in joined
                return CFG.onnx_dir_hint.lower() not in joined or str(meta_path.parent).endswith('perch_meta')

            def try_dir(path: Path) -> tuple[Path, Path] | None:
                if not path.exists():
                    return None
                if path.is_file():
                    for npz_name in npz_names:
                        npz_path = path.parent / npz_name
                        if npz_path.exists():
                            if not require_onnx or CFG.onnx_dir_hint.lower() in str(path.parent).lower() or 'onnx' in path.name.lower() or 'onnx' in npz_name.lower():
                                return path, npz_path
                    return None
                for meta_name in meta_names:
                    meta_path = path / meta_name
                    if not meta_path.exists():
                        continue
                    for npz_name in npz_names:
                        npz_path = path / npz_name
                        if npz_path.exists():
                            if match_pair(meta_path, npz_path):
                                return meta_path, npz_path
                return None

            for candidate in explicit_roots:
                found = try_dir(candidate)
                if found is not None:
                    return found

            ranked = []
            for search_root in search_roots:
                if not search_root.exists():
                    continue
                for meta_name in meta_names:
                    for meta_path in search_root.rglob(meta_name):
                        parent = meta_path.parent
                        for npz_name in npz_names:
                            npz_path = parent / npz_name
                            if not npz_path.exists():
                                continue
                            if not match_pair(meta_path, npz_path):
                                continue
                            path_str = str(parent).lower()
                            score = 0
                            if require_onnx and 'onnx' in path_str:
                                score += 3
                            if (not require_onnx) and path_str.endswith('perch_meta'):
                                score += 2
                            if 'full' in meta_path.name.lower():
                                score += 1
                            ranked.append((score, len(parent.parts), meta_path, npz_path))

            if ranked:
                ranked.sort(key=lambda x: (-x[0], x[1], str(x[2])))
                _, _, meta_path, npz_path = ranked[0]
                return meta_path, npz_path

            cache_type = 'ONNX Perch cache' if require_onnx else 'official Perch cache'
            raise FileNotFoundError(
                f'Could not resolve {cache_type}. '
                f'Set the corresponding override in CFG if your cache lives elsewhere.'
            )


        def detect_array_key(npz_obj: np.lib.npyio.NpzFile, expected_rows: int, preferred: list[str]) -> str:
            for key in preferred:
                if key in npz_obj and getattr(npz_obj[key], 'ndim', 0) == 2 and npz_obj[key].shape[0] == expected_rows:
                    return key
            for key in npz_obj.files:
                arr = npz_obj[key]
                if getattr(arr, 'ndim', 0) == 2 and arr.shape[0] == expected_rows:
                    return key
            raise KeyError(f'Could not detect compatible array key for {expected_rows} rows. Available keys: {list(npz_obj.files)}')


        TEACHER_DIR = resolve_teacher_dir()
        OFFICIAL_META_PATH, OFFICIAL_NPZ_PATH = resolve_perch_cache_dir(CFG.official_perch_dir_override, require_onnx=False)
        ONNX_META_PATH, ONNX_NPZ_PATH = resolve_perch_cache_dir(CFG.onnx_perch_dir_override, require_onnx=True)
        ARTIFACTS_DIR = resolve_artifacts_dir()

        print({
            'teacher_dir': str(TEACHER_DIR),
            'official_meta_path': str(OFFICIAL_META_PATH),
            'official_npz_path': str(OFFICIAL_NPZ_PATH),
            'onnx_meta_path': str(ONNX_META_PATH),
            'onnx_npz_path': str(ONNX_NPZ_PATH),
            'artifacts_dir': str(ARTIFACTS_DIR),
        })
        """
    ),
    code_cell(
        """
        taxonomy = pd.read_csv(DATA / 'taxonomy.csv')
        PRIMARY_LABELS = taxonomy['primary_label'].astype(str).tolist()
        N_CLASSES = len(PRIMARY_LABELS)
        CLASS_NAME_MAP = taxonomy.set_index('primary_label')['class_name'].to_dict()
        TEXTURE_TAXA = {'Amphibia', 'Insecta'}
        TEXTURE_IDX = np.array(
            [i for i, label in enumerate(PRIMARY_LABELS) if CLASS_NAME_MAP.get(label) in TEXTURE_TAXA],
            dtype=np.int64,
        )

        teacher_meta = pd.read_parquet(TEACHER_DIR / 'teacher_meta.parquet').reset_index(drop=True)
        teacher_arr = np.load(TEACHER_DIR / 'teacher_outputs.npz')
        Y_TRUE = teacher_arr['labels'].astype(np.float32)
        TEACHER_LOGITS = teacher_arr['teacher_logits'].astype(np.float32)
        TEACHER_PROBS = teacher_arr['teacher_probs'].astype(np.float32)
        TEACHER_RAW = teacher_arr['raw_scores'].astype(np.float32)


        def load_aligned_cache(meta_path: Path, npz_path: Path, cache_name: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
            meta = pd.read_parquet(meta_path).reset_index(drop=True)
            if 'row_id' not in meta.columns:
                raise KeyError(f'{cache_name} meta is missing required column `row_id`')
            arr = np.load(npz_path)
            score_key = detect_array_key(arr, len(meta), ['scores_full_raw', 'raw_scores', 'scores', 'logits'])
            emb_key = detect_array_key(arr, len(meta), ['emb_full', 'embeddings', 'emb', 'audio_emb', 'audio_embeddings'])
            scores = arr[score_key].astype(np.float32)
            emb = arr[emb_key].astype(np.float32)
            if scores.shape[1] != N_CLASSES:
                raise ValueError(f'{cache_name} scores have wrong class dimension: {scores.shape}')

            lookup = meta[['row_id']].copy()
            lookup['cache_index'] = np.arange(len(lookup))
            aligned = teacher_meta[['row_id']].merge(lookup, on='row_id', how='left')
            if aligned['cache_index'].isna().any():
                missing = aligned.loc[aligned['cache_index'].isna(), 'row_id'].head(5).tolist()
                raise ValueError(f'{cache_name} cache is missing teacher rows, e.g. {missing}')
            order = aligned['cache_index'].astype(int).to_numpy()
            meta_aligned = meta.iloc[order].reset_index(drop=True)
            scores_aligned = scores[order]
            emb_aligned = emb[order]
            assert np.all(meta_aligned['row_id'].to_numpy() == teacher_meta['row_id'].to_numpy())
            return meta_aligned, scores_aligned, emb_aligned


        meta_official, scores_official_raw, emb_official = load_aligned_cache(OFFICIAL_META_PATH, OFFICIAL_NPZ_PATH, 'official')
        meta_onnx, scores_onnx_raw, emb_onnx = load_aligned_cache(ONNX_META_PATH, ONNX_NPZ_PATH, 'onnx')

        assert np.all(meta_official['row_id'].to_numpy() == teacher_meta['row_id'].to_numpy())
        assert np.all(meta_onnx['row_id'].to_numpy() == teacher_meta['row_id'].to_numpy())

        N_WINDOWS = int(teacher_meta.groupby('filename').size().mode().iloc[0])
        assert len(teacher_meta) % N_WINDOWS == 0, f'Expected full-file blocks, got rows={len(teacher_meta)} windows={N_WINDOWS}'

        setup_snapshot = {
            'experiment_id': CFG.experiment_id,
            'experiment_name': CFG.experiment_name,
            'rows': int(len(teacher_meta)),
            'files': int(teacher_meta['filename'].nunique()),
            'n_classes': int(N_CLASSES),
            'n_windows': int(N_WINDOWS),
            'official_emb_dim': int(emb_official.shape[1]),
            'onnx_emb_dim': int(emb_onnx.shape[1]),
        }
        save_json(setup_snapshot, OUTPUT_DIR / 'setup_snapshot.json')
        display(pd.DataFrame([setup_snapshot]))
        """
    ),
    code_cell(
        """
        def macro_auc_skip_empty(y_true, y_score):
            keep = y_true.sum(axis=0) > 0
            if keep.sum() == 0:
                return float('nan')
            return float(roc_auc_score(y_true[:, keep], y_score[:, keep], average='macro'))


        def macro_auc_subset(y_true, y_score, idx):
            idx = np.array(idx, dtype=np.int64)
            idx = idx[(idx >= 0) & (idx < y_true.shape[1])]
            if len(idx) == 0:
                return float('nan')
            keep = y_true[:, idx].sum(axis=0) > 0
            if keep.sum() == 0:
                return float('nan')
            return float(roc_auc_score(y_true[:, idx][:, keep], y_score[:, idx][:, keep], average='macro'))


        def safe_auc(y_true, y_score):
            if y_true.sum() == 0 or y_true.sum() == len(y_true):
                return float('nan')
            return float(roc_auc_score(y_true, y_score))


        def mean_cosine(a, b):
            a_n = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-8, None)
            b_n = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-8, None)
            cos = (a_n * b_n).sum(axis=1)
            return float(cos.mean()), float(cos.min()), float(cos.max())


        def flat_corr(a, b):
            a1 = a.reshape(-1).astype(np.float64)
            b1 = b.reshape(-1).astype(np.float64)
            if np.std(a1) < 1e-12 or np.std(b1) < 1e-12:
                return float('nan')
            return float(np.corrcoef(a1, b1)[0, 1])


        def topk_overlap_ratio(a, b, k=5):
            top_a = np.argpartition(a, -k, axis=1)[:, -k:]
            top_b = np.argpartition(b, -k, axis=1)[:, -k:]
            ratios = []
            for ra, rb in zip(top_a, top_b):
                ratios.append(len(set(ra.tolist()) & set(rb.tolist())) / float(k))
            return float(np.mean(ratios))


        def file_level_max(x):
            return x.reshape(-1, N_WINDOWS, x.shape[1]).max(axis=1)


        def classwise_auc_df(y_true, score_a, score_b, label_a='official', label_b='onnx'):
            rows = []
            file_y = file_level_max(y_true)
            file_a = file_level_max(score_a)
            file_b = file_level_max(score_b)
            for ci, label in enumerate(PRIMARY_LABELS):
                row_auc_a = safe_auc(y_true[:, ci], score_a[:, ci])
                row_auc_b = safe_auc(y_true[:, ci], score_b[:, ci])
                file_auc_a = safe_auc(file_y[:, ci], file_a[:, ci])
                file_auc_b = safe_auc(file_y[:, ci], file_b[:, ci])
                rows.append({
                    'primary_label': label,
                    'taxon': CLASS_NAME_MAP.get(label, 'Unknown'),
                    f'{label_a}_row_auc': row_auc_a,
                    f'{label_b}_row_auc': row_auc_b,
                    'row_auc_delta': row_auc_b - row_auc_a if pd.notna(row_auc_a) and pd.notna(row_auc_b) else np.nan,
                    f'{label_a}_file_auc': file_auc_a,
                    f'{label_b}_file_auc': file_auc_b,
                    'file_auc_delta': file_auc_b - file_auc_a if pd.notna(file_auc_a) and pd.notna(file_auc_b) else np.nan,
                    'positive_rows': int(y_true[:, ci].sum()),
                    'positive_files': int(file_y[:, ci].sum()),
                })
            return pd.DataFrame(rows)


        def taxon_summary_df(classwise_df, label_a='official', label_b='onnx'):
            rows = []
            for taxon, chunk in classwise_df.groupby('taxon'):
                rows.append({
                    'taxon': taxon,
                    f'{label_a}_row_auc': float(chunk[f'{label_a}_row_auc'].dropna().mean()) if chunk[f'{label_a}_row_auc'].notna().any() else np.nan,
                    f'{label_b}_row_auc': float(chunk[f'{label_b}_row_auc'].dropna().mean()) if chunk[f'{label_b}_row_auc'].notna().any() else np.nan,
                    f'{label_a}_file_auc': float(chunk[f'{label_a}_file_auc'].dropna().mean()) if chunk[f'{label_a}_file_auc'].notna().any() else np.nan,
                    f'{label_b}_file_auc': float(chunk[f'{label_b}_file_auc'].dropna().mean()) if chunk[f'{label_b}_file_auc'].notna().any() else np.nan,
                    'row_auc_delta': float(chunk['row_auc_delta'].dropna().mean()) if chunk['row_auc_delta'].notna().any() else np.nan,
                    'file_auc_delta': float(chunk['file_auc_delta'].dropna().mean()) if chunk['file_auc_delta'].notna().any() else np.nan,
                    'scored_classes': int(chunk['positive_rows'].gt(0).sum()),
                })
            return pd.DataFrame(rows).sort_values('row_auc_delta')
        """
    ),
    code_cell(
        """
        raw_metrics = {
            'rows': int(len(teacher_meta)),
            'files': int(teacher_meta['filename'].nunique()),
            'score_mae': float(np.mean(np.abs(scores_official_raw - scores_onnx_raw))),
            'score_max_abs': float(np.max(np.abs(scores_official_raw - scores_onnx_raw))),
            'score_flat_corr': flat_corr(scores_official_raw, scores_onnx_raw),
            'emb_mae': float(np.mean(np.abs(emb_official - emb_onnx))),
            'emb_max_abs': float(np.max(np.abs(emb_official - emb_onnx))),
            'emb_flat_corr': flat_corr(emb_official, emb_onnx),
            'top1_agreement_raw': float((scores_official_raw.argmax(axis=1) == scores_onnx_raw.argmax(axis=1)).mean()),
            f'top{CFG.topk_compare}_overlap_raw': topk_overlap_ratio(scores_official_raw, scores_onnx_raw, k=CFG.topk_compare),
            'official_vs_teacher_raw_mae': float(np.mean(np.abs(scores_official_raw - TEACHER_RAW))),
        }
        emb_cos_mean, emb_cos_min, emb_cos_max = mean_cosine(emb_official, emb_onnx)
        raw_metrics['emb_cosine_mean'] = emb_cos_mean
        raw_metrics['emb_cosine_min'] = emb_cos_min
        raw_metrics['emb_cosine_max'] = emb_cos_max

        save_json(raw_metrics, OUTPUT_DIR / 'raw_compat_metrics.json')
        display(pd.DataFrame([raw_metrics]).T)
        """
    ),
    code_cell(
        """
        class SelectiveSSM(nn.Module):
            def __init__(self, d_model, d_state=16, d_conv=4):
                super().__init__()
                self.d_model = d_model
                self.d_state = d_state
                self.in_proj = nn.Linear(d_model, 2 * d_model, bias=False)
                self.conv1d = nn.Conv1d(d_model, d_model, d_conv, padding=d_conv - 1, groups=d_model)
                self.dt_proj = nn.Linear(d_model, d_model, bias=True)
                A = torch.arange(1, d_state + 1, dtype=torch.float32)
                A = A.unsqueeze(0).expand(d_model, -1)
                self.A_log = nn.Parameter(torch.log(A))
                self.D = nn.Parameter(torch.ones(d_model))
                self.B_proj = nn.Linear(d_model, d_state, bias=False)
                self.C_proj = nn.Linear(d_model, d_state, bias=False)
                self.out_proj = nn.Linear(d_model, d_model, bias=False)

            def forward(self, x):
                bsz, steps, dim = x.shape
                xz = self.in_proj(x)
                x_ssm, _ = xz.chunk(2, dim=-1)
                x_conv = self.conv1d(x_ssm.transpose(1, 2))[:, :, :steps].transpose(1, 2)
                x_conv = F.silu(x_conv)
                dt = F.softplus(self.dt_proj(x_conv))
                A = -torch.exp(self.A_log)
                B = self.B_proj(x_conv)
                C = self.C_proj(x_conv)
                h = torch.zeros(bsz, dim, self.d_state, device=x.device)
                ys = []
                for t in range(steps):
                    dt_t = dt[:, t, :]
                    dA = torch.exp(A[None, :, :] * dt_t[:, :, None])
                    dB = dt_t[:, :, None] * B[:, t, None, :]
                    h = h * dA + x[:, t, :, None] * dB
                    y_t = (h * C[:, t, None, :]).sum(-1)
                    ys.append(y_t)
                y = torch.stack(ys, dim=1)
                return y + x * self.D[None, None, :]


        class TemporalCrossAttention(nn.Module):
            def __init__(self, d_model, n_heads=4, dropout=0.1):
                super().__init__()
                self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
                self.norm = nn.LayerNorm(d_model)
                self.ffn = nn.Sequential(
                    nn.Linear(d_model, d_model * 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model * 2, d_model),
                    nn.Dropout(dropout),
                )
                self.norm2 = nn.LayerNorm(d_model)

            def forward(self, x):
                residual = x
                x = self.norm(x)
                attn_out, _ = self.attn(x, x, x)
                x = residual + attn_out
                residual = x
                x = self.norm2(x)
                x = residual + self.ffn(x)
                return x


        class ProtoSSMv2(nn.Module):
            def __init__(self, d_input=1536, d_model=192, d_state=16,
                         n_ssm_layers=2, n_classes=234, n_windows=12,
                         dropout=0.2, n_sites=20, meta_dim=16,
                         use_cross_attn=True, cross_attn_heads=4):
                super().__init__()
                self.d_model = d_model
                self.n_classes = n_classes
                self.n_windows = n_windows
                self.input_proj = nn.Sequential(
                    nn.Linear(d_input, d_model),
                    nn.LayerNorm(d_model),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                self.pos_enc = nn.Parameter(torch.randn(1, n_windows, d_model) * 0.02)
                self.site_emb = nn.Embedding(n_sites, meta_dim)
                self.hour_emb = nn.Embedding(24, meta_dim)
                self.meta_proj = nn.Linear(2 * meta_dim, d_model)
                self.ssm_fwd = nn.ModuleList([SelectiveSSM(d_model, d_state) for _ in range(n_ssm_layers)])
                self.ssm_bwd = nn.ModuleList([SelectiveSSM(d_model, d_state) for _ in range(n_ssm_layers)])
                self.ssm_merge = nn.ModuleList([nn.Linear(2 * d_model, d_model) for _ in range(n_ssm_layers)])
                self.ssm_norm = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_ssm_layers)])
                self.ssm_drop = nn.ModuleList([nn.Dropout(dropout) for _ in range(n_ssm_layers)])
                self.use_cross_attn = use_cross_attn
                if use_cross_attn:
                    self.cross_attn = TemporalCrossAttention(d_model, n_heads=cross_attn_heads, dropout=dropout)
                self.prototypes = nn.Parameter(torch.randn(n_classes, d_model) * 0.02)
                self.proto_temp = nn.Parameter(torch.tensor(5.0))
                self.class_bias = nn.Parameter(torch.zeros(n_classes))
                self.fusion_alpha = nn.Parameter(torch.zeros(n_classes))
                self.n_families = 0
                self.family_head = None

            def init_family_head(self, n_families, class_to_family):
                self.n_families = n_families
                self.family_head = nn.Linear(self.d_model, n_families)
                self.register_buffer('class_to_family', torch.tensor(class_to_family, dtype=torch.long))

            def forward(self, emb, logits, site_ids=None, hours=None):
                bsz, steps, _ = emb.shape
                h = self.input_proj(emb)
                h = h + self.pos_enc[:, :steps]
                if site_ids is not None and hours is not None:
                    meta = torch.cat([self.site_emb(site_ids), self.hour_emb(hours)], dim=-1)
                    h = h + self.meta_proj(meta)[:, None, :]
                for fwd, bwd, merge, norm, drop in zip(self.ssm_fwd, self.ssm_bwd, self.ssm_merge, self.ssm_norm, self.ssm_drop):
                    hf = fwd(h)
                    hb = torch.flip(bwd(torch.flip(h, dims=[1])), dims=[1])
                    h = h + drop(merge(torch.cat([hf, hb], dim=-1)))
                    h = norm(h)
                if self.use_cross_attn:
                    h = self.cross_attn(h)
                proto = F.normalize(self.prototypes, dim=-1)
                h_norm = F.normalize(h, dim=-1)
                proto_logits = self.proto_temp * torch.einsum('btd,cd->btc', h_norm, proto) + self.class_bias
                alpha = torch.sigmoid(self.fusion_alpha)[None, None, :]
                fused = (1.0 - alpha) * proto_logits + alpha * logits
                return fused, proto_logits, None


        class ResidualSSM(nn.Module):
            def __init__(self, d_input=1536, d_scores=234, d_model=64, d_state=8,
                         n_classes=234, n_windows=12, dropout=0.2, n_sites=20, meta_dim=8):
                super().__init__()
                self.emb_proj = nn.Linear(d_input, d_model)
                self.score_proj = nn.Linear(d_scores, d_model)
                self.pos_enc = nn.Parameter(torch.randn(1, n_windows, d_model) * 0.02)
                self.site_emb = nn.Embedding(n_sites, meta_dim)
                self.hour_emb = nn.Embedding(24, meta_dim)
                self.meta_proj = nn.Linear(2 * meta_dim, d_model)
                self.block = SelectiveSSM(d_model, d_state)
                self.norm = nn.LayerNorm(d_model)
                self.head = nn.Linear(d_model, n_classes)
                self.drop = nn.Dropout(dropout)

            def forward(self, emb, first_pass, site_ids=None, hours=None):
                h = self.emb_proj(emb) + self.score_proj(first_pass) + self.pos_enc[:, :emb.shape[1]]
                if site_ids is not None and hours is not None:
                    meta = torch.cat([self.site_emb(site_ids), self.hour_emb(hours)], dim=-1)
                    h = h + self.meta_proj(meta)[:, None, :]
                h = self.norm(h + self.drop(self.block(h)))
                return self.head(h)


        def seq_features_1d(v):
            assert len(v) % N_WINDOWS == 0, 'Expected full-file blocks'
            x = v.reshape(-1, N_WINDOWS)
            prev_v = np.concatenate([x[:, :1], x[:, :-1]], axis=1).reshape(-1)
            next_v = np.concatenate([x[:, 1:], x[:, -1:]], axis=1).reshape(-1)
            mean_v = np.repeat(x.mean(axis=1), N_WINDOWS)
            max_v = np.repeat(x.max(axis=1), N_WINDOWS)
            std_v = np.repeat(x.std(axis=1), N_WINDOWS)
            return prev_v, next_v, mean_v, max_v, std_v


        def build_class_features(emb_proj, raw_col, prior_col, base_col):
            prev_base, next_base, mean_base, max_base, std_base = seq_features_1d(base_col)
            diff_mean = base_col - mean_base
            diff_prev = base_col - prev_base
            diff_next = base_col - next_base
            feats = np.concatenate([
                emb_proj,
                raw_col[:, None],
                prior_col[:, None],
                base_col[:, None],
                prev_base[:, None],
                next_base[:, None],
                mean_base[:, None],
                max_base[:, None],
                std_base[:, None],
                diff_mean[:, None],
                diff_prev[:, None],
                diff_next[:, None],
                (raw_col * prior_col)[:, None],
                (raw_col * base_col)[:, None],
                (prior_col * base_col)[:, None],
            ], axis=1)
            return feats.astype(np.float32, copy=False)


        def reshape_to_files(flat_array, meta_df):
            filenames = meta_df['filename'].to_numpy()
            unique_files = []
            seen = set()
            for fname in filenames:
                if fname not in seen:
                    unique_files.append(fname)
                    seen.add(fname)
            n_files = len(unique_files)
            assert len(flat_array) == n_files * N_WINDOWS, f'Expected {n_files * N_WINDOWS} rows, got {len(flat_array)}'
            new_shape = (n_files, N_WINDOWS) + flat_array.shape[1:]
            return flat_array.reshape(new_shape), unique_files


        def get_file_metadata(meta_df, file_list, site_to_idx, n_sites_max):
            file_to_row = {}
            for idx, fname in enumerate(meta_df['filename'].to_numpy()):
                if fname not in file_to_row:
                    file_to_row[fname] = idx
            site_ids = np.zeros(len(file_list), dtype=np.int64)
            hour_ids = np.zeros(len(file_list), dtype=np.int64)
            sites = meta_df['site'].to_numpy()
            hours = meta_df['hour_utc'].to_numpy()
            for fi, fname in enumerate(file_list):
                row = file_to_row.get(fname)
                if row is not None:
                    site_ids[fi] = min(site_to_idx.get(str(sites[row]), 0), n_sites_max - 1)
                    hour_ids[fi] = int(hours[row]) % 24
            return site_ids, hour_ids


        BEST = {
            'lambda_event': 0.45,
            'lambda_texture': 1.1,
            'lambda_proxy_texture': 0.9,
            'smooth_texture': 0.35,
            'smooth_event': 0.15,
        }


        ACTIVE_CLASSES = [PRIMARY_LABELS[i] for i in np.where(Y_TRUE.sum(axis=0) > 0)[0]]
        label_to_idx = {c: i for i, c in enumerate(PRIMARY_LABELS)}
        idx_mapped_active_texture = np.array([label_to_idx[c] for c in ACTIVE_CLASSES if CLASS_NAME_MAP.get(c) in TEXTURE_TAXA], dtype=np.int32)
        idx_mapped_active_event = np.array([label_to_idx[c] for c in ACTIVE_CLASSES if CLASS_NAME_MAP.get(c) not in TEXTURE_TAXA], dtype=np.int32)
        idx_selected_proxy_active_texture = np.array([], dtype=np.int32)
        idx_selected_prioronly_active_texture = np.array([], dtype=np.int32)
        idx_selected_prioronly_active_event = np.array([], dtype=np.int32)
        idx_unmapped_inactive = np.array([], dtype=np.int32)


        def prior_logits_from_tables(sites, hours, tables, eps=1e-4):
            n = len(sites)
            p = np.repeat(tables['global_p'][None, :], n, axis=0).astype(np.float32, copy=True)
            site_idx = np.fromiter((tables['site_to_i'].get(str(s), -1) for s in sites), dtype=np.int32, count=n)
            hour_idx = np.fromiter((tables['hour_to_i'].get(int(h), -1) if int(h) >= 0 else -1 for h in hours), dtype=np.int32, count=n)
            sh_idx = np.fromiter((tables['sh_to_i'].get((str(s), int(h)), -1) if int(h) >= 0 else -1 for s, h in zip(sites, hours)), dtype=np.int32, count=n)

            valid = hour_idx >= 0
            if valid.any():
                nh = tables['hour_n'][hour_idx[valid]][:, None]
                wh = nh / (nh + 8.0)
                p[valid] = wh * tables['hour_p'][hour_idx[valid]] + (1.0 - wh) * p[valid]

            valid = site_idx >= 0
            if valid.any():
                ns = tables['site_n'][site_idx[valid]][:, None]
                ws = ns / (ns + 8.0)
                p[valid] = ws * tables['site_p'][site_idx[valid]] + (1.0 - ws) * p[valid]

            valid = sh_idx >= 0
            if valid.any():
                nsh = tables['sh_n'][sh_idx[valid]][:, None]
                wsh = nsh / (nsh + 4.0)
                p[valid] = wsh * tables['sh_p'][sh_idx[valid]] + (1.0 - wsh) * p[valid]

            np.clip(p, eps, 1.0 - eps, out=p)
            return (np.log(p) - np.log1p(-p)).astype(np.float32, copy=False)


        def smooth_cols_fixed12(scores, cols, alpha=0.35):
            if alpha <= 0 or len(cols) == 0:
                return scores.copy()
            s = scores.copy()
            view = s.reshape(-1, N_WINDOWS, s.shape[1])
            x = view[:, :, cols]
            prev_x = np.concatenate([x[:, :1, :], x[:, :-1, :]], axis=1)
            next_x = np.concatenate([x[:, 1:, :], x[:, -1:, :]], axis=1)
            view[:, :, cols] = (1.0 - alpha) * x + 0.5 * alpha * (prev_x + next_x)
            return s


        def smooth_events_fixed12(scores, cols, alpha=0.15):
            if alpha <= 0 or len(cols) == 0:
                return scores.copy()
            s = scores.copy()
            view = s.reshape(-1, N_WINDOWS, s.shape[1])
            x = view[:, :, cols]
            prev_x = np.concatenate([x[:, :1, :], x[:, :-1, :]], axis=1)
            next_x = np.concatenate([x[:, 1:, :], x[:, -1:, :]], axis=1)
            local_max = np.maximum(x, np.maximum(prev_x, next_x))
            view[:, :, cols] = (1.0 - alpha) * x + alpha * local_max
            return s


        def fuse_scores_with_tables(base_scores, sites, hours, tables):
            scores = base_scores.copy()
            prior = prior_logits_from_tables(sites, hours, tables)

            if len(idx_mapped_active_event):
                scores[:, idx_mapped_active_event] += BEST['lambda_event'] * prior[:, idx_mapped_active_event]
            if len(idx_mapped_active_texture):
                scores[:, idx_mapped_active_texture] += BEST['lambda_texture'] * prior[:, idx_mapped_active_texture]
            if len(idx_selected_proxy_active_texture):
                scores[:, idx_selected_proxy_active_texture] += BEST['lambda_proxy_texture'] * prior[:, idx_selected_proxy_active_texture]
            if len(idx_selected_prioronly_active_event):
                scores[:, idx_selected_prioronly_active_event] = prior[:, idx_selected_prioronly_active_event]
            if len(idx_selected_prioronly_active_texture):
                scores[:, idx_selected_prioronly_active_texture] = prior[:, idx_selected_prioronly_active_texture]
            if len(idx_unmapped_inactive):
                scores[:, idx_unmapped_inactive] = prior[:, idx_unmapped_inactive]

            scores = smooth_cols_fixed12(scores, idx_mapped_active_texture, alpha=BEST['smooth_texture'])
            scores = smooth_events_fixed12(scores, idx_mapped_active_event, alpha=BEST['smooth_event'])
            return scores.astype(np.float32), prior.astype(np.float32)


        def sigmoid(x):
            return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


        def file_level_confidence_scale(preds, top_k=2):
            view = preds.reshape(-1, N_WINDOWS, preds.shape[1])
            sorted_view = np.sort(view, axis=1)
            top_k_mean = sorted_view[:, -top_k:, :].mean(axis=1, keepdims=True)
            scaled = view * top_k_mean
            return np.clip(scaled.reshape(preds.shape), 0.0, 1.0)


        def rank_aware_scaling(scores, power=0.5):
            view = scores.reshape(-1, N_WINDOWS, scores.shape[1])
            file_max = view.max(axis=1, keepdims=True)
            scale = np.power(np.clip(file_max, 0.0, 1.0), power)
            scaled = view * scale
            return np.clip(scaled.reshape(scores.shape), 0.0, 1.0)


        def adaptive_delta_smooth(scores, base_alpha=0.20):
            result = scores.copy().reshape(-1, N_WINDOWS, scores.shape[1])
            original = scores.reshape(-1, N_WINDOWS, scores.shape[1])
            for i in range(1, N_WINDOWS - 1):
                conf = original[:, i, :].max(axis=-1, keepdims=True)
                a = base_alpha * (1.0 - conf)
                neighbor_avg = (original[:, i - 1, :] + original[:, i + 1, :]) / 2.0
                result[:, i, :] = (1.0 - a) * original[:, i, :] + a * neighbor_avg
            return np.clip(result.reshape(scores.shape), 0.0, 1.0)


        def apply_per_class_thresholds(scores, thresholds):
            scaled = np.copy(scores)
            for c in range(scores.shape[1]):
                t = float(thresholds[c])
                mask_above = scores[:, c] > t
                scaled[mask_above, c] = 0.5 + 0.5 * (scores[mask_above, c] - t) / (1 - t + 1e-8)
                scaled[~mask_above, c] = 0.5 * scores[~mask_above, c] / (t + 1e-8)
            return np.clip(scaled, 0.0, 1.0)


        def apply_file_level_calibrators(scores, calibrators):
            scores_files = scores.reshape(-1, N_WINDOWS, scores.shape[1]).copy()
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
        """
    ),
    code_cell(
        """
        manifest = json.loads((ARTIFACTS_DIR / 'artifacts_manifest.json').read_text())
        with open(ARTIFACTS_DIR / manifest['artifact_files']['prior_tables'], 'rb') as f:
            final_prior_tables = pickle.load(f)
        with open(ARTIFACTS_DIR / manifest['artifact_files']['sklearn'], 'rb') as f:
            sk_artifacts = pickle.load(f)

        emb_scaler = sk_artifacts['emb_scaler']
        emb_pca = sk_artifacts['emb_pca']
        probe_models = sk_artifacts['probe_models']
        BEST_PROBE = manifest['best_probe']
        ENSEMBLE_WEIGHT_PROTO = float(manifest['ensemble_weight_proto'])
        CORRECTION_WEIGHT = float(manifest.get('correction_weight', 0.0))
        site_to_idx = {str(k): int(v) for k, v in manifest['site_to_idx'].items()}
        class_to_family = [int(x) for x in manifest['class_to_family']]
        n_families = int(manifest['n_families'])
        n_sites_cfg = int(manifest['n_sites_cfg'])
        POST_CFG = {
            'temperature': manifest['temperature'],
            'file_level_top_k': int(manifest.get('file_level_top_k', 0)),
            'tta_shifts': [int(x) for x in manifest.get('tta_shifts', [0])],
            'rank_aware_scale': bool(manifest.get('rank_aware_scale', False)),
            'rank_aware_power': float(manifest.get('rank_aware_power', 0.5)),
            'delta_shift_alpha': float(manifest.get('delta_shift_alpha', 0.0)),
        }

        threshold_rel = manifest['artifact_files'].get('thresholds', 'per_class_thresholds.npy')
        PER_CLASS_THRESHOLDS = np.load(ARTIFACTS_DIR / threshold_rel).astype(np.float32)

        calibrators_rel = manifest['artifact_files'].get('calibrators')
        CALIBRATORS = None
        if calibrators_rel:
            cal_path = ARTIFACTS_DIR / calibrators_rel
            if cal_path.exists():
                with open(cal_path, 'rb') as f:
                    CALIBRATORS = pickle.load(f)

        proto_ckpt = torch.load(ARTIFACTS_DIR / manifest['artifact_files']['proto'], map_location=DEVICE)
        proto_cfg = proto_ckpt['proto_ssm']
        model = ProtoSSMv2(
            d_input=int(emb_official.shape[1]),
            d_model=proto_cfg['d_model'],
            d_state=proto_cfg['d_state'],
            n_ssm_layers=proto_cfg['n_ssm_layers'],
            n_classes=int(manifest['n_classes']),
            n_windows=int(manifest['n_windows']),
            dropout=proto_cfg['dropout'],
            n_sites=proto_cfg['n_sites'],
            meta_dim=proto_cfg['meta_dim'],
            use_cross_attn=proto_cfg.get('use_cross_attn', True),
            cross_attn_heads=proto_cfg.get('cross_attn_heads', 4),
        ).to(DEVICE)
        model.init_family_head(n_families, class_to_family)
        model.load_state_dict(proto_ckpt['state_dict'], strict=True)
        model.eval()

        res_model = None
        if manifest.get('has_residual', False) and manifest['artifact_files'].get('residual'):
            res_ckpt = torch.load(ARTIFACTS_DIR / manifest['artifact_files']['residual'], map_location=DEVICE)
            res_cfg = res_ckpt['residual_ssm']
            res_model = ResidualSSM(
                d_input=int(emb_official.shape[1]),
                d_scores=int(manifest['n_classes']),
                d_model=res_cfg['d_model'],
                d_state=res_cfg['d_state'],
                n_classes=int(manifest['n_classes']),
                n_windows=int(manifest['n_windows']),
                dropout=res_cfg['dropout'],
                n_sites=proto_cfg['n_sites'],
                meta_dim=8,
            ).to(DEVICE)
            res_model.load_state_dict(res_ckpt['state_dict'], strict=True)
            res_model.eval()

        print('Loaded fixed exp_015d artifacts.')
        print('  Proto params:', manifest.get('proto_parameters'))
        print('  Residual enabled:', res_model is not None)
        print('  Probe models:', len(probe_models))
        print('  Threshold mean:', float(PER_CLASS_THRESHOLDS.mean()))
        print('  Calibrators present:', CALIBRATORS is not None)
        """
    ),
    code_cell(
        """
        BATCH_FILES = 256 if DEVICE.type == 'cuda' else 64


        def build_tempered_probs(logit_scores):
            class_temperatures = np.ones(N_CLASSES, dtype=np.float32) * float(POST_CFG['temperature']['aves'])
            for ci, label in enumerate(PRIMARY_LABELS):
                if CLASS_NAME_MAP.get(label, 'Aves') in TEXTURE_TAXA:
                    class_temperatures[ci] = float(POST_CFG['temperature']['texture'])
            return sigmoid(logit_scores / class_temperatures[None, :]).astype(np.float32)


        def proto_forward_batched(model, emb_files, logits_files, site_ids_all, hours_all, batch_files):
            outputs = []
            with torch.no_grad():
                for start in range(0, len(emb_files), batch_files):
                    end = min(start + batch_files, len(emb_files))
                    emb_tensor = to_device_tensor(emb_files[start:end], torch.float32)
                    logits_tensor = to_device_tensor(logits_files[start:end], torch.float32)
                    site_tensor = to_device_tensor(site_ids_all[start:end], torch.long)
                    hour_tensor = to_device_tensor(hours_all[start:end], torch.long)
                    with autocast_context():
                        proto_out, _, _ = model(emb_tensor, logits_tensor, site_ids=site_tensor, hours=hour_tensor)
                    outputs.append(tensor_to_numpy(proto_out))
                    del emb_tensor, logits_tensor, site_tensor, hour_tensor, proto_out
                    if DEVICE.type == 'cuda':
                        torch.cuda.empty_cache()
            return np.concatenate(outputs, axis=0)


        def temporal_shift_tta_batched(emb_files, logits_files, model, site_ids, hours, shifts, batch_files):
            all_preds = []
            for shift in shifts:
                if shift == 0:
                    emb_shifted = emb_files
                    logit_shifted = logits_files
                else:
                    emb_shifted = np.roll(emb_files, shift, axis=1)
                    logit_shifted = np.roll(logits_files, shift, axis=1)
                pred = proto_forward_batched(model, emb_shifted, logit_shifted, site_ids, hours, batch_files)
                if shift != 0:
                    pred = np.roll(pred, -shift, axis=1)
                all_preds.append(pred)
            return np.mean(all_preds, axis=0)


        def residual_forward_batched(res_model, emb_files, first_pass_files, site_ids_all, hours_all, batch_files):
            outputs = []
            with torch.no_grad():
                for start in range(0, len(emb_files), batch_files):
                    end = min(start + batch_files, len(emb_files))
                    emb_tensor = to_device_tensor(emb_files[start:end], torch.float32)
                    first_pass_tensor = to_device_tensor(first_pass_files[start:end], torch.float32)
                    site_tensor = to_device_tensor(site_ids_all[start:end], torch.long)
                    hour_tensor = to_device_tensor(hours_all[start:end], torch.long)
                    with autocast_context():
                        correction = res_model(emb_tensor, first_pass_tensor, site_ids=site_tensor, hours=hour_tensor)
                    outputs.append(tensor_to_numpy(correction))
                    del emb_tensor, first_pass_tensor, site_tensor, hour_tensor, correction
                    if DEVICE.type == 'cuda':
                        torch.cuda.empty_cache()
            return np.concatenate(outputs, axis=0)


        def replay_cache(cache_name: str, meta_df: pd.DataFrame, scores_full_raw: np.ndarray, emb_full: np.ndarray) -> dict[str, tp.Any]:
            cache_t0 = time.time()

            emb_files, file_list = reshape_to_files(emb_full, meta_df)
            logits_files, _ = reshape_to_files(scores_full_raw, meta_df)
            site_ids_all, hours_all = get_file_metadata(meta_df, file_list, site_to_idx, n_sites_cfg)

            base_scores, prior_scores = fuse_scores_with_tables(
                scores_full_raw,
                sites=meta_df['site'].to_numpy(),
                hours=meta_df['hour_utc'].to_numpy(),
                tables=final_prior_tables,
            )

            emb_scaled = emb_scaler.transform(emb_full)
            z_full = emb_pca.transform(emb_scaled).astype(np.float32)
            del emb_scaled

            mlp_scores = base_scores.copy()
            for cls_idx, clf in probe_models.items():
                x_cls = build_class_features(
                    z_full,
                    raw_col=scores_full_raw[:, cls_idx],
                    prior_col=prior_scores[:, cls_idx],
                    base_col=base_scores[:, cls_idx],
                )
                if hasattr(clf, 'predict_proba'):
                    prob = clf.predict_proba(x_cls)[:, 1].astype(np.float32)
                    pred = np.log(prob + 1e-7) - np.log(1 - prob + 1e-7)
                else:
                    pred = clf.decision_function(x_cls).astype(np.float32)
                alpha = float(BEST_PROBE['alpha'])
                mlp_scores[:, cls_idx] = (1.0 - alpha) * base_scores[:, cls_idx] + alpha * pred

            tta_shifts = tuple(int(x) for x in POST_CFG.get('tta_shifts', [0]))
            if tta_shifts == (0,):
                proto_scores_flat = proto_forward_batched(
                    model, emb_files, logits_files, site_ids_all, hours_all, BATCH_FILES
                ).reshape(-1, N_CLASSES).astype(np.float32)
            else:
                proto_scores_flat = temporal_shift_tta_batched(
                    emb_files, logits_files, model, site_ids_all, hours_all, shifts=tta_shifts, batch_files=BATCH_FILES
                ).reshape(-1, N_CLASSES).astype(np.float32)

            final_logits = (
                ENSEMBLE_WEIGHT_PROTO * proto_scores_flat +
                (1.0 - ENSEMBLE_WEIGHT_PROTO) * mlp_scores
            ).astype(np.float32)

            if res_model is not None and CORRECTION_WEIGHT > 0:
                first_pass_files, _ = reshape_to_files(final_logits, meta_df)
                correction_flat = residual_forward_batched(
                    res_model, emb_files, first_pass_files, site_ids_all, hours_all, BATCH_FILES
                ).reshape(-1, N_CLASSES).astype(np.float32)
                final_logits = final_logits + CORRECTION_WEIGHT * correction_flat

            probs = build_tempered_probs(final_logits)
            if CALIBRATORS is not None:
                probs = apply_file_level_calibrators(probs, CALIBRATORS)
            if POST_CFG['file_level_top_k'] > 0:
                probs = file_level_confidence_scale(probs, top_k=POST_CFG['file_level_top_k'])
            if POST_CFG['rank_aware_scale']:
                probs = rank_aware_scaling(probs, power=POST_CFG['rank_aware_power'])
            if POST_CFG['delta_shift_alpha'] > 0:
                probs = adaptive_delta_smooth(probs, base_alpha=POST_CFG['delta_shift_alpha'])
            probs = apply_per_class_thresholds(probs, PER_CLASS_THRESHOLDS)

            file_probs = file_level_max(probs)
            file_y = file_level_max(Y_TRUE)
            result = {
                'cache_name': cache_name,
                'tta_shifts': list(tta_shifts),
                'raw_scores': scores_full_raw.astype(np.float32),
                'proto_scores': proto_scores_flat.astype(np.float32),
                'final_logits': final_logits.astype(np.float32),
                'final_probs': probs.astype(np.float32),
                'row_macro_auc': float(macro_auc_skip_empty(Y_TRUE, probs)),
                'file_macro_auc': float(macro_auc_skip_empty(file_y, file_probs)),
                'row_texture_auc': float(macro_auc_subset(Y_TRUE, probs, TEXTURE_IDX)),
                'file_texture_auc': float(macro_auc_subset(file_y, file_probs, TEXTURE_IDX)),
                'wall_time': float(time.time() - cache_t0),
            }
            return result


        official_replay = replay_cache('official', meta_official, scores_official_raw, emb_official)
        onnx_replay = replay_cache('onnx', meta_onnx, scores_onnx_raw, emb_onnx)

        print('Official replay:', {k: official_replay[k] for k in ['row_macro_auc', 'file_macro_auc', 'row_texture_auc', 'file_texture_auc', 'wall_time']})
        print('ONNX replay:', {k: onnx_replay[k] for k in ['row_macro_auc', 'file_macro_auc', 'row_texture_auc', 'file_texture_auc', 'wall_time']})
        """
    ),
    code_cell(
        """
        downstream_metrics = {
            'official_row_macro_auc': float(official_replay['row_macro_auc']),
            'onnx_row_macro_auc': float(onnx_replay['row_macro_auc']),
            'row_macro_auc_delta_onnx_minus_official': float(onnx_replay['row_macro_auc'] - official_replay['row_macro_auc']),
            'official_file_macro_auc': float(official_replay['file_macro_auc']),
            'onnx_file_macro_auc': float(onnx_replay['file_macro_auc']),
            'file_macro_auc_delta_onnx_minus_official': float(onnx_replay['file_macro_auc'] - official_replay['file_macro_auc']),
            'official_texture_row_auc': float(official_replay['row_texture_auc']),
            'onnx_texture_row_auc': float(onnx_replay['row_texture_auc']),
            'official_texture_file_auc': float(official_replay['file_texture_auc']),
            'onnx_texture_file_auc': float(onnx_replay['file_texture_auc']),
            'final_logit_mae': float(np.mean(np.abs(official_replay['final_logits'] - onnx_replay['final_logits']))),
            'final_prob_mae': float(np.mean(np.abs(official_replay['final_probs'] - onnx_replay['final_probs']))),
            'final_prob_flat_corr': flat_corr(official_replay['final_probs'], onnx_replay['final_probs']),
            'final_top1_agreement': float((official_replay['final_probs'].argmax(axis=1) == onnx_replay['final_probs'].argmax(axis=1)).mean()),
            f'final_top{CFG.topk_compare}_overlap': topk_overlap_ratio(official_replay['final_probs'], onnx_replay['final_probs'], k=CFG.topk_compare),
            'official_vs_teacher_logit_mae': float(np.mean(np.abs(official_replay['final_logits'] - TEACHER_LOGITS))),
            'official_vs_teacher_prob_mae': float(np.mean(np.abs(official_replay['final_probs'] - TEACHER_PROBS))),
            'official_replay_seconds': float(official_replay['wall_time']),
            'onnx_replay_seconds': float(onnx_replay['wall_time']),
        }

        classwise_df = classwise_auc_df(Y_TRUE, official_replay['final_probs'], onnx_replay['final_probs'], label_a='official', label_b='onnx')
        taxon_df = taxon_summary_df(classwise_df, label_a='official', label_b='onnx')

        report_snapshot = {
            'experiment_id': CFG.experiment_id,
            'experiment_name': CFG.experiment_name,
            'rows': int(len(teacher_meta)),
            'files': int(teacher_meta['filename'].nunique()),
            'n_classes': int(N_CLASSES),
            'raw_metrics': raw_metrics,
            'downstream_metrics': downstream_metrics,
            'interpretation_anchor': (
                'Low drift + near-zero downstream delta means ONNX Perch is likely engineering-safe. '
                'Large drift or negative downstream delta means the swap is unsafe for the current exp_015d artifact stack.'
            ),
        }

        classwise_df.to_csv(OUTPUT_DIR / 'classwise_auc_comparison.csv', index=False)
        taxon_df.to_csv(OUTPUT_DIR / 'taxon_summary.csv', index=False)
        save_json(downstream_metrics, OUTPUT_DIR / 'downstream_metrics.json')
        save_json(report_snapshot, OUTPUT_DIR / 'report_snapshot.json')

        display(pd.DataFrame([downstream_metrics]).T)
        display(taxon_df)
        display(classwise_df.sort_values('row_auc_delta').head(12))
        """
    ),
]


nb = {
    "cells": cells,
    "metadata": base_md,
    "nbformat": 4,
    "nbformat_minor": 5,
}


output_path = NOTEBOOKS / "exp_029a_perch_onnx_compat_benchmark.ipynb"
output_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print(output_path)
