from __future__ import annotations

import copy
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
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": lines(text),
    }


def code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines(text),
    }


base_md = load_nb(NOTEBOOKS / "exp_011_hgnetv2_soundscape_supervised.ipynb")["metadata"]
exp027b = load_nb(NOTEBOOKS / "exp_027b_hgnetv2_soundscape_distill_from_exp015d.ipynb")


def write_nb(name: str, cells: list[dict]) -> None:
    payload = {
        "cells": cells,
        "metadata": copy.deepcopy(base_md),
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (NOTEBOOKS / name).write_text(json.dumps(payload, ensure_ascii=False, indent=1))


config_cell = """
from __future__ import annotations

import gc
import json
import os
import random
import typing as tp
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

try:
    from IPython.display import display
except Exception:
    def display(obj: object) -> None:
        print(obj)


def find_repo_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / 'PROJECT_STATE.md').exists() and (candidate / 'data').exists():
            return candidate
    raise FileNotFoundError('Could not resolve repository root')


ROOT = find_repo_root()
DATA = ROOT / 'data' / 'birdclef-2026'
EXPERIMENTS = ROOT / 'experiments'
EXP011_DIR = EXPERIMENTS / 'outputs' / 'exp_011_hgnetv2_soundscape_supervised'
TEACHER_CACHE_DIR = None         # optional explicit path to completed exp_027a outputs


@dataclass
class Config:
    experiment_id: str = 'exp_031b'
    experiment_name: str = 'overlap_ablation'
    fold: int = 0
    n_folds: int = 4
    random_seed: int = 1098

    sample_rate: int = 32_000
    segment_seconds: float = 5.0
    overlap_hop_seconds: float = 2.5
    min_overlap_seconds: float = 0.50
    n_fft: int = 2048
    win_length: int = 626
    hop_length: int = 313
    f_min: int = 20
    n_mels: int = 256
    top_db: float = 80.0
    image_size: tuple[int, int] = (256, 256)

    model_name: str = 'hgnetv2_b0.ssld_stage2_ft_in1k'
    pretrained: bool = True
    drop_path_rate: float = 0.0

    epochs: int = 8
    warmup_epochs: int = 2
    batch_size: int = 16
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    num_workers: int = 0
    use_amp: bool = True
    distill_weight: float = 0.25
    distill_temperature: float = 1.0
    teacher_weight_power: float = 1.0
    min_teacher_confidence_train: float = 0.0

    regime_names: tuple[str, ...] = ('base_only', 'base_plus_overlap', 'overlap_only')
    overlap_eval_on_base_only: bool = True

    max_train_rows: int | None = None
    max_valid_rows: int | None = None

    selection_metric: str = 'soundscape_macro_auc'
    save_every_epoch: bool = True


CFG = Config()
RUN_TRAINING = True
RESUME_TRAINING = True


def has_teacher_cache(path: Path) -> bool:
    return (path / 'teacher_meta.parquet').exists() and (path / 'teacher_outputs.npz').exists()


def resolve_teacher_dir(explicit: str | None = None) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    candidates.extend([
        EXPERIMENTS / 'outputs' / 'exp_027a_exp015d_teacher_cache',
        EXPERIMENTS / 'outputs' / 'exp_027a_exp015d_teacher_cache' / f'fold_{CFG.fold:02d}',
        ROOT / 'outputs' / 'exp_027a_exp015d_teacher_cache',
        Path.home() / 'Downloads' / 'exp_027a_exp015d_teacher_cache',
        Path.home() / 'Downloads' / 'exp_027a_exp015d_teacher_cache' / f'fold_{CFG.fold:02d}',
    ])
    for candidate in candidates:
        if has_teacher_cache(candidate):
            return candidate

    search_roots = [
        EXPERIMENTS / 'outputs',
        Path.home() / 'Downloads',
    ]
    for search_root in search_roots:
        if not search_root.exists():
            continue
        for meta_path in search_root.rglob('teacher_meta.parquet'):
            parent = meta_path.parent
            if has_teacher_cache(parent) and 'exp_027a' in str(parent):
                return parent

    raise FileNotFoundError(
        'Could not resolve exp_027a teacher cache. '
        'Run exp_027a first and make sure it writes teacher_meta.parquet + teacher_outputs.npz, '
        'or set TEACHER_CACHE_DIR to that completed output folder.'
    )


TEACHER_DIR = resolve_teacher_dir(TEACHER_CACHE_DIR)
OUTPUT_DIR = EXPERIMENTS / 'outputs' / f'{CFG.experiment_id}_{CFG.experiment_name}' / f'fold_{CFG.fold:02d}'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
"""


seed_device_cell = "".join(exp027b["cells"][4]["source"])


data_prep_cell = """
taxonomy = pd.read_csv(DATA / 'taxonomy.csv')
CLASSES = taxonomy['primary_label'].astype(str).tolist()
N_CLASSES = len(CLASSES)
label2idx = {label: idx for idx, label in enumerate(CLASSES)}


def resolve_soundscape_path(filename: str, *raw_paths: object) -> str:
    candidates: list[Path] = []
    for raw in raw_paths:
        if raw is None:
            continue
        raw_str = str(raw)
        if raw_str in {'', 'None', 'nan', '<NA>'}:
            continue
        path = Path(raw_str).expanduser()
        candidates.append(path)
        candidates.append(Path(raw_str.replace('/kaggle/input/competitions/birdclef-2026', str(DATA))))
        candidates.append(Path(raw_str.replace('/kaggle/input/birdclef-2026', str(DATA))))
    candidates.append(DATA / 'train_soundscapes' / filename)
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(DATA / 'train_soundscapes' / filename)


teacher_meta = pd.read_parquet(TEACHER_DIR / 'teacher_meta.parquet')
teacher_arr = np.load(TEACHER_DIR / 'teacher_outputs.npz')
teacher_logits = teacher_arr['teacher_logits'].astype(np.float32)
teacher_probs = teacher_arr['teacher_probs'].astype(np.float32)
teacher_labels = teacher_arr['labels'].astype(np.float32)

assert len(teacher_meta) == teacher_logits.shape[0] == teacher_probs.shape[0] == teacher_labels.shape[0]
if 'fold' not in teacher_meta.columns:
    raise KeyError('Teacher cache is missing fold assignments')

teacher_meta = teacher_meta.copy()
source_file_values = teacher_meta['source_file_path'].tolist() if 'source_file_path' in teacher_meta.columns else [None] * len(teacher_meta)
teacher_meta['file_path'] = [
    resolve_soundscape_path(filename, file_path, source_file_path)
    for filename, file_path, source_file_path in zip(
        teacher_meta['filename'].tolist(),
        teacher_meta['file_path'].tolist(),
        source_file_values,
    )
]
if 'source_file_path' in teacher_meta.columns:
    teacher_meta['source_file_path'] = [
        resolve_soundscape_path(filename, source_file_path, file_path)
        for filename, source_file_path, file_path in zip(
            teacher_meta['filename'].tolist(),
            source_file_values,
            teacher_meta['file_path'].tolist(),
        )
    ]

missing_paths = [path for path in teacher_meta['file_path'].tolist() if not Path(path).exists()]
if missing_paths:
    raise FileNotFoundError(
        f'Failed to resolve {len(missing_paths)} teacher soundscape paths. '
        f'First missing path: {missing_paths[0]}'
    )

teacher_meta['start_sec'] = teacher_meta['start_sec'].astype(np.float32)
teacher_meta['end_sec'] = teacher_meta['end_sec'].astype(np.float32)
teacher_meta['clip_start_frame'] = np.round(teacher_meta['start_sec'] * CFG.sample_rate).astype(int)
teacher_meta['clip_end_frame'] = np.round(teacher_meta['end_sec'] * CFG.sample_rate).astype(int)
teacher_meta['source_kind'] = 'base_5s'
if 'teacher_confidence' not in teacher_meta.columns:
    teacher_meta['teacher_confidence'] = teacher_probs.max(axis=1).astype(np.float32)


def build_overlap_manifest(
    frame: pd.DataFrame,
    hard_labels_arr: np.ndarray,
    teacher_probs_arr: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    records: list[dict[str, tp.Any]] = []
    soft_label_parts: list[np.ndarray] = []
    soft_teacher_parts: list[np.ndarray] = []

    for filename, group in frame.groupby('filename', sort=False):
        group = group.sort_values('start_sec').reset_index(drop=True)
        if group.empty:
            continue
        fold_values = group['fold'].dropna().unique().tolist()
        if len(fold_values) != 1:
            raise ValueError(f'Expected a single fold per file for {filename}, got {fold_values}')

        row_indices = group['teacher_row_index'].to_numpy(dtype=np.int64)
        group_labels = hard_labels_arr[row_indices]
        group_teacher = teacher_probs_arr[row_indices]
        base_start = group['start_sec'].to_numpy(dtype=np.float32)
        base_end = group['end_sec'].to_numpy(dtype=np.float32)
        base_conf = group['teacher_confidence'].to_numpy(dtype=np.float32)
        max_end = float(base_end.max())
        if max_end < CFG.segment_seconds:
            continue

        cursor = 0.0
        while cursor <= max_end - CFG.segment_seconds + 1e-6:
            if np.any(np.isclose(base_start, cursor, atol=1e-4)):
                cursor += CFG.overlap_hop_seconds
                continue

            end_sec = cursor + CFG.segment_seconds
            overlap = np.maximum(0.0, np.minimum(base_end, end_sec) - np.maximum(base_start, cursor))
            overlap_total = float(overlap.sum())
            if overlap_total < CFG.min_overlap_seconds:
                cursor += CFG.overlap_hop_seconds
                continue

            weights = (overlap / overlap_total).astype(np.float32)
            soft_label = (group_labels * weights[:, None]).sum(axis=0).astype(np.float32)
            soft_teacher = (group_teacher * weights[:, None]).sum(axis=0).astype(np.float32)
            teacher_conf = float(np.dot(base_conf, weights))

            head = group.iloc[0]
            row_id = f"{Path(filename).stem}_ov_{int(round(cursor * 10)):04d}"
            records.append({
                'row_id': row_id,
                'filename': filename,
                'audio_id': Path(filename).stem,
                'file_path': str(head['file_path']),
                'source_file_path': str(head['source_file_path']) if 'source_file_path' in group.columns else str(head['file_path']),
                'site': head.get('site', None),
                'hour_utc': head.get('hour_utc', None),
                'start_sec': float(cursor),
                'end_sec': float(end_sec),
                'clip_start_frame': int(round(cursor * CFG.sample_rate)),
                'clip_end_frame': int(round(end_sec * CFG.sample_rate)),
                'fold': int(fold_values[0]),
                'source_kind': 'overlap_2p5s',
                'teacher_confidence': teacher_conf,
            })
            soft_label_parts.append(soft_label)
            soft_teacher_parts.append(soft_teacher)
            cursor += CFG.overlap_hop_seconds

    if not records:
        empty = frame.iloc[0:0].copy()
        return empty, np.zeros((0, N_CLASSES), dtype=np.float32), np.zeros((0, N_CLASSES), dtype=np.float32)

    overlap_frame = pd.DataFrame(records)
    overlap_labels = np.stack(soft_label_parts, axis=0).astype(np.float32)
    overlap_teacher = np.stack(soft_teacher_parts, axis=0).astype(np.float32)
    return overlap_frame, overlap_labels, overlap_teacher


teacher_meta['teacher_row_index'] = np.arange(len(teacher_meta), dtype=np.int64)
base_train_mask = teacher_meta['fold'].to_numpy() != CFG.fold
base_valid_mask = teacher_meta['fold'].to_numpy() == CFG.fold

base_train_frame = teacher_meta.loc[base_train_mask].reset_index(drop=True)
base_valid_frame = teacher_meta.loc[base_valid_mask].reset_index(drop=True)
base_train_labels = teacher_labels[base_train_mask]
base_valid_labels = teacher_labels[base_valid_mask]
base_train_teacher_probs = teacher_probs[base_train_mask]
base_valid_teacher_probs = teacher_probs[base_valid_mask]

overlap_train_frame, overlap_train_labels, overlap_train_teacher_probs = build_overlap_manifest(
    teacher_meta.loc[base_train_mask].copy(),
    teacher_labels,
    teacher_probs,
)

def build_regime_payload(regime_name: str) -> dict[str, tp.Any]:
    if regime_name == 'base_only':
        train_frame = base_train_frame.copy()
        train_labels_fold = base_train_labels.copy()
        train_teacher_probs = base_train_teacher_probs.copy()
    elif regime_name == 'base_plus_overlap':
        train_frame = pd.concat([base_train_frame, overlap_train_frame], ignore_index=True)
        train_labels_fold = np.concatenate([base_train_labels, overlap_train_labels], axis=0).astype(np.float32)
        train_teacher_probs = np.concatenate([base_train_teacher_probs, overlap_train_teacher_probs], axis=0).astype(np.float32)
    elif regime_name == 'overlap_only':
        train_frame = overlap_train_frame.copy()
        train_labels_fold = overlap_train_labels.copy()
        train_teacher_probs = overlap_train_teacher_probs.copy()
    else:
        raise ValueError(f'Unknown regime: {regime_name}')

    valid_frame = base_valid_frame.reset_index(drop=True)
    valid_labels_fold = base_valid_labels.astype(np.float32)
    valid_teacher_probs = base_valid_teacher_probs.astype(np.float32)

    if CFG.min_teacher_confidence_train > 0:
        keep = train_frame['teacher_confidence'].to_numpy(dtype=np.float32) >= CFG.min_teacher_confidence_train
        train_frame = train_frame.loc[keep].reset_index(drop=True)
        train_labels_fold = train_labels_fold[keep]
        train_teacher_probs = train_teacher_probs[keep]

    if CFG.max_train_rows is not None:
        keep_n = min(CFG.max_train_rows, len(train_frame))
        selected = train_frame.sample(keep_n, random_state=CFG.random_seed).index.to_numpy()
        train_frame = train_frame.loc[selected].reset_index(drop=True)
        train_labels_fold = train_labels_fold[selected]
        train_teacher_probs = train_teacher_probs[selected]
    if CFG.max_valid_rows is not None:
        keep_n = min(CFG.max_valid_rows, len(valid_frame))
        selected = valid_frame.sample(keep_n, random_state=CFG.random_seed).index.to_numpy()
        valid_frame = valid_frame.loc[selected].reset_index(drop=True)
        valid_labels_fold = valid_labels_fold[selected]
        valid_teacher_probs = valid_teacher_probs[selected]

    manifest_summary = {
        'experiment_id': CFG.experiment_id,
        'fold': CFG.fold,
        'regime_name': regime_name,
        'train_rows_total': int(len(train_frame)),
        'valid_rows_total': int(len(valid_frame)),
        'train_base_rows': int((train_frame['source_kind'] == 'base_5s').sum()),
        'train_overlap_rows': int((train_frame['source_kind'] == 'overlap_2p5s').sum()),
        'valid_base_rows': int((valid_frame['source_kind'] == 'base_5s').sum()),
        'train_files': int(train_frame['filename'].nunique()),
        'valid_files': int(valid_frame['filename'].nunique()),
        'mean_train_teacher_confidence': float(train_frame['teacher_confidence'].mean()),
        'mean_valid_teacher_confidence': float(valid_frame['teacher_confidence'].mean()),
        'overlap_hop_seconds': float(CFG.overlap_hop_seconds),
    }
    return {
        'train_frame': train_frame,
        'train_labels': train_labels_fold.astype(np.float32),
        'train_teacher_probs': train_teacher_probs.astype(np.float32),
        'valid_frame': valid_frame,
        'valid_labels': valid_labels_fold.astype(np.float32),
        'valid_teacher_probs': valid_teacher_probs.astype(np.float32),
        'manifest_summary': manifest_summary,
    }


regime_payloads = {regime_name: build_regime_payload(regime_name) for regime_name in CFG.regime_names}
manifest_df = pd.DataFrame([payload['manifest_summary'] for payload in regime_payloads.values()])
manifest_df.to_csv(OUTPUT_DIR / 'regime_manifest_summary.csv', index=False)
display(manifest_df)
"""


dataset_model_cell = (
    "".join(exp027b["cells"][6]["source"])
    .replace(
        "def read_audio_region(path: str, clip_start_frame: int, clip_end_frame: int, sample_frames: int, training: bool) -> np.ndarray:\n"
        "    with sf.SoundFile(path) as snd:\n",
        "def read_audio_region(path: str, clip_start_frame: int, clip_end_frame: int, sample_frames: int, training: bool) -> np.ndarray:\n"
        "    sample_frames = int(sample_frames)\n"
        "    with sf.SoundFile(path) as snd:\n",
    )
    .replace(
        "        self.sample_frames = CFG.sample_rate * CFG.segment_seconds\n",
        "        self.sample_frames = int(round(CFG.sample_rate * CFG.segment_seconds))\n",
    )
)

train_cell = (
    "".join(exp027b["cells"][7]["source"])
    .replace(
        "def train_one_fold(\n"
        "    train_frame: pd.DataFrame,\n"
        "    valid_frame: pd.DataFrame,\n"
        "    train_labels: np.ndarray,\n"
        "    valid_labels: np.ndarray,\n"
        "    train_teacher_probs: np.ndarray,\n"
        "    valid_teacher_probs: np.ndarray,\n"
        "    output_dir: Path,\n"
        ") -> tuple[pd.DataFrame, dict[str, tp.Any]]:\n",
        "def train_one_fold(\n"
        "    regime_name: str,\n"
        "    train_frame: pd.DataFrame,\n"
        "    valid_frame: pd.DataFrame,\n"
        "    train_labels: np.ndarray,\n"
        "    valid_labels: np.ndarray,\n"
        "    train_teacher_probs: np.ndarray,\n"
        "    valid_teacher_probs: np.ndarray,\n"
        "    output_dir: Path,\n"
        ") -> tuple[pd.DataFrame, dict[str, tp.Any]]:\n",
    )
    .replace("resume_exp027b", "resume_exp031b")
    .replace(
        "'device': str(device),\n    }",
        "'device': str(device),\n"
        "        'regime_name': regime_name,\n"
        "        'train_base_rows': int((train_frame['source_kind'] == 'base_5s').sum()),\n"
        "        'train_overlap_rows': int((train_frame['source_kind'] == 'overlap_2p5s').sum()),\n"
        "        'valid_base_rows': int((valid_frame['source_kind'] == 'base_5s').sum()),\n"
        "    }",
    )
    .replace(
        "if RUN_TRAINING:\n"
        "    history_df, snapshot = train_one_fold(\n"
        "        train_frame,\n"
        "        valid_frame,\n"
        "        train_labels_fold,\n"
        "        valid_labels_fold,\n"
        "        train_teacher_probs,\n"
        "        valid_teacher_probs,\n"
        "        OUTPUT_DIR,\n"
        "    )\n"
        "else:\n"
        "    history_df = None\n"
        "    snapshot = None\n",
        "regime_rows: list[dict[str, tp.Any]] = []\n"
        "history_df = None\n"
        "snapshot = None\n"
        "for regime_name in CFG.regime_names:\n"
        "    payload = regime_payloads[regime_name]\n"
        "    regime_output_dir = OUTPUT_DIR / regime_name\n"
        "    save_json(payload['manifest_summary'], regime_output_dir / 'manifest_summary.json')\n"
        "    if RUN_TRAINING:\n"
        "        history_df, snapshot = train_one_fold(\n"
        "            regime_name,\n"
        "            payload['train_frame'],\n"
        "            payload['valid_frame'],\n"
        "            payload['train_labels'],\n"
        "            payload['valid_labels'],\n"
        "            payload['train_teacher_probs'],\n"
        "            payload['valid_teacher_probs'],\n"
        "            regime_output_dir,\n"
        "        )\n"
        "        regime_rows.append({\n"
        "            **payload['manifest_summary'],\n"
        "            'best_epoch': int(snapshot['best_epoch']),\n"
        "            'best_selection_metric': float(snapshot['best_selection_metric']),\n"
        "            'best_macro_auc': float(snapshot['best_macro_auc']),\n"
        "            'best_soundscape_macro_auc': float(snapshot['best_soundscape_macro_auc']),\n"
        "            'best_valid_loss': float(snapshot['best_valid_loss']),\n"
        "        })\n"
        "\n"
        "if regime_rows:\n"
        "    regime_summary_df = pd.DataFrame(regime_rows).sort_values('best_selection_metric', ascending=False).reset_index(drop=True)\n"
        "    regime_summary_df.to_csv(OUTPUT_DIR / 'regime_summary.csv', index=False)\n"
        "    report_snapshot = {\n"
        "        'experiment_id': CFG.experiment_id,\n"
        "        'experiment_name': CFG.experiment_name,\n"
        "        'fold': CFG.fold,\n"
        "        'regimes': list(CFG.regime_names),\n"
        "        'best_regime': str(regime_summary_df.iloc[0]['regime_name']),\n"
        "        'best_selection_metric': float(regime_summary_df.iloc[0]['best_selection_metric']),\n"
        "        'rows_compared': int(len(regime_summary_df)),\n"
        "        'output_dir': str(OUTPUT_DIR),\n"
        "    }\n"
        "    save_json(report_snapshot, OUTPUT_DIR / 'report_snapshot.json')\n"
        "    display(regime_summary_df)\n",
    )
)

result_cell = """
if (OUTPUT_DIR / 'regime_summary.csv').exists():
    report_snapshot = json.loads((OUTPUT_DIR / 'report_snapshot.json').read_text()) if (OUTPUT_DIR / 'report_snapshot.json').exists() else None
    if report_snapshot is not None:
        print('Report snapshot:')
        print(json.dumps(report_snapshot, indent=2))
    display(pd.read_csv(OUTPUT_DIR / 'regime_summary.csv'))
else:
    print('No ablation artifacts yet. Set RUN_TRAINING = True to run the regime comparison.')
"""


cells = [
    md_cell(
        """
        # Exp 031b: Overlap Ablation

        Run a controlled ablation on the trusted `exp_031` setup: compare `base_only`, `base_plus_overlap`, and `overlap_only` on the same fold and holdout rows.
        """
    ),
    md_cell(
        """
        ## Plan

        1. Reuse the trusted teacher cache and overlap construction from `exp_031`.
        2. Build three train regimes on the same fold: `base_only`, `base_plus_overlap`, and `overlap_only`.
        3. Train the same HGNetV2 scaffold under each regime.
        4. Compare all three on the same original trusted `5s` holdout rows.
        """
    ),
    code_cell(config_cell),
    code_cell(seed_device_cell),
    code_cell(data_prep_cell),
    code_cell(dataset_model_cell),
    code_cell(train_cell),
    code_cell(result_cell),
]


write_nb("exp_031b_overlap_ablation.ipynb", cells)
