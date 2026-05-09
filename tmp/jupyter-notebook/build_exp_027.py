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

exp019 = load_nb(NOTEBOOKS / "exp_019_v18_postproc_ablation.ipynb")
exp011 = load_nb(NOTEBOOKS / "exp_011_hgnetv2_soundscape_supervised.ipynb")
exp018b = load_nb(NOTEBOOKS / "exp_018b_targeted_merge_benchmark.ipynb")


def write_nb(name: str, cells: list[dict]) -> None:
    payload = {
        "cells": cells,
        "metadata": copy.deepcopy(base_md),
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (NOTEBOOKS / name).write_text(json.dumps(payload, ensure_ascii=False, indent=1))


exp027a_cells = [
    md_cell(
        """
        # Exp 027a: exp_015d Teacher Cache

        Build a fold-aware local teacher cache from the fixed `exp_015d` V18 artifact stack on fully labeled soundscape rows.
        """
    ),
    md_cell(
        """
        ## Plan

        1. Resolve the same competition, full-cache, and artifact inputs used by the strongest V18 path.
        2. Replay the fixed `exp_015d` stack on trusted fully labeled soundscape rows.
        3. Save row-aligned teacher logits, postprocessed teacher probabilities, labels, and fold assignments for later native distillation.
        """
    ),
    code_cell(
        """
        # Cell 0 — Input hints and teacher-cache config
        COMPETITION_HINT = None          # e.g. "birdclef-2026"
        PERCH_CACHE_HINT = None          # e.g. "perch-meta"
        ARTIFACTS_HINT = None            # e.g. "birdclef-exp015c-v18-artifacts"

        N_SPLITS = 4
        SAVE_RESULTS = True

        print({
            'COMPETITION_HINT': COMPETITION_HINT,
            'PERCH_CACHE_HINT': PERCH_CACHE_HINT,
            'ARTIFACTS_HINT': ARTIFACTS_HINT,
            'N_SPLITS': N_SPLITS,
            'SAVE_RESULTS': SAVE_RESULTS,
        })
        """
    ),
    code_cell(
        exp019["cells"][2]["source"][0]
        + "".join(exp019["cells"][2]["source"][1:]).replace(
            "exp_019_v18_postproc_ablation", "exp_027a_exp015d_teacher_cache"
        ).replace(
            "Benchmark device:", "Teacher-cache device:"
        )
    ),
    code_cell("".join(exp019["cells"][3]["source"])),
    code_cell(
        "".join(exp019["cells"][4]["source"])
        + "\n\n"
        + textwrap.dedent(
            """
            class MultiLabelStratifiedGroupKFold:
                def __init__(self, n_splits: int, random_state: int):
                    self.n_splits = n_splits
                    self.random_state = random_state

                def split(self, label_arr: np.ndarray, group_ids: np.ndarray):
                    np.random.seed(self.random_state)

                    unique_groups = sorted(set(group_ids))
                    group_to_idx = {group_id: idx for idx, group_id in enumerate(unique_groups)}
                    group_index_arr = np.vectorize(group_to_idx.get)(group_ids)

                    n_groups = len(unique_groups)
                    n_classes = label_arr.shape[1]
                    class_totals = label_arr.sum(axis=0).astype(np.float64)
                    class_totals[class_totals == 0] = 1.0

                    counts_by_group = np.zeros((n_groups, n_classes), dtype=np.int64)
                    for row_idx, group_idx in enumerate(group_index_arr):
                        counts_by_group[group_idx] += label_arr[row_idx].astype(np.int64)

                    counts_by_fold = np.zeros((self.n_splits, n_classes), dtype=np.int64)
                    groups_by_fold = [[] for _ in range(self.n_splits)]

                    order = sorted(
                        list(enumerate(counts_by_group)),
                        key=lambda item: -float(np.std(item[1])),
                    )
                    for group_idx, group_count in order:
                        best_fold = None
                        best_score = None
                        for fold_idx in range(self.n_splits):
                            counts_by_fold[fold_idx] += group_count
                            fold_score = (counts_by_fold / class_totals).std(axis=0).mean()
                            counts_by_fold[fold_idx] -= group_count
                            if best_score is None or fold_score < best_score:
                                best_score = float(fold_score)
                                best_fold = fold_idx
                        counts_by_fold[best_fold] += group_count
                        groups_by_fold[best_fold].append(group_idx)

                    row_indices = np.arange(label_arr.shape[0])
                    for fold_idx in range(self.n_splits):
                        valid_groups = groups_by_fold[fold_idx]
                        valid_mask = np.isin(group_index_arr, valid_groups)
                        yield row_indices[~valid_mask], row_indices[valid_mask]
            """
        )
    ),
    code_cell("".join(exp019["cells"][5]["source"])),
    code_cell("".join(exp019["cells"][6]["source"])),
    code_cell("".join(exp019["cells"][7]["source"])),
    code_cell("".join(exp019["cells"][8]["source"])),
    code_cell(
        """
        # Cell 9 — Build submit-like teacher probabilities and save the cache
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


        teacher_probs = build_tempered_probs(final_scores)
        if CALIBRATORS is not None:
            teacher_probs = apply_file_level_calibrators(teacher_probs, CALIBRATORS, n_windows=N_WINDOWS)
        top_k = int(CFG.get('file_level_top_k', 0))
        if top_k > 0:
            teacher_probs = file_level_confidence_scale(teacher_probs, n_windows=N_WINDOWS, top_k=top_k)
        if bool(CFG.get('rank_aware_scale', False)):
            teacher_probs = rank_aware_scaling(
                teacher_probs,
                n_windows=N_WINDOWS,
                power=float(CFG.get('rank_aware_power', 0.5)),
            )
        delta_alpha = float(CFG.get('delta_shift_alpha', 0.0))
        if delta_alpha > 0:
            teacher_probs = adaptive_delta_smooth(teacher_probs, n_windows=N_WINDOWS, base_alpha=delta_alpha)
        teacher_probs = np.clip(teacher_probs, 0.0, 1.0).astype(np.float32)

        teacher_meta = full_truth_aligned[['row_id', 'filename', 'start_sec', 'end_sec', 'site', 'hour_utc', 'label_list']].copy()
        teacher_meta['audio_id'] = teacher_meta['filename'].map(lambda x: Path(x).stem)
        teacher_meta['labels'] = teacher_meta['label_list'].map(lambda x: ';'.join(x))
        teacher_meta['source'] = 'soundscape_clip'
        teacher_meta['source_file_path'] = teacher_meta['filename'].map(lambda x: str(BASE / 'train_soundscapes' / x))
        teacher_meta['file_path'] = teacher_meta['source_file_path']
        teacher_meta['clip_start_frame'] = (teacher_meta['start_sec'].astype(int) * SR).astype(np.int64)
        teacher_meta['clip_end_frame'] = (teacher_meta['end_sec'].astype(int) * SR).astype(np.int64)
        teacher_meta['clip_duration_sec'] = (teacher_meta['end_sec'] - teacher_meta['start_sec']).astype(np.int64)
        teacher_meta['teacher_confidence'] = teacher_probs.max(axis=1).astype(np.float32)
        teacher_meta['teacher_entropy'] = (-(teacher_probs * np.log(np.clip(teacher_probs, 1e-7, 1.0))).sum(axis=1)).astype(np.float32)
        teacher_meta['teacher_logit_max'] = final_scores.max(axis=1).astype(np.float32)

        splitter = MultiLabelStratifiedGroupKFold(n_splits=N_SPLITS, random_state=1097)
        teacher_meta['fold'] = -1
        splits = list(splitter.split(Y_FULL.astype(np.uint8), teacher_meta['audio_id'].to_numpy()))
        for fold_idx, (_, valid_idx) in enumerate(splits):
            teacher_meta.loc[valid_idx, 'fold'] = fold_idx

        teacher_meta = teacher_meta.drop(columns=['label_list']).reset_index(drop=True)
        fold_summary = (
            teacher_meta.groupby('fold')
            .agg(rows=('row_id', 'size'), files=('filename', 'nunique'))
            .reset_index()
            .sort_values('fold')
            .reset_index(drop=True)
        )

        if SAVE_RESULTS:
            teacher_meta.to_parquet(OUTPUT_DIR / 'teacher_meta.parquet', index=False)
            np.savez_compressed(
                OUTPUT_DIR / 'teacher_outputs.npz',
                teacher_logits=final_scores.astype(np.float32),
                teacher_probs=teacher_probs.astype(np.float32),
                labels=Y_FULL.astype(np.float32),
                proto_scores=proto_scores_flat.astype(np.float32),
                raw_scores=scores_full_raw.astype(np.float32),
            )
            fold_summary.to_csv(OUTPUT_DIR / 'fold_summary.csv', index=False)

        report_snapshot = {
            'experiment_id': 'exp_027a',
            'experiment_name': 'exp015d_teacher_cache',
            'rows': int(len(teacher_meta)),
            'files': int(teacher_meta['filename'].nunique()),
            'n_classes': int(N_CLASSES),
            'n_splits': int(N_SPLITS),
            'teacher_logit_macro_auc': float(macro_auc_skip_empty(Y_FULL, final_scores)),
            'teacher_prob_macro_auc': float(macro_auc_skip_empty(Y_FULL, teacher_probs)),
            'mean_teacher_confidence': float(teacher_meta['teacher_confidence'].mean()),
            'mean_teacher_entropy': float(teacher_meta['teacher_entropy'].mean()),
            'output_dir': str(OUTPUT_DIR),
            'notes': 'Teacher cache uses fully labeled soundscape rows only and keeps the fixed exp_015d artifact stack intact.',
        }
        if SAVE_RESULTS:
            (OUTPUT_DIR / 'report_snapshot.json').write_text(json.dumps(report_snapshot, indent=2))

        print(json.dumps(report_snapshot, indent=2))
        display(fold_summary)
        """
    ),
    code_cell(
        """
        if (OUTPUT_DIR / 'report_snapshot.json').exists():
            snapshot = json.loads((OUTPUT_DIR / 'report_snapshot.json').read_text())
            print('Snapshot:')
            print(json.dumps(snapshot, indent=2))
            if (OUTPUT_DIR / 'fold_summary.csv').exists():
                display(pd.read_csv(OUTPUT_DIR / 'fold_summary.csv'))
        else:
            print('No teacher cache artifacts yet. Run all cells to build the cache.')
        """
    ),
]


exp027b_cells = [
    md_cell(
        """
        # Exp 027b: HGNetV2 Soundscape Distill From exp_015d

        Train a native HGNetV2 student on fully labeled soundscape rows using `exp_015d` as a fixed teacher.
        """
    ),
    md_cell(
        """
        ## Plan

        1. Load the fold-aware teacher cache from `exp_027a`.
        2. Train a soundscape-only HGNetV2 student initialized from the matching `exp_011` fold checkpoint.
        3. Combine supervised BCE with teacher-target distillation and export OOF-style validation outputs for later blend checks.
        """
    ),
    code_cell(
        """
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
        """
    ),
    code_cell(
        """
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
            experiment_id: str = 'exp_027b'
            experiment_name: str = 'hgnetv2_soundscape_distill_from_exp015d'
            fold: int = 0
            n_folds: int = 4
            random_seed: int = 1098

            sample_rate: int = 32_000
            segment_seconds: int = 5
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
            distill_weight: float = 0.35
            distill_temperature: float = 1.0
            teacher_weight_power: float = 1.0
            min_teacher_confidence_train: float = 0.0

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
    ),
    code_cell(
        """
        def set_random_seed(seed: int) -> None:
            os.environ['PYTHONHASHSEED'] = str(seed)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            if hasattr(torch.backends, 'cudnn'):
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False


        def pick_device() -> torch.device:
            if torch.cuda.is_available():
                return torch.device('cuda')
            if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
                return torch.device('mps')
            return torch.device('cpu')


        def autocast_context() -> tp.ContextManager[object]:
            if amp_enabled:
                return torch.amp.autocast('cuda', enabled=True)
            return nullcontext()


        def save_json(payload: dict[str, tp.Any], path: Path) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, indent=2, sort_keys=False))


        set_random_seed(CFG.random_seed)
        device = pick_device()
        amp_enabled = bool(CFG.use_amp and device.type == 'cuda')
        if hasattr(torch, 'set_float32_matmul_precision'):
            torch.set_float32_matmul_precision('high')

        print({
            'root': str(ROOT),
            'teacher_dir': str(TEACHER_DIR),
            'exp011_dir': str(EXP011_DIR),
            'device': str(device),
            'amp_enabled': amp_enabled,
            'output_dir': str(OUTPUT_DIR),
        })
        """
    ),
    code_cell(
        """
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

        train_mask = teacher_meta['fold'].to_numpy() != CFG.fold
        valid_mask = teacher_meta['fold'].to_numpy() == CFG.fold

        train_frame = teacher_meta.loc[train_mask].reset_index(drop=True)
        valid_frame = teacher_meta.loc[valid_mask].reset_index(drop=True)
        train_labels_fold = teacher_labels[train_mask]
        valid_labels_fold = teacher_labels[valid_mask]
        train_teacher_probs = teacher_probs[train_mask]
        valid_teacher_probs = teacher_probs[valid_mask]

        if CFG.min_teacher_confidence_train > 0:
            keep = train_frame['teacher_confidence'].to_numpy() >= CFG.min_teacher_confidence_train
            train_frame = train_frame.loc[keep].reset_index(drop=True)
            train_labels_fold = train_labels_fold[keep]
            train_teacher_probs = train_teacher_probs[keep]

        if CFG.max_train_rows is not None:
            keep = min(CFG.max_train_rows, len(train_frame))
            selected = train_frame.sample(keep, random_state=CFG.random_seed).index.to_numpy()
            train_frame = train_frame.loc[selected].reset_index(drop=True)
            train_labels_fold = train_labels_fold[selected]
            train_teacher_probs = train_teacher_probs[selected]
        if CFG.max_valid_rows is not None:
            keep = min(CFG.max_valid_rows, len(valid_frame))
            selected = valid_frame.sample(keep, random_state=CFG.random_seed).index.to_numpy()
            valid_frame = valid_frame.loc[selected].reset_index(drop=True)
            valid_labels_fold = valid_labels_fold[selected]
            valid_teacher_probs = valid_teacher_probs[selected]

        fold_overview = {
            'fold': CFG.fold,
            'train_rows': int(len(train_frame)),
            'valid_rows': int(len(valid_frame)),
            'train_files': int(train_frame['filename'].nunique()),
            'valid_files': int(valid_frame['filename'].nunique()),
            'mean_train_teacher_confidence': float(train_frame['teacher_confidence'].mean()),
            'mean_valid_teacher_confidence': float(valid_frame['teacher_confidence'].mean()),
        }
        print(json.dumps(fold_overview, indent=2))
        """
    ),
    code_cell(
        """
        def read_audio_region(path: str, clip_start_frame: int, clip_end_frame: int, sample_frames: int, training: bool) -> np.ndarray:
            with sf.SoundFile(path) as snd:
                total_frames = snd.frames
                region_start = max(int(clip_start_frame), 0)
                region_end = total_frames if int(clip_end_frame) <= 0 else min(int(clip_end_frame), total_frames)
                region_frames = max(region_end - region_start, 0)
                if region_frames <= 0:
                    return np.zeros(sample_frames, dtype=np.float32)
                if region_frames >= sample_frames:
                    if training:
                        offset = np.random.randint(region_frames - sample_frames + 1)
                    else:
                        offset = 0
                    snd.seek(region_start + offset)
                    wave = snd.read(frames=sample_frames, dtype='float32', always_2d=True)
                    wave = wave.mean(axis=1).astype(np.float32, copy=False)
                    if wave.shape[0] == sample_frames:
                        return wave
                else:
                    snd.seek(region_start)
                    wave = snd.read(frames=region_frames, dtype='float32', always_2d=True)
                    wave = wave.mean(axis=1).astype(np.float32, copy=False)

            actual_frames = int(wave.shape[0])
            if actual_frames >= sample_frames:
                return wave[:sample_frames]
            padded = np.zeros(sample_frames, dtype=np.float32)
            pad_start = np.random.randint(sample_frames - actual_frames + 1) if training else 0
            padded[pad_start: pad_start + actual_frames] = wave
            return padded


        class DistillSoundscapeDataset(Dataset):
            def __init__(self, frame: pd.DataFrame, label_arr: np.ndarray, teacher_probs_arr: np.ndarray, training: bool):
                self.frame = frame.reset_index(drop=True)
                self.labels = label_arr.astype(np.float32)
                self.teacher_probs = teacher_probs_arr.astype(np.float32)
                self.training = training
                self.sample_frames = CFG.sample_rate * CFG.segment_seconds
                self.teacher_weights = np.power(
                    np.clip(self.frame['teacher_confidence'].to_numpy(dtype=np.float32), 1e-3, 1.0),
                    CFG.teacher_weight_power,
                ).astype(np.float32)

            def __len__(self) -> int:
                return len(self.frame)

            def __getitem__(self, idx: int) -> dict[str, tp.Any]:
                row = self.frame.iloc[idx]
                wave = read_audio_region(
                    path=str(row['file_path']),
                    clip_start_frame=int(row['clip_start_frame']),
                    clip_end_frame=int(row['clip_end_frame']),
                    sample_frames=self.sample_frames,
                    training=self.training,
                )
                return {
                    'wave': wave,
                    'label': self.labels[idx],
                    'teacher_prob': self.teacher_probs[idx],
                    'teacher_weight': self.teacher_weights[idx],
                    'index': idx,
                }


        class LogMelSpectrogramTransform(nn.Module):
            def __init__(self):
                super().__init__()
                self.mel = torchaudio.transforms.MelSpectrogram(
                    sample_rate=CFG.sample_rate,
                    n_fft=CFG.n_fft,
                    win_length=CFG.win_length,
                    hop_length=CFG.hop_length,
                    f_min=CFG.f_min,
                    n_mels=CFG.n_mels,
                    power=2.0,
                    center=True,
                    pad_mode='reflect',
                    norm='slaney',
                    mel_scale='htk',
                )
                self.db = torchaudio.transforms.AmplitudeToDB(stype='power', top_db=CFG.top_db)

            @torch.no_grad()
            def forward(self, wave: torch.Tensor) -> torch.Tensor:
                if wave.ndim == 1:
                    wave = wave.unsqueeze(0)
                mel = self.mel(wave)
                mel = self.db(mel)
                mel = mel.unsqueeze(1)
                mel = F.interpolate(mel, size=CFG.image_size, mode='bilinear', align_corners=False)
                flat = mel.flatten(start_dim=1)
                min_val = flat.min(dim=1).values[:, None, None, None]
                max_val = flat.max(dim=1).values[:, None, None, None]
                mel = (mel - min_val) / (max_val - min_val + 1e-7)
                return mel


        def make_loader(dataset: Dataset, training: bool) -> DataLoader:
            return DataLoader(
                dataset=dataset,
                batch_size=CFG.batch_size,
                shuffle=training,
                drop_last=training and len(dataset) >= CFG.batch_size,
                num_workers=CFG.num_workers,
                pin_memory=(device.type == 'cuda'),
            )


        def compute_macro_auc(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float | int]:
            positive_mask = y_true.sum(axis=0) > 0
            negative_mask = (y_true.shape[0] - y_true.sum(axis=0)) > 0
            scored_mask = positive_mask & negative_mask
            scored_classes = int(scored_mask.sum())
            skipped_no_positive = int((~positive_mask).sum())
            skipped_no_negative = int((~negative_mask).sum())
            if scored_classes == 0:
                return {
                    'macro_auc': np.nan,
                    'scored_classes': scored_classes,
                    'skipped_no_positive': skipped_no_positive,
                    'skipped_no_negative': skipped_no_negative,
                }
            macro_auc = roc_auc_score(y_true[:, scored_mask], y_score[:, scored_mask], average='macro')
            return {
                'macro_auc': float(macro_auc),
                'scored_classes': scored_classes,
                'skipped_no_positive': skipped_no_positive,
                'skipped_no_negative': skipped_no_negative,
            }


        def evaluate_predictions(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float | int]:
            overall = compute_macro_auc(y_true, y_score)
            return {
                **overall,
                'soundscape_macro_auc': overall['macro_auc'],
                'soundscape_scored_classes': overall['scored_classes'],
                'soundscape_skipped_no_positive': overall['skipped_no_positive'],
                'soundscape_skipped_no_negative': overall['skipped_no_negative'],
            }


        def build_model() -> nn.Module:
            return timm.create_model(
                CFG.model_name,
                pretrained=CFG.pretrained,
                in_chans=1,
                num_classes=N_CLASSES,
                drop_path_rate=CFG.drop_path_rate,
            )


        def load_pretrained_state(model: nn.Module, checkpoint_path: Path) -> None:
            payload = torch.load(checkpoint_path, map_location='cpu')
            state_dict = payload['model_state_dict'] if 'model_state_dict' in payload else payload
            model.load_state_dict(state_dict, strict=True)


        def load_resume_payload(output_dir: Path) -> dict[str, tp.Any] | None:
            last_path = output_dir / 'last_model.pt'
            if not last_path.exists():
                return None
            return torch.load(last_path, map_location='cpu')
        """
    ),
    code_cell(
        """
        def train_one_fold(
            train_frame: pd.DataFrame,
            valid_frame: pd.DataFrame,
            train_labels: np.ndarray,
            valid_labels: np.ndarray,
            train_teacher_probs: np.ndarray,
            valid_teacher_probs: np.ndarray,
            output_dir: Path,
        ) -> tuple[pd.DataFrame, dict[str, tp.Any]]:
            output_dir.mkdir(parents=True, exist_ok=True)

            train_dataset = DistillSoundscapeDataset(train_frame, train_labels, train_teacher_probs, training=True)
            valid_dataset = DistillSoundscapeDataset(valid_frame, valid_labels, valid_teacher_probs, training=False)
            train_loader = make_loader(train_dataset, training=True)
            valid_loader = make_loader(valid_dataset, training=False)

            mel_transform = LogMelSpectrogramTransform().to(device).eval()
            model = build_model().to(device)
            init_ckpt = EXP011_DIR / f'fold_{CFG.fold:02d}' / 'best_model.pt'
            if init_ckpt.exists():
                load_pretrained_state(model, init_ckpt)
            optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.learning_rate, weight_decay=CFG.weight_decay)
            scheduler = OneCycleLR(
                optimizer=optimizer,
                max_lr=CFG.learning_rate,
                epochs=CFG.epochs,
                steps_per_epoch=max(1, len(train_loader)),
                pct_start=max(1, CFG.warmup_epochs) / max(1, CFG.epochs),
                div_factor=25,
                final_div_factor=4.0,
            )
            scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled)

            history: list[dict[str, tp.Any]] = []
            start_epoch = 1
            best_metric = -float('inf')
            resume_mode = 'init_from_exp011_fold'

            if RESUME_TRAINING:
                payload = load_resume_payload(output_dir)
                if payload is not None:
                    model.load_state_dict(payload['model_state_dict'])
                    optimizer.load_state_dict(payload['optimizer_state_dict'])
                    scheduler.load_state_dict(payload['scheduler_state_dict'])
                    scaler_state = payload.get('scaler_state_dict')
                    if scaler_state is not None and amp_enabled:
                        scaler.load_state_dict(scaler_state)
                    history = payload.get('history', [])
                    start_epoch = int(payload.get('epoch', 0)) + 1
                    best_metric = float(payload.get('best_metric', -float('inf')))
                    resume_mode = 'resume_exp027b'

            for epoch in range(start_epoch, CFG.epochs + 1):
                model.train()
                train_loss_sum = 0.0
                train_sup_sum = 0.0
                train_distill_sum = 0.0

                for batch in tqdm(train_loader, desc=f'epoch {epoch} train', leave=False):
                    wave = batch['wave'].to(device, dtype=torch.float32)
                    label = batch['label'].to(device, dtype=torch.float32)
                    teacher_prob = batch['teacher_prob'].to(device, dtype=torch.float32)
                    teacher_weight = batch['teacher_weight'].to(device, dtype=torch.float32)

                    optimizer.zero_grad(set_to_none=True)
                    with torch.no_grad():
                        image = mel_transform(wave)
                    with autocast_context():
                        logits = model(image)
                        supervised_loss = F.binary_cross_entropy_with_logits(logits, label)
                        distill_vec = F.binary_cross_entropy_with_logits(
                            logits / CFG.distill_temperature,
                            teacher_prob,
                            reduction='none',
                        ).mean(dim=1)
                        distill_loss = (distill_vec * teacher_weight).mean() * (CFG.distill_temperature ** 2)
                        loss = supervised_loss + CFG.distill_weight * distill_loss
                    if amp_enabled:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()
                    scheduler.step()

                    train_loss_sum += float(loss.item())
                    train_sup_sum += float(supervised_loss.item())
                    train_distill_sum += float(distill_loss.item())
                    del wave, label, teacher_prob, teacher_weight, image, logits, supervised_loss, distill_loss, loss

                train_loss = train_loss_sum / max(1, len(train_loader))
                train_sup = train_sup_sum / max(1, len(train_loader))
                train_distill = train_distill_sum / max(1, len(train_loader))

                model.eval()
                valid_loss_sum = 0.0
                logits_parts: list[np.ndarray] = []
                label_parts: list[np.ndarray] = []
                index_parts: list[np.ndarray] = []
                teacher_conf_parts: list[np.ndarray] = []

                for batch in tqdm(valid_loader, desc=f'epoch {epoch} valid', leave=False):
                    wave = batch['wave'].to(device, dtype=torch.float32)
                    label = batch['label'].to(device, dtype=torch.float32)
                    teacher_prob = batch['teacher_prob'].to(device, dtype=torch.float32)
                    teacher_weight = batch['teacher_weight'].to(device, dtype=torch.float32)
                    with torch.no_grad():
                        image = mel_transform(wave)
                        with autocast_context():
                            logits = model(image)
                            supervised_loss = F.binary_cross_entropy_with_logits(logits, label)
                            distill_vec = F.binary_cross_entropy_with_logits(
                                logits / CFG.distill_temperature,
                                teacher_prob,
                                reduction='none',
                            ).mean(dim=1)
                            distill_loss = (distill_vec * teacher_weight).mean() * (CFG.distill_temperature ** 2)
                            loss = supervised_loss + CFG.distill_weight * distill_loss
                    valid_loss_sum += float(loss.item())
                    logits_parts.append(logits.detach().float().cpu().numpy())
                    label_parts.append(label.detach().float().cpu().numpy())
                    index_parts.append(batch['index'].detach().cpu().numpy())
                    teacher_conf_parts.append(teacher_weight.detach().float().cpu().numpy())
                    del wave, label, teacher_prob, teacher_weight, image, logits, supervised_loss, distill_loss, loss

                valid_loss = valid_loss_sum / max(1, len(valid_loader))
                logits_all = np.concatenate(logits_parts, axis=0)
                labels_all = np.concatenate(label_parts, axis=0)
                index_all = np.concatenate(index_parts, axis=0)
                order = np.argsort(index_all)
                logits_all = logits_all[order]
                labels_all = labels_all[order]
                probs_all = (1.0 / (1.0 + np.exp(-logits_all))).astype(np.float32)
                valid_metrics = evaluate_predictions(labels_all, probs_all)

                selected_metric = valid_metrics[CFG.selection_metric]
                if np.isnan(selected_metric):
                    selected_metric = valid_metrics['macro_auc']

                row = {
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'train_supervised_loss': train_sup,
                    'train_distill_loss': train_distill,
                    'macro_auc': valid_metrics['macro_auc'],
                    'scored_classes': valid_metrics['scored_classes'],
                    'skipped_no_positive': valid_metrics['skipped_no_positive'],
                    'skipped_no_negative': valid_metrics['skipped_no_negative'],
                    'soundscape_macro_auc': valid_metrics['soundscape_macro_auc'],
                    'soundscape_scored_classes': valid_metrics['soundscape_scored_classes'],
                    'soundscape_skipped_no_positive': valid_metrics['soundscape_skipped_no_positive'],
                    'soundscape_skipped_no_negative': valid_metrics['soundscape_skipped_no_negative'],
                    'valid_loss': valid_loss,
                    'learning_rate': float(scheduler.get_last_lr()[0]),
                    'selection_metric': float(selected_metric),
                    'distill_weight': float(CFG.distill_weight),
                    'mean_train_teacher_confidence': float(train_frame['teacher_confidence'].mean()),
                }
                history.append(row)
                history_df = pd.DataFrame(history)
                history_df.to_csv(output_dir / 'history.csv', index=False)

                payload = {
                    'epoch': epoch,
                    'best_metric': max(best_metric, float(selected_metric)),
                    'history': history,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict() if amp_enabled else None,
                    'cfg': asdict(CFG),
                    'classes': CLASSES,
                    'resume_mode': resume_mode,
                }
                if CFG.save_every_epoch:
                    torch.save(payload, output_dir / 'last_model.pt')

                if float(selected_metric) > best_metric:
                    best_metric = float(selected_metric)
                    torch.save(payload, output_dir / 'best_model.pt')
                    np.savez_compressed(
                        output_dir / 'best_valid_outputs.npz',
                        logits=logits_all.astype(np.float32),
                        probs=probs_all.astype(np.float32),
                        labels=labels_all.astype(np.float32),
                    )
                    valid_frame.reset_index(drop=True).to_csv(output_dir / 'best_valid_meta.csv', index=False)

                print(row)

            history_df = pd.DataFrame(history)
            best_row = history_df.loc[history_df['selection_metric'].idxmax()].to_dict()
            snapshot = {
                'experiment_id': CFG.experiment_id,
                'experiment_name': CFG.experiment_name,
                'fold': CFG.fold,
                'best_epoch': int(best_row['epoch']),
                'best_selection_metric': float(best_row['selection_metric']),
                'best_macro_auc': float(best_row['macro_auc']),
                'best_soundscape_macro_auc': float(best_row['soundscape_macro_auc']),
                'scored_classes': int(best_row['scored_classes']),
                'soundscape_scored_classes': int(best_row['soundscape_scored_classes']),
                'best_valid_loss': float(best_row['valid_loss']),
                'train_rows': int(len(train_frame)),
                'valid_rows': int(len(valid_frame)),
                'resume_mode': resume_mode,
                'teacher_dir': str(TEACHER_DIR),
                'output_dir': str(output_dir),
                'model_name': CFG.model_name,
                'distill_weight': float(CFG.distill_weight),
                'device': str(device),
            }
            save_json(snapshot, output_dir / 'result_snapshot.json')
            return history_df, snapshot


        if RUN_TRAINING:
            history_df, snapshot = train_one_fold(
                train_frame,
                valid_frame,
                train_labels_fold,
                valid_labels_fold,
                train_teacher_probs,
                valid_teacher_probs,
                OUTPUT_DIR,
            )
        else:
            history_df = None
            snapshot = None
        """
    ),
    code_cell(
        """
        if (OUTPUT_DIR / 'result_snapshot.json').exists():
            snapshot = json.loads((OUTPUT_DIR / 'result_snapshot.json').read_text())
            print('Snapshot:')
            print(json.dumps(snapshot, indent=2))
            if (OUTPUT_DIR / 'history.csv').exists():
                display(pd.read_csv(OUTPUT_DIR / 'history.csv'))
        else:
            print('No training artifacts yet. Set RUN_TRAINING = True to start the experiment.')
        """
    ),
]


exp027c_cells = [
    md_cell(
        """
        # Exp 027c: exp_015d Native Student Blend Benchmark

        Benchmark whether a soundscape-only native student distilled from `exp_015d` adds complementary signal on top of the fixed teacher.
        """
    ),
    code_cell(
        """
        from __future__ import annotations

        import json
        import typing as tp
        from dataclasses import dataclass
        from pathlib import Path

        import numpy as np
        import pandas as pd
        from sklearn.metrics import roc_auc_score


        def resolve_repo_root(start: Path | None = None) -> Path:
            current = (start or Path.cwd()).resolve()
            for candidate in [current, *current.parents]:
                if (candidate / 'PROJECT_STATE.md').exists() and (candidate / 'data').exists():
                    return candidate
            raise FileNotFoundError('Could not resolve repository root')


        @dataclass
        class Config:
            experiment_id: str = 'exp_027c'
            experiment_name: str = 'exp015d_native_student_blend_benchmark'
            teacher_experiment: str = 'exp_027a_exp015d_teacher_cache'
            student_experiment: str = 'exp_027b_hgnetv2_soundscape_distill_from_exp015d'
            teacher_dir_override: str | None = None
            student_dir_override: str | None = None
            weight_grid: tuple[float, ...] = tuple(round(x, 2) for x in np.linspace(0.0, 1.0, 21))


        CFG = Config()
        ROOT = resolve_repo_root()
        DATA = ROOT / 'data' / 'birdclef-2026'
        OUTPUT_DIR = ROOT / 'experiments' / 'outputs' / f'{CFG.experiment_id}_{CFG.experiment_name}'
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


        def save_json(payload: dict[str, tp.Any], path: Path) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, indent=2, sort_keys=False))


        print({'root': str(ROOT), 'output_dir': str(OUTPUT_DIR)})
        """
    ),
    code_cell(
        """
        def has_teacher_cache(path: Path) -> bool:
            return (path / 'teacher_meta.parquet').exists() and (path / 'teacher_outputs.npz').exists()


        def has_student_outputs(path: Path) -> bool:
            fold_dirs = [p for p in path.iterdir() if p.is_dir() and p.name.startswith('fold_')] if path.exists() else []
            return any((fold_dir / 'best_valid_meta.csv').exists() and (fold_dir / 'best_valid_outputs.npz').exists() for fold_dir in fold_dirs)


        def resolve_teacher_dir() -> Path:
            candidates = []
            if CFG.teacher_dir_override:
                candidates.append(Path(CFG.teacher_dir_override).expanduser())
            candidates.extend([
                ROOT / 'experiments' / 'outputs' / CFG.teacher_experiment,
                Path.home() / 'Downloads' / CFG.teacher_experiment,
            ])
            for candidate in candidates:
                if has_teacher_cache(candidate):
                    return candidate
            for search_root in [ROOT / 'experiments' / 'outputs', Path.home() / 'Downloads']:
                if not search_root.exists():
                    continue
                for meta_path in search_root.rglob('teacher_meta.parquet'):
                    parent = meta_path.parent
                    if has_teacher_cache(parent) and CFG.teacher_experiment in str(parent):
                        return parent
            raise FileNotFoundError(
                'Could not resolve exp_027a teacher cache for exp_027c. '
                'Run exp_027a first or set CFG.teacher_dir_override.'
            )


        def resolve_student_dir() -> Path:
            candidates = []
            if CFG.student_dir_override:
                candidates.append(Path(CFG.student_dir_override).expanduser())
            candidates.extend([
                ROOT / 'experiments' / 'outputs' / CFG.student_experiment,
                Path.home() / 'Downloads' / CFG.student_experiment,
            ])
            for candidate in candidates:
                if has_student_outputs(candidate):
                    return candidate
            for search_root in [ROOT / 'experiments' / 'outputs', Path.home() / 'Downloads']:
                if not search_root.exists():
                    continue
                for fold_meta in search_root.rglob('best_valid_meta.csv'):
                    parent = fold_meta.parent.parent
                    if has_student_outputs(parent) and CFG.student_experiment in str(parent):
                        return parent
            raise FileNotFoundError(
                'Could not resolve exp_027b student outputs for exp_027c. '
                'Run exp_027b first or set CFG.student_dir_override.'
            )


        teacher_dir = resolve_teacher_dir()
        student_dir = resolve_student_dir()

        taxonomy = pd.read_csv(DATA / 'taxonomy.csv')
        classes = taxonomy['primary_label'].astype(str).tolist()
        class_name_map = taxonomy.set_index('primary_label')['class_name'].to_dict()

        teacher_meta = pd.read_parquet(teacher_dir / 'teacher_meta.parquet')
        teacher_arr = np.load(teacher_dir / 'teacher_outputs.npz')
        teacher_probs = teacher_arr['teacher_probs'].astype(np.float32)
        teacher_labels = teacher_arr['labels'].astype(np.float32)

        rows_meta = []
        rows_teacher = []
        rows_student = []
        rows_labels = []

        fold_dirs = sorted(path for path in student_dir.iterdir() if path.is_dir() and path.name.startswith('fold_'))
        if not fold_dirs:
            raise FileNotFoundError(f'No student fold outputs found in {student_dir}')

        for fold_dir in fold_dirs:
            meta_student = pd.read_csv(fold_dir / 'best_valid_meta.csv')
            arr_student = np.load(fold_dir / 'best_valid_outputs.npz')
            probs_student = arr_student['probs'].astype(np.float32)
            labels_student = arr_student['labels'].astype(np.float32)

            teacher_fold = teacher_meta.merge(
                meta_student[['row_id']],
                on='row_id',
                how='inner',
            )
            teacher_idx = teacher_fold.index.to_numpy()
            merged = meta_student[['row_id']].merge(
                teacher_meta.reset_index()[['index', 'row_id']],
                on='row_id',
                how='inner',
            )
            assert len(merged) == len(meta_student), (len(merged), len(meta_student))
            teacher_index = merged['index'].to_numpy()

            rows_meta.append(meta_student.assign(fold_src=fold_dir.name))
            rows_teacher.append(teacher_probs[teacher_index])
            rows_student.append(probs_student)
            rows_labels.append(labels_student)

        meta_all = pd.concat(rows_meta, ignore_index=True)
        teacher_all = np.concatenate(rows_teacher, axis=0)
        student_all = np.concatenate(rows_student, axis=0)
        labels_all = np.concatenate(rows_labels, axis=0)

        assert meta_all.shape[0] == teacher_all.shape[0] == student_all.shape[0] == labels_all.shape[0]

        texture_idx = np.array(
            [i for i, label in enumerate(classes) if class_name_map.get(label) in {'Amphibia', 'Insecta'}],
            dtype=np.int64,
        )

        print({
            'rows': int(len(meta_all)),
            'files': int(meta_all['filename'].nunique()),
            'folds': sorted(meta_all['fold_src'].unique().tolist()),
            'texture_classes': int(len(texture_idx)),
        })
        """
    ),
    code_cell(
        """
        def macro_auc_skip_empty(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, int]:
            auc_values: list[float] = []
            for idx in range(y_true.shape[1]):
                yt = y_true[:, idx]
                if len(np.unique(yt)) < 2:
                    continue
                auc_values.append(float(roc_auc_score(yt, y_score[:, idx])))
            return float(np.mean(auc_values)), int(len(auc_values))


        teacher_macro_auc, teacher_scored = macro_auc_skip_empty(labels_all, teacher_all)
        student_macro_auc, student_scored = macro_auc_skip_empty(labels_all, student_all)
        teacher_texture_auc, teacher_texture_scored = macro_auc_skip_empty(labels_all[:, texture_idx], teacher_all[:, texture_idx])
        student_texture_auc, student_texture_scored = macro_auc_skip_empty(labels_all[:, texture_idx], student_all[:, texture_idx])

        baseline_df = pd.DataFrame([
            {
                'variant': 'teacher_exp015d',
                'macro_auc': teacher_macro_auc,
                'texture_macro_auc': teacher_texture_auc,
                'scored_classes': teacher_scored,
                'texture_scored_classes': teacher_texture_scored,
            },
            {
                'variant': 'student_exp027b',
                'macro_auc': student_macro_auc,
                'texture_macro_auc': student_texture_auc,
                'scored_classes': student_scored,
                'texture_scored_classes': student_texture_scored,
            },
        ])
        display(baseline_df)

        sweep_rows: list[dict[str, tp.Any]] = []
        for w_student in CFG.weight_grid:
            blend = (1.0 - w_student) * teacher_all + w_student * student_all
            macro_auc, scored_classes = macro_auc_skip_empty(labels_all, blend)
            texture_auc, texture_scored_classes = macro_auc_skip_empty(labels_all[:, texture_idx], blend[:, texture_idx])
            sweep_rows.append({
                'w_student': float(w_student),
                'w_teacher': float(1.0 - w_student),
                'macro_auc': macro_auc,
                'texture_macro_auc': texture_auc,
                'scored_classes': scored_classes,
                'texture_scored_classes': texture_scored_classes,
            })

        weight_sweep = pd.DataFrame(sweep_rows).sort_values(['macro_auc', 'texture_macro_auc'], ascending=[False, False]).reset_index(drop=True)
        display(weight_sweep.head(12))
        """
    ),
    code_cell(
        """
        taxon_rows: list[dict[str, tp.Any]] = []
        taxon_series = taxonomy['class_name'].astype(str)
        for taxon in sorted(taxon_series.unique().tolist()):
            idx = np.array([i for i, label in enumerate(classes) if class_name_map.get(label) == taxon], dtype=np.int64)
            if len(idx) == 0:
                continue
            teacher_auc, teacher_sc = macro_auc_skip_empty(labels_all[:, idx], teacher_all[:, idx])
            student_auc, student_sc = macro_auc_skip_empty(labels_all[:, idx], student_all[:, idx])
            best_weight = float(weight_sweep.iloc[0]['w_student'])
            blend = (1.0 - best_weight) * teacher_all[:, idx] + best_weight * student_all[:, idx]
            blend_auc, blend_sc = macro_auc_skip_empty(labels_all[:, idx], blend)
            taxon_rows.append({
                'taxon': taxon,
                'teacher_macro_auc': teacher_auc,
                'student_macro_auc': student_auc,
                'best_blend_macro_auc': blend_auc,
                'teacher_scored_classes': teacher_sc,
                'student_scored_classes': student_sc,
                'blend_scored_classes': blend_sc,
            })

        taxon_summary = pd.DataFrame(taxon_rows).sort_values('best_blend_macro_auc', ascending=False).reset_index(drop=True)
        display(taxon_summary)

        baseline_df.to_csv(OUTPUT_DIR / 'baseline_summary.csv', index=False)
        weight_sweep.to_csv(OUTPUT_DIR / 'weight_sweep.csv', index=False)
        taxon_summary.to_csv(OUTPUT_DIR / 'taxon_summary.csv', index=False)

        report_snapshot = {
            'experiment_id': CFG.experiment_id,
            'experiment_name': CFG.experiment_name,
            'rows': int(len(meta_all)),
            'files': int(meta_all['filename'].nunique()),
            'teacher_macro_auc': float(teacher_macro_auc),
            'student_macro_auc': float(student_macro_auc),
            'teacher_texture_macro_auc': float(teacher_texture_auc),
            'student_texture_macro_auc': float(student_texture_auc),
            'best_weight_student': float(weight_sweep.iloc[0]['w_student']),
            'best_macro_auc': float(weight_sweep.iloc[0]['macro_auc']),
            'best_texture_macro_auc': float(weight_sweep.iloc[0]['texture_macro_auc']),
            'note': 'Teacher scores come from fixed exp_015d replay on fully labeled soundscape rows, not from a fold-safe retraining process.',
        }
        save_json(report_snapshot, OUTPUT_DIR / 'report_snapshot.json')
        print(json.dumps(report_snapshot, indent=2))
        """
    ),
]


write_nb("exp_027a_exp015d_teacher_cache.ipynb", exp027a_cells)
write_nb("exp_027b_hgnetv2_soundscape_distill_from_exp015d.ipynb", exp027b_cells)
write_nb("exp_027c_exp015d_native_student_blend_benchmark.ipynb", exp027c_cells)
