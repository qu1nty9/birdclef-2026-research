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
        # Exp 028a: CLAP Perch Complementarity Benchmark

        Quick local benchmark for a **new external family**: can CLAP embeddings add any complementary signal on top of the trusted `exp_015d` teacher cache?
        """
    ),
    md_cell(
        """
        ## Plan

        1. Load the completed `exp_027a` teacher cache built from the fixed `exp_015d` path.
        2. Load an aligned CLAP cache on the same trusted rows.
        3. Train a small fold-aware CLAP probe OOF on the teacher-cache folds.
        4. Measure whether `teacher + CLAP` blends improve over pure teacher before any larger implementation effort.
        """
    ),
    code_cell(
        """
        from __future__ import annotations

        import json
        import warnings
        import typing as tp
        from dataclasses import dataclass
        from pathlib import Path

        import numpy as np
        import pandas as pd
        from sklearn.decomposition import PCA
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.preprocessing import StandardScaler

        warnings.filterwarnings('ignore', category=ConvergenceWarning)

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
            experiment_id: str = 'exp_028a'
            experiment_name: str = 'clap_perch_complementarity_benchmark'
            teacher_dir_override: str | None = None
            clap_dir_override: str | None = None
            clap_pca_dim: int = 128
            min_pos_rows: int = 8
            min_neg_rows: int = 8
            logreg_c: float = 1.0
            weight_grid: tuple[float, ...] = tuple(round(x, 2) for x in np.linspace(0.0, 1.0, 21))


        CFG = Config()
        ROOT = resolve_repo_root()
        DATA = ROOT / 'data' / 'birdclef-2026'
        EXPERIMENTS = ROOT / 'experiments'
        OUTPUT_DIR = EXPERIMENTS / 'outputs' / f'{CFG.experiment_id}_{CFG.experiment_name}'
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


        def save_json(payload: dict[str, tp.Any], path: Path) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, indent=2, sort_keys=False))


        print({
            'root': str(ROOT),
            'output_dir': str(OUTPUT_DIR),
            'teacher_dir_override': CFG.teacher_dir_override,
            'clap_dir_override': CFG.clap_dir_override,
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
            for search_root in [ROOT / 'experiments' / 'outputs', Path.home() / 'Downloads']:
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


        def resolve_clap_cache() -> tuple[Path, Path]:
            meta_names = ('full_clap_meta.parquet', 'clap_meta.parquet')
            npz_names = ('full_clap_arrays.npz', 'clap_arrays.npz')

            explicit_candidates = []
            if CFG.clap_dir_override:
                explicit_root = Path(CFG.clap_dir_override).expanduser()
                explicit_candidates.append(explicit_root)
                explicit_candidates.extend([explicit_root / name for name in meta_names])

            search_roots = [ROOT / 'data', ROOT / 'processed_data', ROOT / 'experiments', Path.home() / 'Downloads']

            for candidate in explicit_candidates:
                if candidate.is_file() and candidate.name in meta_names:
                    parent = candidate.parent
                    for npz_name in npz_names:
                        npz_path = parent / npz_name
                        if npz_path.exists():
                            return candidate, npz_path
                elif candidate.is_dir():
                    for meta_name in meta_names:
                        meta_path = candidate / meta_name
                        if meta_path.exists():
                            for npz_name in npz_names:
                                npz_path = candidate / npz_name
                                if npz_path.exists():
                                    return meta_path, npz_path

            for search_root in search_roots:
                if not search_root.exists():
                    continue
                for meta_name in meta_names:
                    for meta_path in search_root.rglob(meta_name):
                        parent = meta_path.parent
                        for npz_name in npz_names:
                            npz_path = parent / npz_name
                            if npz_path.exists():
                                return meta_path, npz_path

            raise FileNotFoundError(
                'Could not resolve a CLAP cache. Expected something like '
                '`full_clap_meta.parquet + full_clap_arrays.npz` or '
                '`clap_meta.parquet + clap_arrays.npz`. Set CFG.clap_dir_override if needed.'
            )


        def detect_embedding_key(npz_obj: np.lib.npyio.NpzFile, expected_rows: int) -> str:
            preferred = ['emb_full', 'clap_emb_full', 'embeddings', 'audio_emb', 'audio_embeddings', 'emb']
            for key in preferred:
                if key in npz_obj and getattr(npz_obj[key], 'ndim', 0) == 2 and npz_obj[key].shape[0] == expected_rows:
                    return key
            for key in npz_obj.files:
                arr = npz_obj[key]
                if getattr(arr, 'ndim', 0) == 2 and arr.shape[0] == expected_rows:
                    return key
            raise KeyError(f'Could not detect CLAP embedding key for {expected_rows} rows. Available keys: {list(npz_obj.files)}')


        TEACHER_DIR = resolve_teacher_dir()
        CLAP_META_PATH, CLAP_NPZ_PATH = resolve_clap_cache()

        print({
            'teacher_dir': str(TEACHER_DIR),
            'clap_meta_path': str(CLAP_META_PATH),
            'clap_npz_path': str(CLAP_NPZ_PATH),
        })
        """
    ),
    code_cell(
        """
        taxonomy = pd.read_csv(DATA / 'taxonomy.csv')
        classes = taxonomy['primary_label'].astype(str).tolist()
        class_name_map = taxonomy.set_index('primary_label')['class_name'].to_dict()
        texture_idx = np.array([i for i, label in enumerate(classes) if class_name_map.get(label) in {'Amphibia', 'Insecta'}], dtype=np.int64)

        teacher_meta = pd.read_parquet(TEACHER_DIR / 'teacher_meta.parquet')
        teacher_arr = np.load(TEACHER_DIR / 'teacher_outputs.npz')
        teacher_probs = teacher_arr['teacher_probs'].astype(np.float32)
        teacher_labels = teacher_arr['labels'].astype(np.float32)

        clap_meta = pd.read_parquet(CLAP_META_PATH)
        clap_arr = np.load(CLAP_NPZ_PATH)
        clap_key = detect_embedding_key(clap_arr, len(clap_meta))
        clap_emb = clap_arr[clap_key].astype(np.float32)

        teacher_lookup = teacher_meta[['row_id']].copy()
        teacher_lookup['teacher_index'] = np.arange(len(teacher_lookup))
        clap_lookup = clap_meta[['row_id']].copy()
        clap_lookup['clap_index'] = np.arange(len(clap_lookup))
        aligned = teacher_lookup.merge(clap_lookup, on='row_id', how='inner')
        if len(aligned) != len(teacher_meta):
            raise ValueError(
                f'CLAP cache does not fully align to teacher cache: '
                f'aligned={len(aligned)} teacher_rows={len(teacher_meta)}'
            )

        aligned = aligned.sort_values('teacher_index').reset_index(drop=True)
        X_clap = clap_emb[aligned['clap_index'].to_numpy()]
        assert X_clap.shape[0] == teacher_probs.shape[0] == teacher_labels.shape[0]
        assert np.all(aligned['teacher_index'].to_numpy() == np.arange(len(teacher_meta)))

        setup_snapshot = {
            'experiment_id': CFG.experiment_id,
            'experiment_name': CFG.experiment_name,
            'teacher_rows': int(len(teacher_meta)),
            'teacher_files': int(teacher_meta['filename'].nunique()),
            'n_classes': int(len(classes)),
            'clap_embedding_key': clap_key,
            'clap_embedding_dim': int(X_clap.shape[1]),
            'folds': sorted(pd.Series(teacher_meta['fold']).dropna().astype(int).unique().tolist()),
        }
        save_json(setup_snapshot, OUTPUT_DIR / 'setup_snapshot.json')
        print(json.dumps(setup_snapshot, indent=2))
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


        def fit_clap_probe_oof(meta_df: pd.DataFrame, y_true: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
            unique_folds = sorted(pd.Series(meta_df['fold']).dropna().astype(int).unique().tolist())
            oof = np.zeros_like(y_true, dtype=np.float32)
            trained_counts = np.zeros(y_true.shape[1], dtype=np.int32)
            fold_rows: list[dict[str, tp.Any]] = []

            for fold in unique_folds:
                train_mask = meta_df['fold'].to_numpy() != fold
                valid_mask = meta_df['fold'].to_numpy() == fold

                X_train = X[train_mask]
                X_valid = X[valid_mask]
                y_train = y_true[train_mask]
                y_valid = y_true[valid_mask]

                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_valid_scaled = scaler.transform(X_valid)

                pca_dim = min(CFG.clap_pca_dim, X_train_scaled.shape[1], max(2, X_train_scaled.shape[0] - 1))
                if pca_dim < X_train_scaled.shape[1]:
                    pca = PCA(n_components=pca_dim, random_state=1099)
                    Z_train = pca.fit_transform(X_train_scaled)
                    Z_valid = pca.transform(X_valid_scaled)
                else:
                    Z_train = X_train_scaled
                    Z_valid = X_valid_scaled

                train_prior = np.clip(y_train.mean(axis=0), 1e-4, 1.0 - 1e-4).astype(np.float32)
                fold_pred = np.repeat(train_prior[None, :], y_valid.shape[0], axis=0).astype(np.float32)

                trained_this_fold = 0
                for class_idx in range(y_true.shape[1]):
                    yt = y_train[:, class_idx]
                    positives = int(yt.sum())
                    negatives = int(len(yt) - positives)
                    if positives < CFG.min_pos_rows or negatives < CFG.min_neg_rows:
                        continue
                    clf = LogisticRegression(
                        C=CFG.logreg_c,
                        max_iter=400,
                        solver='liblinear',
                        class_weight='balanced',
                    )
                    clf.fit(Z_train, yt)
                    fold_pred[:, class_idx] = clf.predict_proba(Z_valid)[:, 1].astype(np.float32)
                    trained_counts[class_idx] += 1
                    trained_this_fold += 1

                oof[valid_mask] = fold_pred
                fold_macro_auc, fold_scored = macro_auc_skip_empty(y_valid, fold_pred)
                fold_texture_auc, fold_texture_scored = macro_auc_skip_empty(y_valid[:, texture_idx], fold_pred[:, texture_idx])
                fold_rows.append({
                    'fold': int(fold),
                    'rows': int(valid_mask.sum()),
                    'files': int(meta_df.loc[valid_mask, 'filename'].nunique()),
                    'trained_classes_this_fold': int(trained_this_fold),
                    'macro_auc': fold_macro_auc,
                    'texture_macro_auc': fold_texture_auc,
                    'scored_classes': fold_scored,
                    'texture_scored_classes': fold_texture_scored,
                })

            return oof.astype(np.float32), pd.DataFrame(fold_rows), trained_counts


        clap_oof, fold_summary, trained_counts = fit_clap_probe_oof(teacher_meta, teacher_labels, X_clap)

        teacher_macro_auc, teacher_scored = macro_auc_skip_empty(teacher_labels, teacher_probs)
        clap_macro_auc, clap_scored = macro_auc_skip_empty(teacher_labels, clap_oof)
        teacher_texture_auc, teacher_texture_scored = macro_auc_skip_empty(teacher_labels[:, texture_idx], teacher_probs[:, texture_idx])
        clap_texture_auc, clap_texture_scored = macro_auc_skip_empty(teacher_labels[:, texture_idx], clap_oof[:, texture_idx])

        baseline_df = pd.DataFrame([
            {
                'variant': 'teacher_exp015d',
                'macro_auc': teacher_macro_auc,
                'texture_macro_auc': teacher_texture_auc,
                'scored_classes': teacher_scored,
                'texture_scored_classes': teacher_texture_scored,
            },
            {
                'variant': 'clap_probe_oof',
                'macro_auc': clap_macro_auc,
                'texture_macro_auc': clap_texture_auc,
                'scored_classes': clap_scored,
                'texture_scored_classes': clap_texture_scored,
            },
        ])
        display(baseline_df)
        display(fold_summary)
        """
    ),
    code_cell(
        """
        weight_rows: list[dict[str, tp.Any]] = []
        for w_clap in CFG.weight_grid:
            blend = (1.0 - w_clap) * teacher_probs + w_clap * clap_oof
            macro_auc, scored_classes = macro_auc_skip_empty(teacher_labels, blend)
            texture_macro_auc, texture_scored_classes = macro_auc_skip_empty(teacher_labels[:, texture_idx], blend[:, texture_idx])
            weight_rows.append({
                'w_clap': float(w_clap),
                'w_teacher': float(1.0 - w_clap),
                'macro_auc': macro_auc,
                'texture_macro_auc': texture_macro_auc,
                'scored_classes': scored_classes,
                'texture_scored_classes': texture_scored_classes,
            })

        weight_sweep = pd.DataFrame(weight_rows).sort_values(['macro_auc', 'texture_macro_auc'], ascending=[False, False]).reset_index(drop=True)
        best_w = float(weight_sweep.iloc[0]['w_clap'])
        best_blend = (1.0 - best_w) * teacher_probs + best_w * clap_oof
        display(weight_sweep.head(12))

        taxon_rows: list[dict[str, tp.Any]] = []
        class_rows: list[dict[str, tp.Any]] = []
        for taxon in sorted(set(class_name_map.values())):
            idx = np.array([i for i, label in enumerate(classes) if class_name_map.get(label) == taxon], dtype=np.int64)
            if len(idx) == 0:
                continue
            teacher_auc, teacher_sc = macro_auc_skip_empty(teacher_labels[:, idx], teacher_probs[:, idx])
            clap_auc, clap_sc = macro_auc_skip_empty(teacher_labels[:, idx], clap_oof[:, idx])
            blend_auc, blend_sc = macro_auc_skip_empty(teacher_labels[:, idx], best_blend[:, idx])
            taxon_rows.append({
                'taxon': taxon,
                'teacher_macro_auc': teacher_auc,
                'clap_macro_auc': clap_auc,
                'best_blend_macro_auc': blend_auc,
                'teacher_scored_classes': teacher_sc,
                'clap_scored_classes': clap_sc,
                'blend_scored_classes': blend_sc,
            })

        for idx, label in enumerate(classes):
            yt = teacher_labels[:, idx]
            if len(np.unique(yt)) < 2:
                continue
            teacher_auc = float(roc_auc_score(yt, teacher_probs[:, idx]))
            clap_auc = float(roc_auc_score(yt, clap_oof[:, idx]))
            blend_auc = float(roc_auc_score(yt, best_blend[:, idx]))
            class_rows.append({
                'primary_label': label,
                'taxon': class_name_map.get(label, 'Unknown'),
                'teacher_auc': teacher_auc,
                'clap_auc': clap_auc,
                'best_blend_auc': blend_auc,
                'clap_minus_teacher': float(clap_auc - teacher_auc),
                'blend_minus_teacher': float(blend_auc - teacher_auc),
                'trained_folds': int(trained_counts[idx]),
            })

        taxon_summary = pd.DataFrame(taxon_rows).sort_values('best_blend_macro_auc', ascending=False).reset_index(drop=True)
        classwise_gain = pd.DataFrame(class_rows).sort_values('blend_minus_teacher', ascending=False).reset_index(drop=True)
        display(taxon_summary)
        display(classwise_gain.head(20))

        baseline_df.to_csv(OUTPUT_DIR / 'baseline_summary.csv', index=False)
        fold_summary.to_csv(OUTPUT_DIR / 'fold_summary.csv', index=False)
        weight_sweep.to_csv(OUTPUT_DIR / 'weight_sweep.csv', index=False)
        taxon_summary.to_csv(OUTPUT_DIR / 'taxon_summary.csv', index=False)
        classwise_gain.to_csv(OUTPUT_DIR / 'classwise_gain.csv', index=False)

        report_snapshot = {
            'experiment_id': CFG.experiment_id,
            'experiment_name': CFG.experiment_name,
            'rows': int(len(teacher_meta)),
            'files': int(teacher_meta['filename'].nunique()),
            'teacher_macro_auc': float(teacher_macro_auc),
            'clap_macro_auc': float(clap_macro_auc),
            'teacher_texture_macro_auc': float(teacher_texture_auc),
            'clap_texture_macro_auc': float(clap_texture_auc),
            'best_weight_clap': float(best_w),
            'best_macro_auc': float(weight_sweep.iloc[0]['macro_auc']),
            'best_texture_macro_auc': float(weight_sweep.iloc[0]['texture_macro_auc']),
            'trained_classes_any_fold': int((trained_counts > 0).sum()),
            'clap_embedding_key': clap_key,
            'clap_embedding_dim': int(X_clap.shape[1]),
            'note': 'Teacher scores come from fixed exp_015d replay on trusted rows. This notebook is a scouting benchmark for new-family complementarity, not a deployable submit path.',
        }
        save_json(report_snapshot, OUTPUT_DIR / 'report_snapshot.json')
        print(json.dumps(report_snapshot, indent=2))
        """
    ),
    code_cell(
        """
        if (OUTPUT_DIR / 'report_snapshot.json').exists():
            snapshot = json.loads((OUTPUT_DIR / 'report_snapshot.json').read_text())
            print('Snapshot:')
            print(json.dumps(snapshot, indent=2))
            if (OUTPUT_DIR / 'weight_sweep.csv').exists():
                display(pd.read_csv(OUTPUT_DIR / 'weight_sweep.csv').head(12))
        else:
            print('No outputs yet. Provide a CLAP cache and run the notebook top-to-bottom.')
        """
    ),
]


payload = {
    "cells": cells,
    "metadata": copy.deepcopy(base_md),
    "nbformat": 4,
    "nbformat_minor": 5,
}

(NOTEBOOKS / "exp_028a_clap_perch_complementarity_benchmark.ipynb").write_text(
    json.dumps(payload, ensure_ascii=False, indent=1)
)

