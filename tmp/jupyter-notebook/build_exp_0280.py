from __future__ import annotations

import json
import textwrap
from pathlib import Path


ROOT = Path("/Users/yaroslav/Documents/Kaggle/BirdCLEF_2026")
NOTEBOOKS = ROOT / "notebooks"


def src(text: str) -> list[str]:
    text = textwrap.dedent(text).strip("\n")
    return [line + "\n" for line in text.splitlines()]


nb = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": src(
                """
                # Exp 0280 — CLAP Cache Build

                Build a `row_id`-aligned CLAP embedding cache for the same trusted soundscape rows used by `exp_027a`, so that `exp_028a_clap_perch_complementarity_benchmark.ipynb` can run without further format conversion.
                """
            ),
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": src(
                """
                # Config
                from pathlib import Path

                ROOT = Path("/Users/yaroslav/Documents/Kaggle/BirdCLEF_2026")

                class CFG:
                    experiment_id = "exp_0280"
                    experiment_name = "clap_cache_build"

                    teacher_dir_override = None
                    clap_model_dir_override = None
                    clap_model_hint = None
                    clap_model_id = "laion/larger_clap_general"

                    output_dir_override = None

                    target_sr = 48000
                    batch_size = 16
                    max_rows = None
                    use_fp16 = False

                OUTPUT_DIR = Path(CFG.output_dir_override).expanduser() if CFG.output_dir_override else ROOT / "experiments" / "outputs" / "exp_0280_clap_cache_build"
                OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

                print({
                    "experiment_id": CFG.experiment_id,
                    "teacher_dir_override": CFG.teacher_dir_override,
                    "clap_model_dir_override": CFG.clap_model_dir_override,
                    "clap_model_hint": CFG.clap_model_hint,
                    "clap_model_id": CFG.clap_model_id,
                    "target_sr": CFG.target_sr,
                    "batch_size": CFG.batch_size,
                    "max_rows": CFG.max_rows,
                    "output_dir": str(OUTPUT_DIR),
                })
                """
            ),
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": src(
                """
                # Optional dependency bootstrap
                import importlib.util
                import subprocess
                import sys

                required_modules = ["transformers", "tokenizers", "sentencepiece"]
                missing = [name for name in required_modules if importlib.util.find_spec(name) is None]

                if missing:
                    print(f"Installing missing CLAP dependencies: {missing}")
                    subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "pip",
                            "install",
                            "-q",
                            "transformers>=4.40,<5",
                            "tokenizers>=0.15,<1",
                            "sentencepiece",
                        ],
                        check=True,
                    )
                else:
                    print("CLAP dependencies already available")
                """
            ),
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": src(
                """
                # Imports and path resolution
                import json
                import time
                from pathlib import Path

                import numpy as np
                import pandas as pd
                import soundfile as sf
                import torch
                import torchaudio

                from tqdm.auto import tqdm

                try:
                    from transformers import ClapFeatureExtractor, ClapModel
                except ImportError as e:
                    raise ImportError(
                        "transformers is required for exp_0280. "
                        "Install it locally or run in an environment where it is available."
                    ) from e


                def save_json(obj, path: Path):
                    path.write_text(json.dumps(obj, indent=2, default=str))


                DATA = ROOT / "data" / "birdclef-2026"
                LOCAL_TRAIN_SOUNDSCAPES = DATA / "train_soundscapes"


                def has_teacher_cache(path: Path) -> bool:
                    return (path / "teacher_meta.parquet").exists() and (path / "teacher_outputs.npz").exists()


                def resolve_teacher_dir() -> Path:
                    candidates = []
                    if CFG.teacher_dir_override:
                        candidates.append(Path(CFG.teacher_dir_override).expanduser())
                    candidates.extend([
                        ROOT / "experiments" / "outputs" / "exp_027a_exp015d_teacher_cache",
                        Path.home() / "Downloads" / "exp_027a_exp015d_teacher_cache",
                    ])
                    for candidate in candidates:
                        if has_teacher_cache(candidate):
                            return candidate
                    for search_root in [ROOT / "experiments" / "outputs", Path.home() / "Downloads"]:
                        if not search_root.exists():
                            continue
                        for meta_path in search_root.rglob("teacher_meta.parquet"):
                            parent = meta_path.parent
                            if has_teacher_cache(parent) and "exp_027a" in str(parent):
                                return parent
                    raise FileNotFoundError(
                        "Could not resolve exp_027a teacher cache. "
                        "Run exp_027a first or set CFG.teacher_dir_override."
                    )


                def is_clap_model_dir(path: Path) -> bool:
                    if not path.is_dir():
                        return False
                    if not (path / "config.json").exists():
                        return False
                    has_weights = (path / "model.safetensors").exists() or (path / "pytorch_model.bin").exists()
                    has_audio_cfg = (path / "preprocessor_config.json").exists() or (path / "processor_config.json").exists()
                    return bool(has_weights and has_audio_cfg)


                def resolve_clap_model_source():
                    explicit = []
                    if CFG.clap_model_dir_override:
                        explicit.append(Path(CFG.clap_model_dir_override).expanduser())
                    search_roots = [
                        ROOT / "data",
                        ROOT / "models",
                        ROOT / "processed_data",
                        ROOT / "experiments",
                        Path.home() / "Downloads",
                        Path("/kaggle/input"),
                    ]
                    for candidate in explicit:
                        if is_clap_model_dir(candidate):
                            return candidate
                    for search_root in search_roots:
                        if not search_root.exists():
                            continue
                        for config_path in search_root.rglob("config.json"):
                            parent = config_path.parent
                            if CFG.clap_model_hint and CFG.clap_model_hint.lower() not in str(parent).lower():
                                continue
                            if is_clap_model_dir(parent):
                                return parent
                    return CFG.clap_model_id


                def resolve_audio_path(row: pd.Series) -> Path:
                    candidates = []
                    for col in ["file_path", "source_file_path"]:
                        raw = row.get(col)
                        if isinstance(raw, str) and raw:
                            candidates.append(Path(raw))
                    if LOCAL_TRAIN_SOUNDSCAPES.exists():
                        candidates.append(LOCAL_TRAIN_SOUNDSCAPES / str(row["filename"]))
                    kaggle_comp = Path("/kaggle/input/competitions/birdclef-2026/train_soundscapes") / str(row["filename"])
                    kaggle_flat = Path("/kaggle/input/birdclef-2026/train_soundscapes") / str(row["filename"])
                    candidates.extend([kaggle_comp, kaggle_flat])

                    for candidate in candidates:
                        if candidate.exists():
                            return candidate
                    raise FileNotFoundError(f"Could not resolve audio path for row_id={row['row_id']} filename={row['filename']}")


                def choose_device() -> torch.device:
                    if torch.cuda.is_available():
                        return torch.device("cuda")
                    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                        return torch.device("mps")
                    return torch.device("cpu")


                TEACHER_DIR = resolve_teacher_dir()
                CLAP_MODEL_SOURCE = resolve_clap_model_source()
                DEVICE = choose_device()

                print({
                    "teacher_dir": str(TEACHER_DIR),
                    "clap_model_source": str(CLAP_MODEL_SOURCE),
                    "device": str(DEVICE),
                })
                """
            ),
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": src(
                """
                # Load trusted rows and CLAP model
                teacher_meta = pd.read_parquet(TEACHER_DIR / "teacher_meta.parquet").copy()
                if CFG.max_rows is not None:
                    teacher_meta = teacher_meta.iloc[: int(CFG.max_rows)].reset_index(drop=True)

                if teacher_meta["row_id"].duplicated().any():
                    raise ValueError("teacher_meta row_id must be unique")

                feature_extractor = ClapFeatureExtractor.from_pretrained(CLAP_MODEL_SOURCE)
                model = ClapModel.from_pretrained(CLAP_MODEL_SOURCE).to(DEVICE)
                model.eval()

                setup_snapshot = {
                    "experiment_id": CFG.experiment_id,
                    "experiment_name": CFG.experiment_name,
                    "teacher_rows": int(len(teacher_meta)),
                    "teacher_files": int(teacher_meta["filename"].nunique()),
                    "clap_model_source": str(CLAP_MODEL_SOURCE),
                    "device": str(DEVICE),
                    "target_sr": int(CFG.target_sr),
                    "batch_size": int(CFG.batch_size),
                    "output_dir": str(OUTPUT_DIR),
                }
                save_json(setup_snapshot, OUTPUT_DIR / "setup_snapshot.json")
                print(json.dumps(setup_snapshot, indent=2))
                """
            ),
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": src(
                """
                # Build CLAP cache
                start_time = time.time()

                audio_cache = {}


                def load_audio(path: Path):
                    key = str(path)
                    if key not in audio_cache:
                        wave, sr = sf.read(path, dtype="float32", always_2d=False)
                        if wave.ndim == 2:
                            wave = wave.mean(axis=1)
                        audio_cache[key] = (wave.astype(np.float32, copy=False), int(sr))
                    return audio_cache[key]


                def extract_clip(row: pd.Series) -> np.ndarray:
                    path = resolve_audio_path(row)
                    wave, sr = load_audio(path)
                    start_sec = float(row["start_sec"])
                    end_sec = float(row["end_sec"])
                    start = max(0, int(round(start_sec * sr)))
                    end = min(len(wave), int(round(end_sec * sr)))
                    clip = wave[start:end]
                    if clip.size == 0:
                        clip = np.zeros(int(round((end_sec - start_sec) * sr)), dtype=np.float32)
                    if sr != int(CFG.target_sr):
                        clip_t = torch.from_numpy(clip[None, :])
                        clip = torchaudio.functional.resample(clip_t, sr, int(CFG.target_sr)).squeeze(0).numpy()
                    return clip.astype(np.float32, copy=False)


                emb_chunks = []
                iterator = range(0, len(teacher_meta), int(CFG.batch_size))
                iterator = tqdm(iterator, total=(len(teacher_meta) + int(CFG.batch_size) - 1) // int(CFG.batch_size), desc="CLAP batches")

                use_amp = bool(CFG.use_fp16 and DEVICE.type == "cuda")

                for start_idx in iterator:
                    batch_df = teacher_meta.iloc[start_idx:start_idx + int(CFG.batch_size)]
                    clips = [extract_clip(row) for _, row in batch_df.iterrows()]
                    inputs = feature_extractor(
                        raw_speech=clips,
                        sampling_rate=int(CFG.target_sr),
                        return_tensors="pt",
                        padding=True,
                    )
                    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

                    with torch.inference_mode():
                        if use_amp:
                            with torch.amp.autocast("cuda", enabled=True):
                                batch_emb = model.get_audio_features(**inputs)
                        else:
                            batch_emb = model.get_audio_features(**inputs)
                    emb_chunks.append(batch_emb.detach().cpu().numpy().astype(np.float32, copy=False))

                clap_emb = np.concatenate(emb_chunks, axis=0).astype(np.float32, copy=False)
                assert clap_emb.shape[0] == len(teacher_meta)

                clap_meta = teacher_meta.copy()
                clap_meta["clap_model_source"] = str(CLAP_MODEL_SOURCE)

                clap_meta.to_parquet(OUTPUT_DIR / "clap_meta.parquet", index=False)
                clap_meta.to_parquet(OUTPUT_DIR / "full_clap_meta.parquet", index=False)

                row_id_values = clap_meta["row_id"].astype(str).tolist()
                max_row_id_len = max((len(x) for x in row_id_values), default=1)
                row_id_array = np.asarray(row_id_values, dtype=f"<U{max_row_id_len}")

                np.savez_compressed(
                    OUTPUT_DIR / "clap_arrays.npz",
                    clap_emb_full=clap_emb,
                    row_ids=row_id_array,
                )
                np.savez_compressed(
                    OUTPUT_DIR / "full_clap_arrays.npz",
                    clap_emb_full=clap_emb,
                    row_ids=row_id_array,
                )

                report_snapshot = {
                    "experiment_id": CFG.experiment_id,
                    "experiment_name": CFG.experiment_name,
                    "rows": int(len(clap_meta)),
                    "files": int(clap_meta["filename"].nunique()),
                    "embedding_dim": int(clap_emb.shape[1]),
                    "device": str(DEVICE),
                    "target_sr": int(CFG.target_sr),
                    "batch_size": int(CFG.batch_size),
                    "elapsed_seconds": float(time.time() - start_time),
                    "clap_model_source": str(CLAP_MODEL_SOURCE),
                    "output_dir": str(OUTPUT_DIR),
                }
                save_json(report_snapshot, OUTPUT_DIR / "report_snapshot.json")
                print(json.dumps(report_snapshot, indent=2))
                """
            ),
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": src(
                """
                # Quick sanity check
                saved_meta = pd.read_parquet(OUTPUT_DIR / "clap_meta.parquet")
                saved_npz = np.load(OUTPUT_DIR / "clap_arrays.npz")

                print({
                    "saved_rows": int(len(saved_meta)),
                    "saved_files": int(saved_meta["filename"].nunique()),
                    "embedding_key": "clap_emb_full",
                    "embedding_shape": tuple(saved_npz["clap_emb_full"].shape),
                    "row_id_match": bool(np.all(saved_meta["row_id"].astype(str).to_numpy() == saved_npz["row_ids"].astype(str))),
                })
                """
            ),
        },
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.12",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


dst_path = NOTEBOOKS / "exp_0280_clap_cache_build.ipynb"
dst_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
print(f"Wrote {dst_path}")
