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
        # Exp 034: Smart Audio Submit Benchmark

        First Kaggle-facing benchmark for the locally winning smart-audio family.
        Default path uses the trusted `finetuned_only` variant from `LB872.pt`.
        """
    ),
    md_cell(
        """
        ## Plan

        1. Resolve the competition data and the smart-audio checkpoint dataset.
        2. Build the exact inference manifest from `sample_submission.csv` row ids.
        3. Run the smart-audio EfficientNet model on each required `5s` window.
        4. Keep the reference notebook heuristics that were part of the winning local benchmark:
           - mel pipeline
           - temporal smoothing
           - file-level leakage
        5. Save `submission.csv` and a short runtime log for audit.
        """
    ),
    code_cell(
        """
        # Cell 0 — Kaggle input hints and submit config
        COMPETITION_HINT = "birdclef-2026"
        SMART_MODEL_HINT = "birdclef-2026-model"
        SMART_MODEL_DIR_OVERRIDE = None
        SMART_FINETUNED_WEIGHT_OVERRIDE = None
        SMART_BASELINE_WEIGHT_OVERRIDE = None

        SUBMIT_VARIANT = "finetuned_only"   # finetuned_only | baseline_only | prob_ft80_base20 | prob_ft70_base30 | prob_ft50_base50 | logit_ft80_base20 | logit_ft70_base30 | logit_ft50_base50
        IO_WORKERS = 4
        TORCH_INTRA_OP_THREADS = 4
        PRINT_INPUT_DIAGNOSTICS = True
        ALLOW_MISSING_TEST_AUDIO_FALLBACK = True
        LOCAL_SMOKE_TEST = False
        LOCAL_SMOKE_FILES = 2

        print({
            "COMPETITION_HINT": COMPETITION_HINT,
            "SMART_MODEL_HINT": SMART_MODEL_HINT,
            "SMART_MODEL_DIR_OVERRIDE": SMART_MODEL_DIR_OVERRIDE,
            "SMART_FINETUNED_WEIGHT_OVERRIDE": SMART_FINETUNED_WEIGHT_OVERRIDE,
            "SMART_BASELINE_WEIGHT_OVERRIDE": SMART_BASELINE_WEIGHT_OVERRIDE,
            "SUBMIT_VARIANT": SUBMIT_VARIANT,
            "IO_WORKERS": IO_WORKERS,
            "TORCH_INTRA_OP_THREADS": TORCH_INTRA_OP_THREADS,
            "PRINT_INPUT_DIAGNOSTICS": PRINT_INPUT_DIAGNOSTICS,
            "ALLOW_MISSING_TEST_AUDIO_FALLBACK": ALLOW_MISSING_TEST_AUDIO_FALLBACK,
            "LOCAL_SMOKE_TEST": LOCAL_SMOKE_TEST,
            "LOCAL_SMOKE_FILES": LOCAL_SMOKE_FILES,
        })
        """
    ),
    code_cell(
        """
        from __future__ import annotations

        import json
        import os
        import time
        import warnings
        import typing as tp
        from dataclasses import dataclass
        from pathlib import Path
        from concurrent.futures import ThreadPoolExecutor

        import librosa
        import numpy as np
        import pandas as pd
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        import timm
        import torchaudio.transforms as TA

        warnings.filterwarnings("ignore")

        START_TIME = time.time()


        def save_json(payload: dict[str, tp.Any], path: Path) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, indent=2, sort_keys=False, default=str))


        def candidate_input_roots() -> list[Path]:
            roots = []
            if Path("/kaggle/input").exists():
                roots.append(Path("/kaggle/input"))
            roots.extend([
                Path.cwd() / "data",
                Path("/Users/yaroslav/Documents/Kaggle/BirdCLEF_2026/data"),
            ])
            return roots


        def collect_weight_like_files(limit: int = 200) -> list[str]:
            matches = []
            for root in candidate_input_roots():
                if not root.exists():
                    continue
                for pattern in ("*.pt", "*.pth", "*.ckpt", "*.bin"):
                    matches.extend(str(p) for p in root.rglob(pattern))
            return sorted(set(matches))[:limit]


        def print_input_diagnostics() -> None:
            print("Input roots:")
            for root in candidate_input_roots():
                print(" -", str(root), "exists=", root.exists())
                if root.exists():
                    try:
                        children = sorted([p.name for p in root.iterdir()])[:100]
                        print("   children:", children)
                    except Exception as exc:
                        print("   could not list children:", repr(exc))
            weight_like = collect_weight_like_files(limit=200)
            print("Weight-like files under input roots:", weight_like)


        def resolve_competition_dir() -> Path:
            candidates = []
            for root in candidate_input_roots():
                if not root.exists():
                    continue
                if COMPETITION_HINT:
                    for p in root.iterdir():
                        if COMPETITION_HINT.lower() in p.name.lower() and p.is_dir():
                            candidates.append(p)
                candidates.append(root / "birdclef-2026")
                candidates.append(root / "competitions" / "birdclef-2026")

            for candidate in candidates:
                if candidate.exists() and (candidate / "sample_submission.csv").exists():
                    return candidate
            raise FileNotFoundError("Could not resolve competition directory with sample_submission.csv")


        def resolve_model_dir() -> Path:
            candidates = []
            direct_weights = []
            if SMART_FINETUNED_WEIGHT_OVERRIDE:
                direct_weights.append(Path(SMART_FINETUNED_WEIGHT_OVERRIDE).expanduser())
            if SMART_BASELINE_WEIGHT_OVERRIDE:
                direct_weights.append(Path(SMART_BASELINE_WEIGHT_OVERRIDE).expanduser())
            for weight_path in direct_weights:
                if weight_path.exists():
                    candidates.append(weight_path.parent)
            if SMART_MODEL_DIR_OVERRIDE:
                override = Path(SMART_MODEL_DIR_OVERRIDE).expanduser()
                if override.is_file() and override.name == "LB872.pt":
                    candidates.append(override.parent)
                elif override.is_dir():
                    candidates.append(override)
                    for weight_path in override.rglob("LB872.pt"):
                        candidates.append(weight_path.parent)
            for root in candidate_input_roots():
                if not root.exists():
                    continue
                if SMART_MODEL_HINT:
                    for p in root.rglob("*"):
                        if p.is_dir() and SMART_MODEL_HINT.lower() in p.name.lower():
                            candidates.append(p)
                            for weight_path in p.rglob("LB872.pt"):
                                candidates.append(weight_path.parent)
                for weight_path in root.rglob("LB872.pt"):
                    candidates.append(weight_path.parent)
                candidates.append(root / "BirdCLEF-2026-model")
                candidates.append(root / "datasets" / "tonylica" / "birdclef-2026-model")

            seen = set()
            ordered = []
            for candidate in candidates:
                key = str(candidate)
                if key in seen:
                    continue
                seen.add(key)
                ordered.append(candidate)

            for candidate in ordered:
                if candidate.exists() and (candidate / "LB872.pt").exists():
                    return candidate

            debug_matches = collect_weight_like_files(limit=200)
            print("Could not auto-resolve LB872.pt. Weight-like files:", debug_matches)
            raise FileNotFoundError(
                "Could not resolve smart-audio model directory with LB872.pt. "
                "Set SMART_MODEL_DIR_OVERRIDE to the attached dataset directory or "
                "SMART_FINETUNED_WEIGHT_OVERRIDE to the full LB872.pt path if needed."
            )


        def resolve_weight_paths(model_dir: Path) -> tuple[Path, Path | None]:
            ft_path = Path(SMART_FINETUNED_WEIGHT_OVERRIDE).expanduser() if SMART_FINETUNED_WEIGHT_OVERRIDE else model_dir / "LB872.pt"
            base_path = Path(SMART_BASELINE_WEIGHT_OVERRIDE).expanduser() if SMART_BASELINE_WEIGHT_OVERRIDE else model_dir / "LB862.pt"
            if not ft_path.exists():
                raise FileNotFoundError(f"Could not find finetuned smart-audio weight: {ft_path}")
            if SUBMIT_VARIANT != "finetuned_only" and not base_path.exists():
                raise FileNotFoundError(f"Could not find baseline smart-audio weight: {base_path}")
            return ft_path, base_path if base_path.exists() else None


        BLEND_TABLE = {
            "finetuned_only": {"mode": "prob", "ft": 1.0, "base": 0.0},
            "baseline_only": {"mode": "prob", "ft": 0.0, "base": 1.0},
            "prob_ft80_base20": {"mode": "prob", "ft": 0.8, "base": 0.2},
            "prob_ft70_base30": {"mode": "prob", "ft": 0.7, "base": 0.3},
            "prob_ft50_base50": {"mode": "prob", "ft": 0.5, "base": 0.5},
            "logit_ft80_base20": {"mode": "logit", "ft": 0.8, "base": 0.2},
            "logit_ft70_base30": {"mode": "logit", "ft": 0.7, "base": 0.3},
            "logit_ft50_base50": {"mode": "logit", "ft": 0.5, "base": 0.5},
        }
        if SUBMIT_VARIANT not in BLEND_TABLE:
            raise KeyError(f"Unknown SUBMIT_VARIANT: {SUBMIT_VARIANT}")
        BLEND_SPEC = BLEND_TABLE[SUBMIT_VARIANT]


        @dataclass
        class AudioConfig:
            sr: int = 32000
            window_sec: float = 5.0
            mel_bins: int = 224
            fft: int = 2048
            hop: int = 512
            fmin: int = 0
            fmax: int = 16000
            power: float = 2.0
            top_db: float = 80
            backbone: str = "tf_efficientnet_b0.ns_jft_in1k"
            classes: int = 234
            channels: int = 3
            gem_init: float = 3.0
            workers: int = int(IO_WORKERS)

            @property
            def samples(self) -> int:
                return int(self.sr * self.window_sec)


        CFG = AudioConfig()

        torch.set_num_threads(int(TORCH_INTRA_OP_THREADS))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if PRINT_INPUT_DIAGNOSTICS:
            print_input_diagnostics()

        COMPETITION_DIR = resolve_competition_dir()
        MODEL_DIR = resolve_model_dir()
        FINETUNED_WEIGHT_PATH, BASELINE_WEIGHT_PATH = resolve_weight_paths(MODEL_DIR)
        SAMPLE_SUB_PATH = COMPETITION_DIR / "sample_submission.csv"
        TEST_AUDIO_DIR = COMPETITION_DIR / "test_soundscapes"
        TRAIN_AUDIO_DIR = COMPETITION_DIR / "train_soundscapes"

        print({
            "competition_dir": str(COMPETITION_DIR),
            "model_dir": str(MODEL_DIR),
            "finetuned_weight": str(FINETUNED_WEIGHT_PATH),
            "baseline_weight": None if BASELINE_WEIGHT_PATH is None else str(BASELINE_WEIGHT_PATH),
            "device": str(device),
            "submit_variant": SUBMIT_VARIANT,
        })
        """
    ),
    code_cell(
        """
        sample_submission = pd.read_csv(SAMPLE_SUB_PATH)
        BIRD_LIST = [str(col) for col in sample_submission.columns[1:]]
        CFG.classes = len(BIRD_LIST)

        def resolve_audio_path(stem: str) -> Path:
            test_path = TEST_AUDIO_DIR / f"{stem}.ogg"
            if test_path.exists():
                return test_path
            train_path = TRAIN_AUDIO_DIR / f"{stem}.ogg"
            if train_path.exists():
                return train_path
            raise FileNotFoundError(f"Could not resolve audio file for stem={stem}")


        NO_AUDIO_FALLBACK = False
        NO_AUDIO_FALLBACK_REASON = None


        def build_submission_manifest() -> pd.DataFrame:
            global NO_AUDIO_FALLBACK, NO_AUDIO_FALLBACK_REASON
            manifest = sample_submission[["row_id"]].copy()
            row_parts = manifest["row_id"].str.rsplit("_", n=1, expand=True)
            manifest["stem"] = row_parts[0]
            manifest["end_sec"] = row_parts[1].astype(int)
            manifest["start_sec"] = manifest["end_sec"] - int(CFG.window_sec)
            audio_paths = []
            missing_stems = []
            for stem in manifest["stem"]:
                try:
                    audio_paths.append(str(resolve_audio_path(stem)))
                except FileNotFoundError:
                    audio_paths.append("")
                    missing_stems.append(str(stem))
            if missing_stems:
                unique_missing = sorted(set(missing_stems))
                if not ALLOW_MISSING_TEST_AUDIO_FALLBACK:
                    raise FileNotFoundError(
                        "Could not resolve audio files for sample_submission stems. "
                        f"First missing stems: {unique_missing[:10]}"
                    )
                NO_AUDIO_FALLBACK = True
                NO_AUDIO_FALLBACK_REASON = (
                    "sample_submission references test audio that is not mounted in this runtime; "
                    "writing sample_submission fallback for interactive/non-hidden run"
                )
                print("No audio fallback is active.")
                print("Missing stems count:", len(unique_missing))
                print("First missing stems:", unique_missing[:10])
            manifest["audio_path"] = audio_paths
            return manifest


        def build_local_smoke_manifest() -> pd.DataFrame:
            files = sorted(TRAIN_AUDIO_DIR.glob("*.ogg"))[: int(LOCAL_SMOKE_FILES)]
            if not files:
                raise FileNotFoundError("LOCAL_SMOKE_TEST=True but no train_soundscapes/*.ogg files were found")
            rows = []
            for path in files:
                stem = path.stem
                duration = librosa.get_duration(path=str(path))
                n_windows = max(1, int(duration // CFG.window_sec))
                for i in range(n_windows):
                    end_sec = int((i + 1) * CFG.window_sec)
                    rows.append({
                        "row_id": f"{stem}_{end_sec}",
                        "stem": stem,
                        "start_sec": end_sec - int(CFG.window_sec),
                        "end_sec": end_sec,
                        "audio_path": str(path),
                    })
            return pd.DataFrame(rows)


        if LOCAL_SMOKE_TEST:
            manifest = build_local_smoke_manifest()
            print("LOCAL_SMOKE_TEST is active: using train_soundscapes smoke manifest, not sample_submission rows.")
        else:
            manifest = build_submission_manifest()

        print({
            "rows": int(len(manifest)),
            "files": int(manifest["stem"].nunique()),
            "classes": int(CFG.classes),
            "local_smoke_test": bool(LOCAL_SMOKE_TEST),
            "no_audio_fallback": bool(NO_AUDIO_FALLBACK),
            "no_audio_fallback_reason": NO_AUDIO_FALLBACK_REASON,
        })
        manifest.head()
        """
    ),
    code_cell(
        """
        class GeMFreq(nn.Module):
            def __init__(self, p: float = 3.0):
                super().__init__()
                self.p = nn.Parameter(torch.tensor(float(p)))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                p = self.p.clamp(min=1.0)
                x = x.clamp(min=1e-6).pow(p)
                x = x.mean(dim=2)
                return x.pow(1.0 / p)


        class BirdModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = timm.create_model(
                    CFG.backbone,
                    pretrained=False,
                    in_chans=CFG.channels,
                    num_classes=0,
                    global_pool="",
                )
                feat = self.backbone.num_features
                self.gem_pool = GeMFreq(CFG.gem_init)
                self.head = nn.Module()
                self.head.fc = nn.Sequential(
                    nn.Linear(feat, feat),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                )
                self.head.att_conv = nn.Conv1d(feat, CFG.classes, 1)
                self.head.cls_conv = nn.Conv1d(feat, CFG.classes, 1)

            def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
                feat = self.backbone(x)
                pooled = self.gem_pool(feat)
                x = pooled.permute(0, 2, 1)
                x = self.head.fc(x)
                x = x.permute(0, 2, 1)
                att = torch.tanh(self.head.att_conv(x))
                att = F.softmax(att, dim=-1)
                cls = self.head.cls_conv(x)
                logits = (att * cls).sum(-1)
                probs = torch.sigmoid(logits)
                return {"clipwise_prob": probs, "clipwise_logit": logits}


        class MelBuilder(nn.Module):
            def __init__(self):
                super().__init__()
                self.mel = TA.MelSpectrogram(
                    sample_rate=CFG.sr,
                    n_fft=CFG.fft,
                    hop_length=CFG.hop,
                    n_mels=CFG.mel_bins,
                    f_min=CFG.fmin,
                    f_max=CFG.fmax,
                    power=CFG.power,
                )
                self.db = TA.AmplitudeToDB()

            @torch.no_grad()
            def forward(self, wav: torch.Tensor) -> torch.Tensor:
                wav = torch.nan_to_num(wav)
                mel = self.mel(wav)
                mel = self.db(mel)
                B = mel.shape[0]
                mel_flat = mel.reshape(B, -1)
                mmin = mel_flat.min(dim=1)[0].view(B, 1, 1)
                mmax = mel_flat.max(dim=1)[0].view(B, 1, 1)
                mel = (mel - mmin) / (mmax - mmin + 1e-7)
                mel = mel.unsqueeze(1).repeat(1, 3, 1, 1)
                return mel


        def load_weights(path: Path) -> BirdModel:
            try:
                ckpt = torch.load(path, map_location="cpu", weights_only=False)
            except TypeError:
                ckpt = torch.load(path, map_location="cpu")

            model = BirdModel()
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                model.load_state_dict(ckpt["model_state_dict"])
            else:
                model.load_state_dict(ckpt)
            return model.to(device).eval()


        def blend_predictions(base_prob, ft_prob, base_logit, ft_logit):
            if BLEND_SPEC["mode"] == "prob":
                return BLEND_SPEC["ft"] * ft_prob + BLEND_SPEC["base"] * base_prob
            logits = BLEND_SPEC["ft"] * ft_logit + BLEND_SPEC["base"] * base_logit
            return torch.sigmoid(logits)


        def smooth_and_leak(probs: np.ndarray) -> np.ndarray:
            out = probs.astype(np.float32, copy=True)
            n = out.shape[0]
            if n > 4:
                p = out ** 1.5
                kernel = np.array([0.05, 0.15, 0.6, 0.15, 0.05], dtype=np.float32)
                pad = np.pad(p, ((2, 2), (0, 0)), mode="edge")
                out = (
                    kernel[0] * pad[:-4]
                    + kernel[1] * pad[1:-3]
                    + kernel[2] * pad[2:-2]
                    + kernel[3] * pad[3:-1]
                    + kernel[4] * pad[4:]
                ) ** (1.0 / 1.5)
            file_max = np.max(out, axis=0, keepdims=True)
            out = out + 0.05 * file_max
            return np.clip(out, 0.0, 1.0).astype(np.float32)


        mel_builder = MelBuilder().to(device)
        finetuned_net = load_weights(FINETUNED_WEIGHT_PATH)
        baseline_net = None
        if SUBMIT_VARIANT != "finetuned_only":
            if BASELINE_WEIGHT_PATH is None:
                raise FileNotFoundError("Baseline weight is required for this SUBMIT_VARIANT but BASELINE_WEIGHT_PATH is None")
            baseline_net = load_weights(BASELINE_WEIGHT_PATH)

        print({
            "finetuned_weight": str(FINETUNED_WEIGHT_PATH),
            "baseline_weight": None if baseline_net is None else str(BASELINE_WEIGHT_PATH),
        })
        """
    ),
    code_cell(
        """
        if NO_AUDIO_FALLBACK:
            sub = sample_submission.copy()
            sub.to_csv("submission.csv", index=False)

            runtime_log = {
                "experiment_id": "exp_034",
                "submit_variant": SUBMIT_VARIANT,
                "competition_dir": str(COMPETITION_DIR),
                "model_dir": str(MODEL_DIR),
                "device": str(device),
                "rows": int(len(sub)),
                "files": int(manifest["stem"].nunique()),
                "audio_workers": int(CFG.workers),
                "torch_threads": int(TORCH_INTRA_OP_THREADS),
                "local_smoke_test": bool(LOCAL_SMOKE_TEST),
                "no_audio_fallback": True,
                "no_audio_fallback_reason": NO_AUDIO_FALLBACK_REASON,
                "elapsed_seconds": float(time.time() - START_TIME),
            }
            save_json(
                runtime_log,
                Path("/kaggle/working/exp_034_smart_audio_submit_logs.json")
                if Path("/kaggle/working").exists()
                else Path("exp_034_smart_audio_submit_logs.json"),
            )
            print("Saved sample_submission fallback as submission.csv")
            print(json.dumps(runtime_log, indent=2))
            print(sub.head())


        def read_audio(path_str: str) -> tuple[str, np.ndarray]:
            audio, _ = librosa.load(path_str, sr=CFG.sr, mono=True)
            return path_str, np.nan_to_num(audio).astype(np.float32, copy=False)


        unique_audio_paths = [] if NO_AUDIO_FALLBACK else manifest["audio_path"].drop_duplicates().tolist()
        audio_lookup: dict[str, np.ndarray] = {}
        with ThreadPoolExecutor(max_workers=int(CFG.workers)) as pool:
            iterator = pool.map(read_audio, unique_audio_paths)
            for path_str, audio in iterator:
                audio_lookup[path_str] = audio

        pred_chunks = []
        grouped = [] if NO_AUDIO_FALLBACK else manifest.groupby("stem", sort=False)

        with torch.no_grad():
            for stem, group in grouped:
                group = group.sort_values("end_sec").reset_index(drop=True)
                audio = audio_lookup[group.loc[0, "audio_path"]]
                clips = []
                for _, row in group.iterrows():
                    start = int(round(float(row["start_sec"]) * CFG.sr))
                    end = int(round(float(row["end_sec"]) * CFG.sr))
                    clip = audio[start:end]
                    if len(clip) < CFG.samples:
                        clip = np.pad(clip, (0, CFG.samples - len(clip)))
                    else:
                        clip = clip[: CFG.samples]
                    clips.append(clip.astype(np.float32, copy=False))

                tensor_audio = torch.from_numpy(np.stack(clips)).float().to(device)
                mel = mel_builder(tensor_audio)

                ft_out = finetuned_net(mel)
                if baseline_net is None:
                    probs = ft_out["clipwise_prob"].detach().cpu().numpy()
                else:
                    base_out = baseline_net(mel)
                    probs = blend_predictions(
                        base_out["clipwise_prob"],
                        ft_out["clipwise_prob"],
                        base_out["clipwise_logit"],
                        ft_out["clipwise_logit"],
                    ).detach().cpu().numpy()

                probs = smooth_and_leak(probs)
                pred_chunk = pd.DataFrame(probs, columns=BIRD_LIST)
                pred_chunk.insert(0, "row_id", group["row_id"].tolist())
                pred_chunks.append(pred_chunk)

        if NO_AUDIO_FALLBACK:
            pred_df = sample_submission.copy()
        elif pred_chunks:
            pred_df = pd.concat(pred_chunks, axis=0, ignore_index=True)
        else:
            raise RuntimeError("No prediction chunks were produced")

        if NO_AUDIO_FALLBACK:
            sub = pred_df.copy()
        elif LOCAL_SMOKE_TEST:
            sub = pred_df.copy()
        else:
            sub = sample_submission[["row_id"]].merge(pred_df, on="row_id", how="left")
        missing_rows = int(sub[BIRD_LIST].isna().any(axis=1).sum())
        if missing_rows > 0:
            raise RuntimeError(f"Submission has {missing_rows} missing prediction rows")

        sub.to_csv("submission.csv", index=False)

        runtime_log = {
            "experiment_id": "exp_034",
            "submit_variant": SUBMIT_VARIANT,
            "competition_dir": str(COMPETITION_DIR),
            "model_dir": str(MODEL_DIR),
            "device": str(device),
            "rows": int(len(sub)),
            "files": int(manifest["stem"].nunique()),
            "audio_workers": int(CFG.workers),
            "torch_threads": int(TORCH_INTRA_OP_THREADS),
            "local_smoke_test": bool(LOCAL_SMOKE_TEST),
            "no_audio_fallback": bool(NO_AUDIO_FALLBACK),
            "no_audio_fallback_reason": NO_AUDIO_FALLBACK_REASON,
            "elapsed_seconds": float(time.time() - START_TIME),
        }
        save_json(runtime_log, Path("/kaggle/working/exp_034_smart_audio_submit_logs.json") if Path("/kaggle/working").exists() else Path("exp_034_smart_audio_submit_logs.json"))

        print("Saved submission.csv")
        print(json.dumps(runtime_log, indent=2))
        sub.head()
        """
    ),
]


payload = {
    "cells": cells,
    "metadata": copy.deepcopy(base_md),
    "nbformat": 4,
    "nbformat_minor": 5,
}

(NOTEBOOKS / "kaggle_submission_exp_034_smart_audio_submit_benchmark.ipynb").write_text(
    json.dumps(payload, ensure_ascii=False, indent=1)
)
