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
        # Exp 033: Smart Audio Family Benchmark

        Small local scouting benchmark for the most structurally distinct remaining local top-solution family:
        can the `smart-audio-bird-detector` EfficientNet-style branch add any complementary signal over the fixed `exp_015d` teacher cache on trusted rows?
        """
    ),
    md_cell(
        """
        ## Plan

        1. Load the completed `exp_027a` trusted-row teacher cache.
        2. Load the local `smart-audio` checkpoints (`LB862.pt`, `LB872.pt`) and replay them on the same `5s` trusted windows.
        3. Compare `baseline`, `finetuned`, and notebook-style blended variants on local macro AUC.
        4. Run late `teacher + smart-audio` blend sweeps and see whether this new family shows any real complementarity before any larger port or Kaggle attempt.
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
        from concurrent.futures import ThreadPoolExecutor

        import librosa
        import numpy as np
        import pandas as pd
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        import timm
        import torchaudio.transforms as TA
        from sklearn.metrics import roc_auc_score

        try:
            from IPython.display import display
        except Exception:
            def display(obj: object) -> None:
                print(obj)

        try:
            from tqdm.auto import tqdm
        except Exception:
            def tqdm(x, **kwargs):
                return x

        warnings.filterwarnings("ignore", message=".*weights_only=False.*")


        def resolve_repo_root(start: Path | None = None) -> Path:
            current = (start or Path.cwd()).resolve()
            for candidate in [current, *current.parents]:
                if (candidate / "PROJECT_STATE.md").exists() and (candidate / "data").exists():
                    return candidate
            raise FileNotFoundError("Could not resolve repository root")


        @dataclass
        class Config:
            experiment_id: str = "exp_033"
            experiment_name: str = "smart_audio_family_benchmark"
            teacher_dir_override: str | None = None
            smart_model_dir_override: str | None = None
            output_dir_override: str | None = None
            max_files: int | None = None
            audio_workers: int = 4
            use_fp16: bool = False

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

            teacher_weight_grid: tuple[float, ...] = tuple(round(x, 2) for x in np.linspace(0.0, 1.0, 21))

            @property
            def samples(self) -> int:
                return int(self.sr * self.window_sec)


        CFG = Config()
        ROOT = resolve_repo_root()
        DATA = ROOT / "data" / "birdclef-2026"
        OUTPUT_DIR = (
            Path(CFG.output_dir_override).expanduser()
            if CFG.output_dir_override
            else ROOT / "experiments" / "outputs" / f"{CFG.experiment_id}_{CFG.experiment_name}"
        )
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        BLEND_TABLE = [
            ("finetuned_only", {"mode": "prob", "ft": 1.0, "base": 0.0}),
            ("baseline_only", {"mode": "prob", "ft": 0.0, "base": 1.0}),
            ("prob_ft80_base20", {"mode": "prob", "ft": 0.8, "base": 0.2}),
            ("prob_ft70_base30", {"mode": "prob", "ft": 0.7, "base": 0.3}),
            ("prob_ft50_base50", {"mode": "prob", "ft": 0.5, "base": 0.5}),
            ("logit_ft80_base20", {"mode": "logit", "ft": 0.8, "base": 0.2}),
            ("logit_ft70_base30", {"mode": "logit", "ft": 0.7, "base": 0.3}),
            ("logit_ft50_base50", {"mode": "logit", "ft": 0.5, "base": 0.5}),
        ]


        def save_json(payload: dict[str, tp.Any], path: Path) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, indent=2, sort_keys=False, default=str))


        device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

        print({
            "root": str(ROOT),
            "output_dir": str(OUTPUT_DIR),
            "device": str(device),
            "teacher_dir_override": CFG.teacher_dir_override,
            "smart_model_dir_override": CFG.smart_model_dir_override,
            "max_files": CFG.max_files,
        })
        """
    ),
    code_cell(
        """
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
            raise FileNotFoundError("Could not resolve exp_027a teacher cache")


        def has_smart_weights(path: Path) -> bool:
            return (path / "LB862.pt").exists() and (path / "LB872.pt").exists()


        def resolve_smart_model_dir() -> Path:
            candidates = []
            if CFG.smart_model_dir_override:
                candidates.append(Path(CFG.smart_model_dir_override).expanduser())
            candidates.extend([
                ROOT / "data" / "BirdCLEF-2026-model",
                ROOT / "data",
                Path.home() / "Downloads",
            ])
            for candidate in candidates:
                if candidate.is_dir() and has_smart_weights(candidate):
                    return candidate
            for search_root in [ROOT / "data", Path.home() / "Downloads"]:
                if not search_root.exists():
                    continue
                for weight_path in search_root.rglob("LB862.pt"):
                    parent = weight_path.parent
                    if has_smart_weights(parent):
                        return parent
            raise FileNotFoundError("Could not resolve smart-audio checkpoint directory with LB862.pt and LB872.pt")


        def remap_audio_path(path_str: str, filename: str) -> Path:
            candidate = Path(path_str)
            if candidate.exists():
                return candidate
            fallback = DATA / "train_soundscapes" / filename
            if fallback.exists():
                return fallback
            raise FileNotFoundError(f"Could not resolve local audio path for {filename}")


        TEACHER_DIR = resolve_teacher_dir()
        SMART_MODEL_DIR = resolve_smart_model_dir()

        teacher_meta = pd.read_parquet(TEACHER_DIR / "teacher_meta.parquet").reset_index(drop=True)
        teacher_npz = np.load(TEACHER_DIR / "teacher_outputs.npz")
        teacher_probs = teacher_npz["teacher_probs"].astype(np.float32)
        labels = teacher_npz["labels"].astype(np.float32)

        submission_template = pd.read_csv(DATA / "sample_submission.csv", nrows=1)
        BIRD_LIST = [str(col) for col in submission_template.columns[1:]]
        CFG.classes = len(BIRD_LIST)
        label2idx = {label: idx for idx, label in enumerate(BIRD_LIST)}

        train_meta = pd.read_csv(DATA / "train.csv", usecols=["primary_label", "class_name"])
        train_meta["primary_label"] = train_meta["primary_label"].astype(str)
        class_name_map = (
            train_meta[["primary_label", "class_name"]]
            .drop_duplicates("primary_label")
            .set_index("primary_label")["class_name"]
            .to_dict()
        )
        texture_taxa = {"Amphibia", "Insecta"}
        texture_idx = np.array(
            [i for i, label in enumerate(BIRD_LIST) if class_name_map.get(label, "Aves") in texture_taxa],
            dtype=np.int32,
        )

        if CFG.max_files is not None:
            keep_files = teacher_meta["filename"].drop_duplicates().tolist()[: int(CFG.max_files)]
            keep_mask = teacher_meta["filename"].isin(keep_files).to_numpy()
            teacher_meta = teacher_meta.loc[keep_mask].reset_index(drop=True)
            teacher_probs = teacher_probs[keep_mask]
            labels = labels[keep_mask]

        teacher_meta["local_audio_path"] = [
            str(remap_audio_path(path_str, filename))
            for path_str, filename in zip(teacher_meta["file_path"], teacher_meta["filename"])
        ]

        setup_snapshot = {
            "experiment_id": CFG.experiment_id,
            "experiment_name": CFG.experiment_name,
            "rows": int(len(teacher_meta)),
            "files": int(teacher_meta["filename"].nunique()),
            "teacher_dir": str(TEACHER_DIR),
            "smart_model_dir": str(SMART_MODEL_DIR),
            "device": str(device),
        }
        save_json(setup_snapshot, OUTPUT_DIR / "setup_snapshot.json")
        print(setup_snapshot)
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
            model = model.to(device).eval()
            return model


        def blend_predictions(base_prob, ft_prob, base_logit, ft_logit, spec):
            if spec["mode"] == "prob":
                return spec["ft"] * ft_prob + spec["base"] * base_prob
            logits = spec["ft"] * ft_logit + spec["base"] * base_logit
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
        baseline_net = load_weights(SMART_MODEL_DIR / "LB862.pt")
        finetuned_net = load_weights(SMART_MODEL_DIR / "LB872.pt")

        print({
            "baseline_weight": str(SMART_MODEL_DIR / "LB862.pt"),
            "finetuned_weight": str(SMART_MODEL_DIR / "LB872.pt"),
        })
        """
    ),
    code_cell(
        """
        def read_audio(path_str: str) -> tuple[str, np.ndarray]:
            audio, _ = librosa.load(path_str, sr=CFG.sr, mono=True)
            audio = np.nan_to_num(audio).astype(np.float32, copy=False)
            return path_str, audio


        grouped = teacher_meta.groupby("filename", sort=True)
        file_order = list(grouped.groups.keys())
        path_order = [teacher_meta.loc[grouped.groups[fname][0], "local_audio_path"] for fname in file_order]

        audio_lookup: dict[str, np.ndarray] = {}
        with ThreadPoolExecutor(max_workers=int(CFG.audio_workers)) as pool:
            for path_str, audio in tqdm(pool.map(read_audio, path_order), total=len(path_order), desc="Loading audio"):
                audio_lookup[path_str] = audio

        variant_rows: dict[str, list[np.ndarray]] = {name: [] for name, _ in BLEND_TABLE}
        row_ids: list[str] = []

        use_autocast = bool(CFG.use_fp16) and device.type in {"cuda", "mps"}

        with torch.no_grad():
            for filename in tqdm(file_order, desc="Smart-audio inference"):
                file_rows = teacher_meta.loc[grouped.groups[filename]].sort_values("start_sec").reset_index(drop=True)
                audio = audio_lookup[file_rows.loc[0, "local_audio_path"]]
                clips = []
                for _, row in file_rows.iterrows():
                    start = int(round(float(row["start_sec"]) * CFG.sr))
                    end = int(round(float(row["end_sec"]) * CFG.sr))
                    clip = audio[start:end]
                    if len(clip) < CFG.samples:
                        clip = np.pad(clip, (0, CFG.samples - len(clip)))
                    else:
                        clip = clip[: CFG.samples]
                    clips.append(clip.astype(np.float32, copy=False))
                    row_ids.append(str(row["row_id"]))

                tensor_audio = torch.from_numpy(np.stack(clips)).float().to(device)
                if use_autocast and device.type == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        mel = mel_builder(tensor_audio)
                        base_out = baseline_net(mel)
                        ft_out = finetuned_net(mel)
                else:
                    mel = mel_builder(tensor_audio)
                    base_out = baseline_net(mel)
                    ft_out = finetuned_net(mel)

                base_p = base_out["clipwise_prob"]
                ft_p = ft_out["clipwise_prob"]
                base_logit = base_out["clipwise_logit"]
                ft_logit = ft_out["clipwise_logit"]

                for variant_name, spec in BLEND_TABLE:
                    probs = blend_predictions(base_p, ft_p, base_logit, ft_logit, spec).detach().cpu().numpy()
                    probs = smooth_and_leak(probs)
                    variant_rows[variant_name].append(probs)

        expected_row_ids = teacher_meta["row_id"].astype(str).tolist()
        if row_ids != expected_row_ids:
            raise RuntimeError("Smart-audio inference rows did not align with teacher cache row order")

        variant_pred_map = {
            name: np.vstack(chunks).astype(np.float32)
            for name, chunks in variant_rows.items()
        }

        inference_snapshot = {
            "rows": int(len(row_ids)),
            "files": int(len(file_order)),
            "variants": list(variant_pred_map.keys()),
            "audio_workers": int(CFG.audio_workers),
            "use_fp16": bool(CFG.use_fp16),
        }
        save_json(inference_snapshot, OUTPUT_DIR / "inference_snapshot.json")
        print(inference_snapshot)
        """
    ),
    code_cell(
        """
        def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, int]:
            keep = (y_true.sum(axis=0) > 0) & ((1.0 - y_true).sum(axis=0) > 0)
            if keep.sum() == 0:
                return float("nan"), 0
            score = roc_auc_score(y_true[:, keep], y_score[:, keep], average="macro")
            return float(score), int(keep.sum())


        def safe_auc_subset(y_true: np.ndarray, y_score: np.ndarray, idx: np.ndarray) -> tuple[float, int]:
            if len(idx) == 0:
                return float("nan"), 0
            return safe_auc(y_true[:, idx], y_score[:, idx])


        variant_records = []
        for variant_name, preds in variant_pred_map.items():
            macro_auc, scored_classes = safe_auc(labels, preds)
            texture_auc, texture_scored = safe_auc_subset(labels, preds, texture_idx)
            variant_records.append({
                "variant_name": variant_name,
                "macro_auc": macro_auc,
                "scored_classes": scored_classes,
                "texture_macro_auc": texture_auc,
                "texture_scored_classes": texture_scored,
            })

        variant_results = pd.DataFrame(variant_records).sort_values(
            ["macro_auc", "texture_macro_auc", "variant_name"],
            ascending=[False, False, True],
        ).reset_index(drop=True)

        best_variant_name = str(variant_results.loc[0, "variant_name"])
        best_smart_probs = variant_pred_map[best_variant_name]

        weight_records = []
        best_blend = None
        for w_smart in CFG.teacher_weight_grid:
            w_teacher = 1.0 - float(w_smart)
            blended = w_teacher * teacher_probs + float(w_smart) * best_smart_probs
            macro_auc, scored_classes = safe_auc(labels, blended)
            texture_auc, texture_scored = safe_auc_subset(labels, blended, texture_idx)
            record = {
                "w_smart": float(w_smart),
                "w_teacher": float(w_teacher),
                "macro_auc": macro_auc,
                "scored_classes": scored_classes,
                "texture_macro_auc": texture_auc,
                "texture_scored_classes": texture_scored,
            }
            weight_records.append(record)
            if best_blend is None or macro_auc > best_blend["macro_auc"]:
                best_blend = record.copy()

        weight_sweep = pd.DataFrame(weight_records).sort_values("w_smart").reset_index(drop=True)

        taxon_records = []
        for taxon in sorted(train_meta["class_name"].dropna().unique()):
            idx = np.array([i for i, label in enumerate(BIRD_LIST) if class_name_map.get(label, "Aves") == taxon], dtype=np.int32)
            if len(idx) == 0:
                continue
            teacher_auc, teacher_scored = safe_auc_subset(labels, teacher_probs, idx)
            smart_auc, smart_scored = safe_auc_subset(labels, best_smart_probs, idx)
            blend_auc, blend_scored = safe_auc_subset(
                labels,
                (1.0 - float(best_blend["w_smart"])) * teacher_probs + float(best_blend["w_smart"]) * best_smart_probs,
                idx,
            )
            taxon_records.append({
                "taxon": taxon,
                "teacher_macro_auc": teacher_auc,
                "smart_macro_auc": smart_auc,
                "best_blend_macro_auc": blend_auc,
                "teacher_scored_classes": teacher_scored,
                "smart_scored_classes": smart_scored,
                "blend_scored_classes": blend_scored,
            })

        taxon_summary = pd.DataFrame(taxon_records).sort_values("teacher_macro_auc", ascending=False).reset_index(drop=True)

        classwise_records = []
        best_blended_probs = (1.0 - float(best_blend["w_smart"])) * teacher_probs + float(best_blend["w_smart"]) * best_smart_probs
        for ci, label in enumerate(BIRD_LIST):
            y = labels[:, ci]
            pos = int(y.sum())
            neg = int((1.0 - y).sum())
            if pos == 0 or neg == 0:
                continue
            teacher_auc = float(roc_auc_score(y, teacher_probs[:, ci]))
            smart_auc = float(roc_auc_score(y, best_smart_probs[:, ci]))
            blend_auc = float(roc_auc_score(y, best_blended_probs[:, ci]))
            classwise_records.append({
                "label": label,
                "taxon": class_name_map.get(label, "Aves"),
                "teacher_auc": teacher_auc,
                "smart_auc": smart_auc,
                "best_blend_auc": blend_auc,
                "smart_minus_teacher": smart_auc - teacher_auc,
                "blend_minus_teacher": blend_auc - teacher_auc,
                "positives": pos,
            })

        classwise_comparison = pd.DataFrame(classwise_records).sort_values("blend_minus_teacher", ascending=False).reset_index(drop=True)

        teacher_macro_auc, teacher_scored_classes = safe_auc(labels, teacher_probs)
        teacher_texture_auc, teacher_texture_scored = safe_auc_subset(labels, teacher_probs, texture_idx)

        report_snapshot = {
            "experiment_id": CFG.experiment_id,
            "experiment_name": CFG.experiment_name,
            "rows": int(len(teacher_meta)),
            "files": int(teacher_meta["filename"].nunique()),
            "teacher_macro_auc": float(teacher_macro_auc),
            "teacher_texture_macro_auc": float(teacher_texture_auc),
            "best_smart_variant": best_variant_name,
            "best_smart_macro_auc": float(variant_results.loc[0, "macro_auc"]),
            "best_smart_texture_macro_auc": float(variant_results.loc[0, "texture_macro_auc"]),
            "best_weight_smart": float(best_blend["w_smart"]),
            "best_weight_teacher": float(best_blend["w_teacher"]),
            "best_macro_auc": float(best_blend["macro_auc"]),
            "best_texture_macro_auc": float(best_blend["texture_macro_auc"]),
            "note": "This is a trusted-row scouting benchmark for a new family, not a deployable submit path.",
        }

        variant_results.to_csv(OUTPUT_DIR / "variant_results.csv", index=False)
        weight_sweep.to_csv(OUTPUT_DIR / "teacher_blend_weight_sweep.csv", index=False)
        taxon_summary.to_csv(OUTPUT_DIR / "taxon_summary.csv", index=False)
        classwise_comparison.to_csv(OUTPUT_DIR / "classwise_comparison.csv", index=False)
        save_json(report_snapshot, OUTPUT_DIR / "report_snapshot.json")

        print("Variant results:")
        display(variant_results)
        print("Teacher + smart blend sweep:")
        display(weight_sweep)
        print("Taxon summary:")
        display(taxon_summary)
        print("Snapshot:")
        print(json.dumps(report_snapshot, indent=2))
        """
    ),
]


payload = {
    "cells": cells,
    "metadata": copy.deepcopy(base_md),
    "nbformat": 4,
    "nbformat_minor": 5,
}

(NOTEBOOKS / "exp_033_smart_audio_family_benchmark.ipynb").write_text(
    json.dumps(payload, ensure_ascii=False, indent=1)
)
