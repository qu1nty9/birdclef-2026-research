from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import timm
    import torchaudio.transforms as T
except ModuleNotFoundError:
    torch = None
    nn = None
    F = None
    timm = None
    T = None

try:
    import librosa
except ModuleNotFoundError:
    librosa = None

try:
    from sklearn.metrics import roc_auc_score
except ModuleNotFoundError:
    roc_auc_score = None


DEFAULT_BASELINE_CKPT = "LB862.pt"
DEFAULT_FINETUNED_CKPT = "LB872.pt"


@dataclass(frozen=True)
class BlendStrategy:
    name: str
    mode: str
    ft_weight: float
    base_weight: float
    apply_heuristics: bool = False


DEFAULT_STRATEGIES = [
    BlendStrategy("baseline_only", "prob", 0.0, 1.0, False),
    BlendStrategy("finetuned_only", "prob", 1.0, 0.0, False),
    BlendStrategy("prob_ft80_base20", "prob", 0.8, 0.2, False),
    BlendStrategy("prob_ft80_base20_plus_heuristics", "prob", 0.8, 0.2, True),
]


@dataclass
class ReferenceModelConfig:
    sr: int = 32_000
    chunk_duration: float = 5.0
    n_mels: int = 224
    n_fft: int = 2048
    hop_length: int = 512
    fmin: int = 0
    fmax: int = 16_000
    top_db: float = 80.0
    power: float = 2.0
    norm: str = "slaney"
    mel_scale: str = "htk"
    backbone: str = "tf_efficientnet_b0.ns_jft_in1k"
    num_classes: int = 234
    in_channels: int = 3
    dropout: float = 0.1
    drop_path_rate: float = 0.0
    gem_p_init: float = 3.0
    max_workers: int = 4

    @property
    def chunk_samples(self) -> int:
        return int(self.sr * self.chunk_duration)


@dataclass
class ValidationTargets:
    species: list[str]
    label_df: pd.DataFrame
    row_ids: list[str]
    required_end_seconds: dict[str, list[int]]
    soundscape_paths: list[Path]
    summary: dict[str, Any]


def parse_time_to_seconds(value: Any) -> int:
    text = str(value)
    parts = text.split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(float(parts[2]))
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(float(parts[1]))
    return int(float(text))


def parse_species_string(value: Any) -> list[str]:
    if pd.isna(value):
        return []
    return [item.strip() for item in str(value).split(";") if item.strip()]


def merge_species_strings(values: pd.Series) -> str:
    merged: list[str] = []
    seen: set[str] = set()
    for value in values:
        for label in parse_species_string(value):
            if label not in seen:
                merged.append(label)
                seen.add(label)
    return ";".join(merged)


def build_validation_targets(data_dir: Path) -> ValidationTargets:
    labels_path = data_dir / "train_soundscapes_labels.csv"
    sample_submission_path = data_dir / "sample_submission.csv"
    soundscape_dir = data_dir / "train_soundscapes"

    labels = pd.read_csv(labels_path)
    species = pd.read_csv(sample_submission_path).columns[1:].tolist()
    label_to_idx = {label: idx for idx, label in enumerate(species)}

    raw_rows = len(labels)
    labels = labels.drop_duplicates(
        subset=["filename", "start", "end", "primary_label"]
    ).copy()
    deduped_rows = len(labels)

    labels["end_sec"] = labels["end"].map(parse_time_to_seconds)
    labels["start_sec"] = labels["start"].map(parse_time_to_seconds)

    grouped = (
        labels.groupby(["filename", "start_sec", "end_sec"], as_index=False)["primary_label"]
        .agg(merge_species_strings)
        .sort_values(["filename", "end_sec"])
        .reset_index(drop=True)
    )

    vectors = np.zeros((len(grouped), len(species)), dtype=np.float32)
    row_ids: list[str] = []
    required_end_seconds: dict[str, list[int]] = defaultdict(list)

    for row_index, row in grouped.iterrows():
        stem = Path(row["filename"]).stem
        end_sec = int(row["end_sec"])
        row_ids.append(f"{stem}_{end_sec}")
        required_end_seconds[stem].append(end_sec)

        for label in parse_species_string(row["primary_label"]):
            if label in label_to_idx:
                vectors[row_index, label_to_idx[label]] = 1.0

    label_df = pd.DataFrame(vectors, columns=species, index=row_ids)
    label_df.index.name = "row_id"

    file_lookup = {path.stem: path for path in soundscape_dir.glob("*.ogg")}
    missing_files = sorted(set(required_end_seconds) - set(file_lookup))
    if missing_files:
        preview = ", ".join(missing_files[:5])
        raise FileNotFoundError(
            f"Missing {len(missing_files)} labeled soundscape files. First examples: {preview}"
        )

    soundscape_paths = [file_lookup[stem] for stem in sorted(required_end_seconds)]
    summary = {
        "raw_label_rows": raw_rows,
        "deduped_label_rows": deduped_rows,
        "unique_segment_rows": int(len(grouped)),
        "labeled_soundscape_files": len(soundscape_paths),
        "scored_species_with_local_positives": int((label_df.sum(axis=0) > 0).sum()),
        "max_segment_end_sec": int(grouped["end_sec"].max()),
    }
    return ValidationTargets(
        species=species,
        label_df=label_df,
        row_ids=row_ids,
        required_end_seconds={key: sorted(set(value)) for key, value in required_end_seconds.items()},
        soundscape_paths=soundscape_paths,
        summary=summary,
    )


def require_full_ml_stack() -> None:
    missing = []
    if torch is None or nn is None or F is None:
        missing.append("torch")
    if timm is None:
        missing.append("timm")
    if T is None:
        missing.append("torchaudio")
    if librosa is None:
        missing.append("librosa")
    if roc_auc_score is None:
        missing.append("scikit-learn")
    if missing:
        joined = ", ".join(sorted(set(missing)))
        raise ModuleNotFoundError(
            f"Full evaluation requires these packages: {joined}"
        )


def safe_load_checkpoint(path: Path) -> Any:
    assert torch is not None
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def extract_state_dict(ckpt: Any) -> Any:
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    return ckpt


def extract_checkpoint_meta(ckpt: Any) -> dict[str, Any]:
    if not isinstance(ckpt, dict):
        return {"stage": "raw_state_dict"}
    metrics = ckpt.get("metrics", {})
    return {
        "epoch": ckpt.get("epoch"),
        "stage": ckpt.get("stage"),
        "metrics": metrics if isinstance(metrics, dict) else {},
    }


def build_reference_model_components(cfg: ReferenceModelConfig) -> tuple[Any, Any]:
    assert nn is not None and F is not None and timm is not None and T is not None

    class GEMFreqPool(nn.Module):
        def __init__(self, p_init: float = 3.0, eps: float = 1e-6) -> None:
            super().__init__()
            self.p = nn.Parameter(torch.tensor(p_init))
            self.eps = eps

        def forward(self, x: Any) -> Any:
            p = self.p.clamp(min=1.0)
            x = x.clamp(min=self.eps).pow(p)
            x = x.mean(dim=2)
            return x.pow(1.0 / p)

    class AttentionSEDHead(nn.Module):
        def __init__(self, feat_dim: int, num_classes: int, dropout: float = 0.1) -> None:
            super().__init__()
            self.fc = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
            self.att_conv = nn.Conv1d(feat_dim, num_classes, kernel_size=1)
            self.cls_conv = nn.Conv1d(feat_dim, num_classes, kernel_size=1)

        def forward(self, x: Any) -> dict[str, Any]:
            x = x.permute(0, 2, 1)
            x = self.fc(x)
            x = x.permute(0, 2, 1)
            att = torch.tanh(self.att_conv(x))
            att = F.softmax(att, dim=-1)
            cls = self.cls_conv(x)
            clipwise_logit = (att * cls).sum(dim=-1)
            clipwise_prob = torch.sigmoid(clipwise_logit)
            return {
                "clipwise_logit": clipwise_logit,
                "clipwise_prob": clipwise_prob,
            }

    class SEDModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = timm.create_model(
                cfg.backbone,
                pretrained=False,
                in_chans=cfg.in_channels,
                features_only=False,
                global_pool="",
                num_classes=0,
                drop_path_rate=cfg.drop_path_rate,
            )
            feat_dim = self.backbone.num_features
            self.gem_pool = GEMFreqPool(p_init=cfg.gem_p_init)
            self.head = AttentionSEDHead(feat_dim, cfg.num_classes, cfg.dropout)

        def forward(self, x: Any) -> dict[str, Any]:
            features = self.backbone(x)
            pooled = self.gem_pool(features)
            return self.head(pooled)

    class MelSpectrogramTransform(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mel = T.MelSpectrogram(
                sample_rate=cfg.sr,
                n_fft=cfg.n_fft,
                hop_length=cfg.hop_length,
                n_mels=cfg.n_mels,
                f_min=cfg.fmin,
                f_max=cfg.fmax,
                power=cfg.power,
                norm=cfg.norm,
                mel_scale=cfg.mel_scale,
            )
            self.db = T.AmplitudeToDB(stype="power", top_db=cfg.top_db)

        @torch.no_grad()
        def forward(self, waveforms: Any) -> Any:
            waveforms = torch.nan_to_num(waveforms.float(), nan=0.0, posinf=0.0, neginf=0.0)
            mel = self.mel(waveforms)
            mel = torch.nan_to_num(mel, nan=0.0, posinf=0.0, neginf=0.0)
            mel = self.db(mel)
            mel = torch.nan_to_num(mel, nan=-80.0, posinf=0.0, neginf=-80.0)

            batch_size = mel.shape[0]
            mel_flat = mel.reshape(batch_size, -1)
            mel_min = mel_flat.min(dim=1, keepdim=True)[0].unsqueeze(-1)
            mel_max = mel_flat.max(dim=1, keepdim=True)[0].unsqueeze(-1)

            mel = (mel - mel_min) / (mel_max - mel_min + 1e-7)
            mel = torch.nan_to_num(mel, nan=0.0, posinf=1.0, neginf=0.0)
            return mel.unsqueeze(1).repeat(1, 3, 1, 1)

    return SEDModel, MelSpectrogramTransform


def load_model(
    cfg: ReferenceModelConfig, ckpt_path: Path, device: Any
) -> tuple[Any, dict[str, Any]]:
    SEDModel, _ = build_reference_model_components(cfg)
    model = SEDModel()
    checkpoint = safe_load_checkpoint(ckpt_path)
    model.load_state_dict(extract_state_dict(checkpoint), strict=True)
    model.to(device).eval()
    return model, extract_checkpoint_meta(checkpoint)


def build_mel_transform(cfg: ReferenceModelConfig, device: Any) -> Any:
    _, MelSpectrogramTransform = build_reference_model_components(cfg)
    return MelSpectrogramTransform().to(device).eval()


def load_soundscape_audio(path: Path, sample_rate: int) -> np.ndarray:
    assert librosa is not None
    audio, _ = librosa.load(path, sr=sample_rate, mono=True)
    return np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def blend_predictions(
    base_logits: Any,
    ft_logits: Any,
    base_probs: Any,
    ft_probs: Any,
    strategy: BlendStrategy,
) -> np.ndarray:
    assert torch is not None
    if strategy.mode == "prob":
        probs = strategy.ft_weight * ft_probs + strategy.base_weight * base_probs
    elif strategy.mode == "logit":
        logits = strategy.ft_weight * ft_logits + strategy.base_weight * base_logits
        probs = torch.sigmoid(logits)
    else:
        raise ValueError(f"Unknown blend mode: {strategy.mode}")

    probs = torch.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
    probs = torch.clamp(probs, 0.0, 1.0)
    return probs.detach().cpu().numpy()


def apply_reference_heuristics(probs: np.ndarray) -> np.ndarray:
    if len(probs) > 4:
        sharpen_power = 1.5
        probs_sharp = probs ** sharpen_power
        weights = np.array([0.05, 0.15, 0.60, 0.15, 0.05], dtype=np.float32)
        padded = np.pad(probs_sharp, ((2, 2), (0, 0)), mode="edge")
        smoothed = (
            weights[0] * padded[:-4]
            + weights[1] * padded[1:-3]
            + weights[2] * padded[2:-2]
            + weights[3] * padded[3:-1]
            + weights[4] * padded[4:]
        )
        probs = smoothed ** (1.0 / sharpen_power)
    elif len(probs) > 2:
        weights = np.array([0.20, 0.60, 0.20], dtype=np.float32)
        padded = np.pad(probs, ((1, 1), (0, 0)), mode="edge")
        probs = (
            weights[0] * padded[:-2]
            + weights[1] * padded[1:-1]
            + weights[2] * padded[2:]
        )

    file_max = np.max(probs, axis=0, keepdims=True)
    return probs + 0.05 * file_max


def select_strategies(requested_names: list[str] | None) -> list[BlendStrategy]:
    available = {strategy.name: strategy for strategy in DEFAULT_STRATEGIES}
    if not requested_names:
        return DEFAULT_STRATEGIES
    missing = [name for name in requested_names if name not in available]
    if missing:
        missing_text = ", ".join(missing)
        raise KeyError(f"Unknown strategies requested: {missing_text}")
    return [available[name] for name in requested_names]


def score_macro_auc(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    species: list[str],
) -> dict[str, Any]:
    assert roc_auc_score is not None
    per_class_scores: dict[str, float] = {}
    skipped_no_positive: list[str] = []
    skipped_no_negative: list[str] = []

    for class_index, label in enumerate(species):
        target = y_true[:, class_index]
        pred = y_pred[:, class_index]

        if target.sum() == 0:
            skipped_no_positive.append(label)
            continue
        if target.sum() == len(target):
            skipped_no_negative.append(label)
            continue

        per_class_scores[label] = float(roc_auc_score(target, pred))

    macro_auc = float(np.mean(list(per_class_scores.values()))) if per_class_scores else float("nan")
    return {
        "macro_auc": macro_auc,
        "scored_classes": len(per_class_scores),
        "skipped_no_positive": len(skipped_no_positive),
        "skipped_no_negative": len(skipped_no_negative),
        "skipped_no_positive_labels": skipped_no_positive,
        "skipped_no_negative_labels": skipped_no_negative,
        "per_class_auc": per_class_scores,
    }


def run_reference_evaluation(
    data_dir: Path,
    model_dir: Path,
    output_dir: Path,
    device_name: str,
    requested_strategies: list[str] | None,
    limit_files: int | None,
    save_predictions: bool,
) -> dict[str, Any]:
    require_full_ml_stack()
    targets = build_validation_targets(data_dir)
    strategies = select_strategies(requested_strategies)

    cfg = ReferenceModelConfig(num_classes=len(targets.species))
    device = torch.device(device_name if device_name != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))

    baseline_path = model_dir / DEFAULT_BASELINE_CKPT
    finetuned_path = model_dir / DEFAULT_FINETUNED_CKPT
    if not baseline_path.exists():
        raise FileNotFoundError(f"Missing baseline checkpoint: {baseline_path}")
    if not finetuned_path.exists():
        raise FileNotFoundError(f"Missing finetuned checkpoint: {finetuned_path}")

    baseline_model, baseline_meta = load_model(cfg, baseline_path, device)
    finetuned_model, finetuned_meta = load_model(cfg, finetuned_path, device)
    mel_transform = build_mel_transform(cfg, device)

    row_index = {row_id: idx for idx, row_id in enumerate(targets.row_ids)}
    strategy_predictions = {
        strategy.name: np.zeros_like(targets.label_df.values, dtype=np.float32)
        for strategy in strategies
    }

    soundscape_paths = targets.soundscape_paths
    if limit_files is not None:
        soundscape_paths = soundscape_paths[:limit_files]

    with torch.no_grad():
        for path in soundscape_paths:
            stem = path.stem
            required_end_seconds = targets.required_end_seconds[stem]
            n_chunks = max(required_end_seconds) // int(cfg.chunk_duration)

            audio = load_soundscape_audio(path, cfg.sr)
            padded_len = n_chunks * cfg.chunk_samples
            if len(audio) < padded_len:
                audio = np.pad(audio, (0, padded_len - len(audio)))
            else:
                audio = audio[:padded_len]

            chunks = audio.reshape(n_chunks, cfg.chunk_samples)
            chunk_tensor = torch.from_numpy(chunks).float().to(device)
            mel = mel_transform(chunk_tensor)

            base_out = baseline_model(mel)
            ft_out = finetuned_model(mel)

            base_probs = torch.nan_to_num(base_out["clipwise_prob"], nan=0.0, posinf=1.0, neginf=0.0)
            ft_probs = torch.nan_to_num(ft_out["clipwise_prob"], nan=0.0, posinf=1.0, neginf=0.0)

            row_ids_for_file = [
                f"{stem}_{(chunk_idx + 1) * int(cfg.chunk_duration)}"
                for chunk_idx in range(n_chunks)
            ]

            for strategy in strategies:
                probs = blend_predictions(
                    base_logits=base_out["clipwise_logit"],
                    ft_logits=ft_out["clipwise_logit"],
                    base_probs=base_probs,
                    ft_probs=ft_probs,
                    strategy=strategy,
                )
                if strategy.apply_heuristics:
                    probs = apply_reference_heuristics(probs)

                for local_idx, row_id in enumerate(row_ids_for_file):
                    if row_id not in row_index:
                        continue
                    strategy_predictions[strategy.name][row_index[row_id]] = probs[local_idx]

    metrics_by_strategy: dict[str, Any] = {}
    summary_rows: list[dict[str, Any]] = []

    for strategy in strategies:
        preds = strategy_predictions[strategy.name]
        metrics = score_macro_auc(targets.label_df.values, preds, targets.species)
        metrics_by_strategy[strategy.name] = metrics
        summary_rows.append(
            {
                "strategy": strategy.name,
                "macro_auc": metrics["macro_auc"],
                "scored_classes": metrics["scored_classes"],
                "skipped_no_positive": metrics["skipped_no_positive"],
                "skipped_no_negative": metrics["skipped_no_negative"],
            }
        )

        if save_predictions:
            pred_df = pd.DataFrame(preds, columns=targets.species, index=targets.row_ids)
            pred_df.index.name = "row_id"
            pred_df.reset_index().to_csv(
                output_dir / f"{strategy.name}_predictions.csv",
                index=False,
            )

    summary_df = pd.DataFrame(summary_rows).sort_values("macro_auc", ascending=False)
    summary_df.to_csv(output_dir / "strategy_summary.csv", index=False)

    result = {
        "device": str(device),
        "targets_summary": targets.summary,
        "baseline_checkpoint": {
            "path": str(baseline_path),
            "meta": baseline_meta,
        },
        "finetuned_checkpoint": {
            "path": str(finetuned_path),
            "meta": finetuned_meta,
        },
        "strategies": [strategy.__dict__ for strategy in strategies],
        "metrics_by_strategy": metrics_by_strategy,
    }

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    return result


def build_dry_run_payload(
    targets: ValidationTargets,
    data_dir: Path,
    model_dir: Path,
) -> dict[str, Any]:
    return {
        "data_dir": str(data_dir),
        "model_dir": str(model_dir),
        "targets_summary": targets.summary,
        "first_row_ids": targets.row_ids[:10],
        "first_soundscape_files": [str(path) for path in targets.soundscape_paths[:10]],
    }


def metrics_to_summary_frame(metrics_by_strategy: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for strategy_name, metrics in metrics_by_strategy.items():
        rows.append(
            {
                "strategy": strategy_name,
                "macro_auc": metrics["macro_auc"],
                "scored_classes": metrics["scored_classes"],
                "skipped_no_positive": metrics["skipped_no_positive"],
                "skipped_no_negative": metrics["skipped_no_negative"],
            }
        )
    return pd.DataFrame(rows).sort_values("macro_auc", ascending=False).reset_index(drop=True)
