import os

# Set before importing NumPy or PyTorch so DataLoader subprocesses do not mix
# MKL's INTEL threading runtime with PyTorch's libgomp runtime.
os.environ["MKL_THREADING_LAYER"] = "GNU"

import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from data_augmentation import SpatialAugmentation
from model import AIGCClassifier


# =============================================================================
# DATASET DIRECTORY LISTS
# =============================================================================
# Add training data by appending real and AI directories independently.
TRAIN_REAL_DIRS = [
    "/home/utakata/LLMs/techjam2026/SID/real",
    "/home/utakata/LLMs/techjam/data/com/real",
]

TRAIN_AI_DIRS = [
    "/home/utakata/LLMs/techjam2026/SID/real",
    "/home/utakata/LLMs/techjam/data/com/real",
]

# Entries at the same index form one named validation dataset. To add another
# dataset, append one name, one real directory, and one AI directory.
VALIDATION_NAMES = [
    "SID",
    "CompEval",
]

VAL_REAL_DIRS = [
    "/home/utakata/LLMs/techjam2026/SID/test/real",
    "/home/utakata/LLMs/techjam/data/com/val/real",
]

VAL_AI_DIRS = [
    "/home/utakata/LLMs/techjam2026/SID/test/full_synthetic",
    "/home/utakata/LLMs/techjam/data/com/val/ai",
]
# =============================================================================


IMAGE_SIZE = 224
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class AIGCDataset(Dataset):
    """Spatial-only real/AI image dataset."""

    def __init__(self, real_image_paths, ai_image_paths, transform=None):
        self.paths = [(path, 0) for path in real_image_paths] + [
            (path, 1) for path in ai_image_paths
        ]
        self.transform = transform
        self.eval_geometry = transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            antialias=True,
        )
        self.to_model_tensor = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=IMAGENET_MEAN,
                    std=IMAGENET_STD,
                ),
            ]
        )

    def __len__(self):
        return len(self.paths)

    @staticmethod
    def _load_rgb_image(path):
        with Image.open(path) as source:
            source = ImageOps.exif_transpose(source)
            source.load()
            has_transparency = (
                source.mode in {"RGBA", "LA"}
                or (source.mode == "P" and "transparency" in source.info)
            )

            if has_transparency:
                rgba = source.convert("RGBA")
                background = Image.new(
                    "RGBA",
                    rgba.size,
                    (255, 255, 255, 255),
                )
                return Image.alpha_composite(background, rgba).convert("RGB")

            return source.convert("RGB").copy()

    def __getitem__(self, index):
        path, label = self.paths[index]
        image = self._load_rgb_image(path)
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = self.eval_geometry(image)
        return self.to_model_tensor(image), label


def get_image_paths(folder_path):
    """Return a deterministic recursive list of supported images."""
    folder = Path(folder_path).expanduser()
    if not folder.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {folder}")

    paths = sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        raise ValueError(f"No supported images found under: {folder}")
    return paths


def collect_image_paths(directory_paths, group_name):
    """Combine directories deterministically and remove duplicate paths."""
    combined = []
    seen = set()

    for directory in directory_paths:
        found_paths = get_image_paths(directory)
        added = 0
        for path in found_paths:
            identity = path.resolve()
            if identity in seen:
                continue
            seen.add(identity)
            combined.append(path)
            added += 1

        print(
            f"  {group_name}: {directory} -> "
            f"{len(found_paths):,} found, {added:,} added"
        )

    if not combined:
        raise ValueError(f"No images collected for {group_name}")
    return combined


def _resolved_path_set(paths):
    return {path.resolve() for path in paths}


def check_class_overlap(real_paths, ai_paths, dataset_name):
    overlap = _resolved_path_set(real_paths).intersection(
        _resolved_path_set(ai_paths)
    )
    if overlap:
        example = next(iter(overlap))
        raise ValueError(
            f"{dataset_name} contains {len(overlap)} image(s) in both "
            f"classes. Example: {example}"
        )


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def binary_roc_auc(labels, scores):
    """Calculate tie-aware binary ROC AUC without a scikit-learn dependency."""
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    positive_count = int((labels == 1).sum())
    negative_count = int((labels == 0).sum())
    if positive_count == 0 or negative_count == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)

    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        # Ranks are one-based; tied values receive their average rank.
        average_rank = 0.5 * ((start + 1) + end)
        ranks[order[start:end]] = average_rank
        start = end

    positive_rank_sum = ranks[labels == 1].sum()
    mann_whitney_u = (
        positive_rank_sum
        - positive_count * (positive_count + 1) / 2.0
    )
    return float(mann_whitney_u / (positive_count * negative_count))


def validate(model, data_loader, criterion, device):
    model.eval()
    loss_sum = 0.0
    class_correct = {0: 0, 1: 0}
    class_total = {0: 0, 1: 0}
    all_probabilities = []
    all_labels = []

    with torch.inference_mode():
        for images, labels in data_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            float_labels = labels.float().unsqueeze(1)

            logits = model(images)
            loss = criterion(logits, float_labels)
            loss_sum += loss.item() * images.size(0)

            probabilities = torch.sigmoid(logits).reshape(-1)
            predictions = (probabilities >= 0.5).long()
            for class_id in (0, 1):
                mask = labels == class_id
                class_total[class_id] += mask.sum().item()
                class_correct[class_id] += (
                    predictions[mask] == labels[mask]
                ).sum().item()

            all_probabilities.extend(probabilities.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    real_accuracy = class_correct[0] / max(class_total[0], 1)
    ai_accuracy = class_correct[1] / max(class_total[1], 1)
    return {
        "loss": loss_sum / len(data_loader.dataset),
        "balanced_accuracy": 0.5 * (real_accuracy + ai_accuracy),
        "real_accuracy": real_accuracy,
        "ai_accuracy": ai_accuracy,
        "roc_auc": binary_roc_auc(all_labels, all_probabilities),
    }


def path_configuration():
    return {
        "train_real_dirs": [str(path) for path in TRAIN_REAL_DIRS],
        "train_ai_dirs": [str(path) for path in TRAIN_AI_DIRS],
        "validation_names": list(VALIDATION_NAMES),
        "validation_real_dirs": [str(path) for path in VAL_REAL_DIRS],
        "validation_ai_dirs": [str(path) for path in VAL_AI_DIRS],
    }


def atomic_torch_save(payload, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.temporary")
    try:
        torch.save(payload, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def save_checkpoints(
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    metrics,
    args,
):
    save_path = Path(args.save_path)

    # Raw state dict preserves compatibility with the original spatial model.
    atomic_torch_save(model.state_dict(), save_path)

    if args.training_checkpoint_path:
        training_path = Path(args.training_checkpoint_path)
        atomic_torch_save(
            {
                "checkpoint_type": "spatial_only_training",
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "validation_metrics": metrics,
                "arguments": vars(args),
                "path_configuration": path_configuration(),
            },
            training_path,
        )
        print(f"Saved optional full training checkpoint to {training_path}")

    metadata = {
        "positive_class": "ai_generated",
        "label_mapping": {"0": "real", "1": "ai_generated"},
        "score_meaning": "sigmoid(logit) = P(ai_generated)",
        "decision_rule": "ai_generated if score >= 0.5",
        "preprocessing": {
            "resize": [IMAGE_SIZE, IMAGE_SIZE],
            "color_mode": "RGB",
            "normalization_mean": IMAGENET_MEAN,
            "normalization_std": IMAGENET_STD,
        },
        "model": {
            "training_method": "spatial_only_supervised",
            "backbone": "convnext_small",
            "pretrained_initialization": "ImageNet-1K",
            "frequency_branch": False,
            "stal": False,
        },
        "training_data": path_configuration(),
        "selected_epoch": epoch,
        "selection_metric": args.selection_metric,
        "validation_metrics": metrics,
        "inference_script": "detect.py",
    }
    metadata_path = Path(f"{save_path}.meta.json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)

    print(f"Saved spatial-only inference weights to {save_path}")


def _build_balancing(args, class_counts, labels, device):
    imbalance_ratio = class_counts.min() / class_counts.max()
    balance_strategy = args.balance_strategy
    if balance_strategy == "auto":
        balance_strategy = (
            "none"
            if imbalance_ratio >= args.balance_tolerance
            else "loss"
        )

    sampler = None
    positive_class_weight = None
    if balance_strategy == "sampler":
        class_weights = 1.0 / class_counts
        sample_weights = torch.as_tensor(
            [class_weights[label] for label in labels],
            dtype=torch.double,
        )
        sampler_generator = torch.Generator()
        sampler_generator.manual_seed(args.seed)
        sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
            generator=sampler_generator,
        )
    elif balance_strategy == "loss":
        positive_class_weight = torch.tensor(
            [class_counts[0] / class_counts[1]],
            dtype=torch.float32,
            device=device,
        )

    return balance_strategy, sampler, positive_class_weight, imbalance_ratio


def _print_validation(name, metrics):
    print(
        f"  {name} validation | loss={metrics['loss']:.4f}, "
        f"balanced acc={metrics['balanced_accuracy']:.4f}, "
        f"real acc={metrics['real_accuracy']:.4f}, "
        f"AI acc={metrics['ai_accuracy']:.4f}, "
        f"ROC AUC={metrics['roc_auc']:.4f}"
    )


def build_path_sets():
    if not TRAIN_REAL_DIRS or not TRAIN_AI_DIRS:
        raise ValueError("TRAIN_REAL_DIRS and TRAIN_AI_DIRS cannot be empty")
    if not (
        len(VALIDATION_NAMES)
        == len(VAL_REAL_DIRS)
        == len(VAL_AI_DIRS)
    ):
        raise ValueError(
            "VALIDATION_NAMES, VAL_REAL_DIRS, and VAL_AI_DIRS must have "
            "the same length"
        )
    if not VALIDATION_NAMES:
        raise ValueError("At least one validation dataset is required")
    if len(set(VALIDATION_NAMES)) != len(VALIDATION_NAMES):
        raise ValueError("Every validation dataset name must be unique")

    print("Collecting training images:")
    train_real_paths = collect_image_paths(
        TRAIN_REAL_DIRS,
        "training real",
    )
    train_ai_paths = collect_image_paths(
        TRAIN_AI_DIRS,
        "training AI",
    )
    check_class_overlap(train_real_paths, train_ai_paths, "Training data")

    validation_path_sets = {}
    for name, real_directory, ai_directory in zip(
        VALIDATION_NAMES,
        VAL_REAL_DIRS,
        VAL_AI_DIRS,
    ):
        print(f"Collecting {name} validation images:")
        real_paths = collect_image_paths(
            [real_directory],
            f"{name} real",
        )
        ai_paths = collect_image_paths(
            [ai_directory],
            f"{name} AI",
        )
        check_class_overlap(real_paths, ai_paths, f"{name} validation")
        validation_path_sets[name] = (real_paths, ai_paths)

    training_identities = _resolved_path_set(
        train_real_paths + train_ai_paths
    )
    for name, (real_paths, ai_paths) in validation_path_sets.items():
        overlap = training_identities.intersection(
            _resolved_path_set(real_paths + ai_paths)
        )
        if overlap:
            example = next(iter(overlap))
            raise ValueError(
                f"Training data overlaps {name} validation by "
                f"{len(overlap)} image(s). Example: {example}"
            )

    return train_real_paths, train_ai_paths, validation_path_sets


def _loader_process_options(args, persistent):
    options = {
        "num_workers": args.num_workers,
        "persistent_workers": persistent and args.num_workers > 0,
    }
    if args.num_workers > 0:
        options["prefetch_factor"] = args.prefetch_factor
    return options


def train(args):
    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    torch.set_float32_matmul_precision("high")

    (
        train_real_paths,
        train_ai_paths,
        validation_path_sets,
    ) = build_path_sets()

    print(
        "Combined training size | "
        f"real={len(train_real_paths):,}, AI={len(train_ai_paths):,}"
    )
    for name, (real_paths, ai_paths) in validation_path_sets.items():
        print(
            f"{name} validation size | "
            f"real={len(real_paths):,}, AI={len(ai_paths):,}"
        )

    train_dataset = AIGCDataset(
        train_real_paths,
        train_ai_paths,
        transform=SpatialAugmentation(size=IMAGE_SIZE),
    )
    validation_datasets = {
        name: AIGCDataset(real_paths, ai_paths, transform=None)
        for name, (real_paths, ai_paths) in validation_path_sets.items()
    }

    labels = np.asarray([label for _, label in train_dataset.paths])
    class_counts = np.bincount(labels, minlength=2)
    if np.any(class_counts == 0):
        raise ValueError(
            f"Both classes are required; counts={class_counts.tolist()}"
        )

    (
        balance_strategy,
        sampler,
        positive_class_weight,
        imbalance_ratio,
    ) = _build_balancing(args, class_counts, labels, device)
    print(
        f"Class counts: real={class_counts[0]}, AI={class_counts[1]} | "
        f"minority/majority={imbalance_ratio:.4f} | "
        f"balancing={balance_strategy}"
    )

    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)
    common_loader_options = {
        "batch_size": args.batch_size,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker,
    }
    train_loader = DataLoader(
        train_dataset,
        sampler=sampler,
        shuffle=sampler is None,
        generator=loader_generator,
        **common_loader_options,
        **_loader_process_options(args, persistent=True),
    )
    validation_loaders = {
        name: DataLoader(
            dataset,
            shuffle=False,
            **common_loader_options,
            **_loader_process_options(args, persistent=False),
        )
        for name, dataset in validation_datasets.items()
    }

    model = AIGCClassifier(pretrained=True).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=positive_class_weight)

    backbone_parameters = list(model.backbone.features.parameters())
    backbone_parameter_ids = {
        id(parameter) for parameter in backbone_parameters
    }
    task_parameters = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in backbone_parameter_ids
    ]
    optimizer = optim.AdamW(
        [
            {"params": backbone_parameters, "lr": args.backbone_lr},
            {"params": task_parameters, "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs, 1),
    )

    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_selection_score = -1.0

    for epoch in range(args.epochs):
        model.train()
        loss_sum = 0.0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float().unsqueeze(1)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                logits = model(images)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

            loss_sum += loss.item() * images.size(0)

        scheduler.step()

        validation_metrics = {
            name: validate(model, loader, criterion, device)
            for name, loader in validation_loaders.items()
        }
        selection_score = float(
            np.mean(
                [
                    metrics[args.selection_metric]
                    for metrics in validation_metrics.values()
                ]
            )
        )
        checkpoint_metrics = {
            "selection_score": selection_score,
            "selection_metric": args.selection_metric,
            "validation": validation_metrics,
        }

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"train loss={loss_sum / len(train_loader.dataset):.4f} | "
            f"mean validation {args.selection_metric}="
            f"{selection_score:.4f}"
        )
        for name, metrics in validation_metrics.items():
            _print_validation(name, metrics)

        if selection_score > best_selection_score:
            best_selection_score = selection_score
            save_checkpoints(
                model,
                optimizer,
                scheduler,
                scaler,
                epoch + 1,
                checkpoint_metrics,
                args,
            )


def build_argument_parser():
    parser = argparse.ArgumentParser(
        description="Train a spatial-only ConvNeXt-Small AI-image detector"
    )
    # Dataset paths intentionally are not command-line arguments. Edit the
    # directory lists at the top of this file.
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--backbone_lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument(
        "--prefetch_factor",
        type=int,
        default=2,
        help="Batches prefetched by each worker; ignored with zero workers",
    )
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument(
        "--selection_metric",
        choices=("balanced_accuracy", "roc_auc"),
        default="roc_auc",
        help="Per-dataset metric averaged equally to select the checkpoint",
    )
    parser.add_argument(
        "--balance_strategy",
        choices=("auto", "none", "loss", "sampler"),
        default="auto",
        help=(
            "auto uses normal shuffling when classes are nearly balanced and "
            "weighted BCE otherwise; sampler oversamples with replacement"
        ),
    )
    parser.add_argument(
        "--balance_tolerance",
        type=float,
        default=0.95,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--save_path",
        default="weights.pth",
    )
    parser.add_argument(
        "--training_checkpoint_path",
        default=None,
        help=(
            "Optional full optimizer checkpoint. Omit it to save only the "
            "smaller inference weights."
        ),
    )
    return parser


def validate_arguments(parser, args):
    if args.batch_size < 1:
        parser.error("--batch_size must be positive")
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.num_workers < 0:
        parser.error("--num_workers cannot be negative")
    if args.prefetch_factor < 1:
        parser.error("--prefetch_factor must be positive")
    if args.grad_clip_norm <= 0:
        parser.error("--grad_clip_norm must be positive")
    if not 0.0 < args.balance_tolerance <= 1.0:
        parser.error("--balance_tolerance must lie in (0, 1]")


if __name__ == "__main__":
    argument_parser = build_argument_parser()
    parsed_args = argument_parser.parse_args()
    validate_arguments(argument_parser, parsed_args)
    train(parsed_args)
