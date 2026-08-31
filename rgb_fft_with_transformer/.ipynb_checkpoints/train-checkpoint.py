import os

# Set this before importing NumPy or PyTorch. It prevents DataLoader workers
# from mixing MKL's INTEL threading runtime with PyTorch's libgomp runtime.
os.environ["MKL_THREADING_LAYER"] = "GNU"

import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from data_augmentation import FrequencyAwareAugmentation
from model import Full2DFrequencyFusionClassifier


# =============================================================================
# DATASET DIRECTORY LISTS
# =============================================================================
# Add training datasets by appending their real and AI directories to these
# lists. The real and AI list lengths do not have to match.
TRAIN_REAL_DIRS = [
    "/home/utakata/LLMs/techjam2026/SID/real",
    "/home/utakata/LLMs/techjam/data/com/real",
]

TRAIN_AI_DIRS = [
    "/home/utakata/LLMs/techjam2026/SID/ai",
    "/home/utakata/LLMs/techjam/data/com/ai",
]

# Validation entries at the same index form one named validation dataset.
# To add a dataset, append one item to each of these three lists.
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
    """Return spatial and full-spectrum inputs from exactly the same pixels."""

    def __init__(self, real_image_paths, fake_image_paths, transform=None):
        self.paths = [(path, 0) for path in real_image_paths] + [
            (path, 1) for path in fake_image_paths
        ]
        self.transform = transform
        self.eval_geometry = transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            antialias=True,
        )
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        )

    def __len__(self):
        return len(self.paths)

    @staticmethod
    def _load_rgb_image(path):
        with Image.open(path) as source:
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

            return source.convert("RGB")

    def __getitem__(self, index):
        path, label = self.paths[index]
        image = self._load_rgb_image(path)

        if self.transform is not None:
            view = self.transform(image)
        else:
            view = self.eval_geometry(image)

        # Both branches see precisely the same augmented pixels. Only the
        # spatial copy receives ImageNet normalization.
        frequency_tensor = self.to_tensor(view)
        spatial_tensor = self.normalize(frequency_tensor)
        return spatial_tensor, frequency_tensor, label


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
    """Combine directory lists deterministically and remove duplicate files."""
    combined = []
    seen = set()

    for directory in directory_paths:
        directory_paths_found = get_image_paths(directory)
        added = 0
        for path in directory_paths_found:
            identity = path.resolve()
            if identity in seen:
                continue
            seen.add(identity)
            combined.append(path)
            added += 1

        print(
            f"  {group_name}: {directory} -> "
            f"{len(directory_paths_found):,} found, {added:,} added"
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


def _empty_class_counter():
    return {
        "correct": {0: 0, 1: 0},
        "total": {0: 0, 1: 0},
    }


def _update_class_counter(counter, logits, integer_labels):
    predictions = (torch.sigmoid(logits) >= 0.5).long().reshape(-1)
    for class_id in (0, 1):
        mask = integer_labels == class_id
        counter["total"][class_id] += mask.sum().item()
        counter["correct"][class_id] += (
            predictions[mask] == integer_labels[mask]
        ).sum().item()


def _finish_class_counter(counter):
    real_accuracy = counter["correct"][0] / max(counter["total"][0], 1)
    ai_accuracy = counter["correct"][1] / max(counter["total"][1], 1)
    return {
        "balanced_accuracy": 0.5 * (real_accuracy + ai_accuracy),
        "real_accuracy": real_accuracy,
        "ai_accuracy": ai_accuracy,
    }


def validate(model, data_loader, criterion, device):
    """Evaluate fused prediction and both diagnostic branches."""
    model.eval()
    loss_sum = 0.0
    counters = {
        "fused": _empty_class_counter(),
        "spatial": _empty_class_counter(),
        "frequency": _empty_class_counter(),
    }

    with torch.inference_mode():
        for spatial_images, frequency_images, labels in data_loader:
            spatial_images = spatial_images.to(device, non_blocking=True)
            frequency_images = frequency_images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float().unsqueeze(1)

            outputs = model(spatial_images, frequency_images)
            loss = criterion(outputs["fused_logits"], labels)
            loss_sum += loss.item() * spatial_images.size(0)

            integer_labels = labels.long().reshape(-1)
            _update_class_counter(
                counters["fused"],
                outputs["fused_logits"],
                integer_labels,
            )
            _update_class_counter(
                counters["spatial"],
                outputs["spatial_logits"],
                integer_labels,
            )
            _update_class_counter(
                counters["frequency"],
                outputs["frequency_logits"],
                integer_labels,
            )

    head_metrics = {
        name: _finish_class_counter(counter)
        for name, counter in counters.items()
    }
    return {
        "loss": loss_sum / len(data_loader.dataset),
        **head_metrics["fused"],
        "heads": head_metrics,
        "fusion_gate": torch.sigmoid(model.fusion_gate_logit).item(),
    }


def path_configuration():
    return {
        "train_real_dirs": [str(path) for path in TRAIN_REAL_DIRS],
        "train_ai_dirs": [str(path) for path in TRAIN_AI_DIRS],
        "validation_names": list(VALIDATION_NAMES),
        "validation_real_dirs": [str(path) for path in VAL_REAL_DIRS],
        "validation_ai_dirs": [str(path) for path in VAL_AI_DIRS],
    }


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
    save_path.parent.mkdir(parents=True, exist_ok=True)

    model_config = model.model_config()
    inference_checkpoint = {
        "checkpoint_type": "inference",
        "model_config": model_config,
        "model_state_dict": model.state_dict(),
    }
    torch.save(inference_checkpoint, save_path)

    if args.training_checkpoint_path:
        training_path = Path(args.training_checkpoint_path)
    else:
        training_path = save_path.with_name(
            f"{save_path.stem}_training{save_path.suffix}"
        )
    training_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "checkpoint_type": "training",
            "epoch": epoch,
            "model_config": model_config,
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

    metadata = {
        "positive_class": "ai_generated",
        "label_mapping": {"0": "real", "1": "ai_generated"},
        "score_meaning": "sigmoid(fused_logit) = P(ai_generated)",
        "decision_rule": "ai_generated if score >= 0.5",
        "preprocessing": {
            "resize": [IMAGE_SIZE, IMAGE_SIZE],
            "color_mode": "RGB",
            "spatial_normalization_mean": IMAGENET_MEAN,
            "spatial_normalization_std": IMAGENET_STD,
            "frequency_input_range": [0.0, 1.0],
            "frequency_window": "2-D Hann",
            "frequency_transform": "centered RGB log1p FFT power",
            "frequency_standardization": "per image and RGB channel",
        },
        "model": {
            "training_method": "supervised_spatial_full2d_spectral_fusion",
            "frequency_descriptor": "full_2d_rgb_log_power_spectrum",
            "frequency_encoder": "spectral_patch_transformer_cls_token",
            "radial_averaging": False,
            "tail_branch": False,
            "teacher_student_alignment": False,
            "fusion": "learned_gated_residual_on_spatial_logit",
            **model_config,
        },
        "training_data": path_configuration(),
        "selected_epoch": epoch,
        "validation_metrics": metrics,
        "inference_script": "detect.py",
    }
    with Path(f"{save_path}.meta.json").open(
        "w",
        encoding="utf-8",
    ) as metadata_file:
        json.dump(metadata, metadata_file, indent=2)

    print(f"Saved full-2D fusion inference checkpoint to {save_path}")
    print(f"Saved complete training checkpoint to {training_path}")


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
    heads = metrics["heads"]
    print(
        f"  {name} validation | loss={metrics['loss']:.4f}, "
        f"fused balanced acc={metrics['balanced_accuracy']:.4f}, "
        f"real acc={metrics['real_accuracy']:.4f}, "
        f"AI acc={metrics['ai_accuracy']:.4f}"
    )
    print(
        "    Diagnostic branch balanced accuracy | "
        f"spatial={heads['spatial']['balanced_accuracy']:.4f}, "
        f"frequency={heads['frequency']['balanced_accuracy']:.4f} | "
        f"fusion gate={metrics['fusion_gate']:.4f}"
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
        transform=FrequencyAwareAugmentation(size=IMAGE_SIZE),
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
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker,
    }
    train_loader = DataLoader(
        train_dataset,
        sampler=sampler,
        shuffle=sampler is None,
        generator=loader_generator,
        persistent_workers=args.num_workers > 0,
        **common_loader_options,
    )
    validation_loaders = {
        name: DataLoader(
            dataset,
            shuffle=False,
            persistent_workers=False,
            **common_loader_options,
        )
        for name, dataset in validation_datasets.items()
    }

    model = Full2DFrequencyFusionClassifier(
        pretrained=True,
        image_size=IMAGE_SIZE,
        patch_size=args.patch_size,
        frequency_embedding_dim=args.frequency_embedding_dim,
        frequency_depth=args.frequency_depth,
        frequency_num_heads=args.frequency_num_heads,
        frequency_mlp_ratio=args.frequency_mlp_ratio,
        fusion_hidden_dim=args.fusion_hidden_dim,
        dropout=args.dropout,
        initial_fusion_gate=args.initial_fusion_gate,
    ).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=positive_class_weight)

    backbone_parameters = list(
        model.spatial_detector.backbone.features.parameters()
    )
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
        loss_sums = {
            "total": 0.0,
            "fused": 0.0,
            "spatial_aux": 0.0,
            "frequency_aux": 0.0,
        }

        for spatial_images, frequency_images, labels in train_loader:
            spatial_images = spatial_images.to(device, non_blocking=True)
            frequency_images = frequency_images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float().unsqueeze(1)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                outputs = model(spatial_images, frequency_images)
                fused_loss = criterion(outputs["fused_logits"], labels)
                spatial_aux_loss = criterion(
                    outputs["spatial_logits"],
                    labels,
                )
                frequency_aux_loss = criterion(
                    outputs["frequency_logits"],
                    labels,
                )
                loss = (
                    fused_loss
                    + args.spatial_aux_loss_weight * spatial_aux_loss
                    + args.frequency_aux_loss_weight * frequency_aux_loss
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

            batch_size = spatial_images.size(0)
            loss_sums["total"] += loss.item() * batch_size
            loss_sums["fused"] += fused_loss.item() * batch_size
            loss_sums["spatial_aux"] += spatial_aux_loss.item() * batch_size
            loss_sums["frequency_aux"] += (
                frequency_aux_loss.item() * batch_size
            )

        scheduler.step()

        validation_metrics = {
            name: validate(model, loader, criterion, device)
            for name, loader in validation_loaders.items()
        }
        selection_score = float(
            np.mean(
                [
                    metrics["balanced_accuracy"]
                    for metrics in validation_metrics.values()
                ]
            )
        )
        checkpoint_metrics = {
            "selection_score": selection_score,
            "validation": validation_metrics,
        }

        denominator = len(train_loader.dataset)
        averaged_losses = {
            name: value / denominator
            for name, value in loss_sums.items()
        }

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"train total={averaged_losses['total']:.4f}, "
            f"fused={averaged_losses['fused']:.4f}, "
            f"spatial aux={averaged_losses['spatial_aux']:.4f}, "
            f"frequency aux={averaged_losses['frequency_aux']:.4f} | "
            f"selection score={selection_score:.4f}"
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
        description=(
            "Train an AI-image detector using ConvNeXt spatial features and "
            "the complete 2-D RGB FFT log-power spectrum"
        )
    )
    # Dataset paths intentionally are not command-line arguments. Edit the
    # directory lists at the top of this file.
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--backbone_lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--frequency_embedding_dim", type=int, default=256)
    parser.add_argument("--frequency_depth", type=int, default=4)
    parser.add_argument("--frequency_num_heads", type=int, default=8)
    parser.add_argument("--frequency_mlp_ratio", type=float, default=4.0)
    parser.add_argument("--fusion_hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--initial_fusion_gate", type=float, default=0.1)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument(
        "--spatial_aux_loss_weight",
        type=float,
        default=0.1,
        help="Auxiliary supervision for the spatial-only diagnostic head",
    )
    parser.add_argument(
        "--frequency_aux_loss_weight",
        type=float,
        default=0.1,
        help="Auxiliary supervision for the frequency-only diagnostic head",
    )
    parser.add_argument(
        "--balance_strategy",
        choices=("auto", "none", "loss", "sampler"),
        default="auto",
        help=(
            "auto uses shuffling when class counts are nearly balanced and "
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
        default="aigid.pth",
    )
    parser.add_argument(
        "--training_checkpoint_path",
        default=None,
        help="Optional path for the complete training checkpoint",
    )
    return parser


def validate_arguments(parser, args):
    if args.batch_size < 1:
        parser.error("--batch_size must be positive")
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.num_workers < 0:
        parser.error("--num_workers cannot be negative")
    if IMAGE_SIZE % args.patch_size != 0:
        parser.error(f"--patch_size must divide IMAGE_SIZE={IMAGE_SIZE}")
    if args.frequency_embedding_dim % args.frequency_num_heads != 0:
        parser.error(
            "--frequency_embedding_dim must be divisible by "
            "--frequency_num_heads"
        )
    if args.frequency_depth < 1:
        parser.error("--frequency_depth must be positive")
    if args.frequency_mlp_ratio <= 0:
        parser.error("--frequency_mlp_ratio must be positive")
    if not 0.0 <= args.dropout < 1.0:
        parser.error("--dropout must lie in [0, 1)")
    if not 0.0 < args.initial_fusion_gate < 1.0:
        parser.error("--initial_fusion_gate must lie in (0, 1)")
    if args.grad_clip_norm <= 0:
        parser.error("--grad_clip_norm must be positive")
    if not 0.0 < args.balance_tolerance <= 1.0:
        parser.error("--balance_tolerance must lie in (0, 1]")
    if args.spatial_aux_loss_weight < 0:
        parser.error("--spatial_aux_loss_weight must be nonnegative")
    if args.frequency_aux_loss_weight < 0:
        parser.error("--frequency_aux_loss_weight must be nonnegative")


if __name__ == "__main__":
    argument_parser = build_argument_parser()
    parsed_args = argument_parser.parse_args()
    validate_arguments(argument_parser, parsed_args)
    train(parsed_args)
