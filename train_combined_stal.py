import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from data_augmentation import STALAugmentation
from model import STALTrainingModel, class_balanced_alignment_loss


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class AIGCDataset(Dataset):
    """Return an augmented spatial view and a clean aligned frequency view."""

    def __init__(self, real_image_paths, fake_image_paths, transform=None):
        self.paths = [(path, 0) for path in real_image_paths] + [
            (path, 1) for path in fake_image_paths
        ]
        self.transform = transform
        self.eval_geometry = transforms.Resize((224, 224), antialias=True)
        self.spatial_to_tensor = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
        self.frequency_to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path, label = self.paths[index]
        with Image.open(path) as source:
            source.load()
        
            has_transparency = (
                source.mode in {"RGBA", "LA"}
                or (
                    source.mode == "P"
                    and "transparency" in source.info
                )
            )
        
            if has_transparency:
                rgba = source.convert("RGBA")
        
                # Composite transparent pixels onto white instead of silently
                # discarding the alpha channel.
                background = Image.new(
                    "RGBA",
                    rgba.size,
                    (255, 255, 255, 255),
                )
                image = Image.alpha_composite(
                    background,
                    rgba,
                ).convert("RGB")
            else:
                image = source.convert("RGB")

        if self.transform is not None:
            spatial_view, frequency_view = self.transform(image)
        else:
            # Keep validation preprocessing identical to deployment.
            shared_view = self.eval_geometry(image)
            spatial_view = shared_view
            frequency_view = shared_view

        spatial_tensor = self.spatial_to_tensor(spatial_view)
        frequency_tensor = self.frequency_to_tensor(frequency_view)
        return spatial_tensor, frequency_tensor, label


def get_image_paths(folder_path):
    """Return a deterministic recursive list of image paths."""
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Image directory does not exist: {folder}")
    paths = sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        raise ValueError(f"No supported images found under: {folder}")
    return paths


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


def stal_curriculum_weight(step, total_steps):
    """Warm up, hold, then remove auxiliary frequency supervision.

    The frequency teacher shapes the representation early in training without
    becoming a permanent crutch for the spatial detector.
    """
    progress = step / max(total_steps - 1, 1)
    warmup_end = 0.05
    decay_start = 0.15
    decay_end = 0.45

    if progress < warmup_end:
        return progress / warmup_end
    if progress < decay_start:
        return 1.0
    if progress < decay_end:
        phase = (progress - decay_start) / (decay_end - decay_start)
        return 0.5 * (1.0 + math.cos(math.pi * phase))
    return 0.0


def validate(model, data_loader, criterion, device):
    model.eval()
    loss_sum = 0.0
    class_correct = {0: 0, 1: 0}
    class_total = {0: 0, 1: 0}

    with torch.inference_mode():
        for spatial_images, _, labels in data_loader:
            spatial_images = spatial_images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float().unsqueeze(1)
            logits = model.spatial_detector(spatial_images)
            loss = criterion(logits, labels)
            loss_sum += loss.item() * spatial_images.size(0)

            predictions = (torch.sigmoid(logits) >= 0.5).long().reshape(-1)
            integer_labels = labels.long().reshape(-1)
            for class_id in (0, 1):
                mask = integer_labels == class_id
                class_total[class_id] += mask.sum().item()
                class_correct[class_id] += (
                    predictions[mask] == integer_labels[mask]
                ).sum().item()

    class_accuracy = {
        class_id: class_correct[class_id] / max(class_total[class_id], 1)
        for class_id in (0, 1)
    }
    balanced_accuracy = 0.5 * (class_accuracy[0] + class_accuracy[1])
    return {
        "loss": loss_sum / len(data_loader.dataset),
        "balanced_accuracy": balanced_accuracy,
        "real_accuracy": class_accuracy[0],
        "fake_accuracy": class_accuracy[1],
    }


def save_checkpoints(model, optimizer, epoch, metrics, args):
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # This file contains only the deployment model and remains compatible with
    # detect_aigc.py and with the original AIGCClassifier class.
    torch.save(model.spatial_state_dict(), save_path)

    if args.stal_checkpoint_path:
        training_path = Path(args.stal_checkpoint_path)
    else:
        training_path = save_path.with_name(f"{save_path.stem}_stal_training{save_path.suffix}")
    training_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "validation_metrics": metrics,
            "arguments": vars(args),
        },
        training_path,
    )

    metadata = {
        "positive_class": "ai_generated",
        "label_mapping": {"0": "real", "1": "ai_generated"},
        "score_meaning": "sigmoid(logit) = P(ai_generated)",
        "decision_rule": "ai_generated if score >= 0.5",
        "preprocessing": {
            "resize": [224, 224],
            "color_mode": "RGB",
            "normalization_mean": IMAGENET_MEAN,
            "normalization_std": IMAGENET_STD,
        },
        "training_method": "lightweight_STAL_style",
        "training_data": {
            "train_real_dir": args.train_real_dir,
            "train_fake_dir": args.train_fake_dir,
            "community_train_real_dir": args.community_train_real_dir,
            "community_train_fake_dir": args.community_train_fake_dir,
            "val_real_dir": args.val_real_dir,
            "val_fake_dir": args.val_fake_dir,
            "community_val_real_dir": args.community_val_real_dir,
            "community_val_fake_dir": args.community_val_fake_dir,
        },
        "selected_epoch": epoch,
        "validation_metrics": metrics,
        "note": "The full STAL research implementation also uses local DCT and supervised contrastive losses.",
    }
    with open(f"{save_path}.meta.json", "w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)

    print(f"Saved inference weights to {save_path}")
    print(f"Saved full STAL training checkpoint to {training_path}")


def train(args):
    seed_everything(args.seed)
    device = torch.device(args.device)

    if bool(args.val_real_dir) != bool(args.val_fake_dir):
        raise ValueError("Provide both SID validation directories")
    if bool(args.community_val_real_dir) != bool(args.community_val_fake_dir):
        raise ValueError("Provide both CommunityForensics validation directories")

    # Training data are combined. Validation data remain in separate loaders so
    # a strong SID score cannot conceal weak cross-dataset generalization.
    train_real_paths = (
        get_image_paths(args.train_real_dir)
        + get_image_paths(args.community_train_real_dir)
    )
    train_fake_paths = (
        get_image_paths(args.train_fake_dir)
        + get_image_paths(args.community_train_fake_dir)
    )
    sid_val_real_paths = get_image_paths(args.val_real_dir)
    sid_val_fake_paths = get_image_paths(args.val_fake_dir)
    community_val_real_paths = get_image_paths(args.community_val_real_dir)
    community_val_fake_paths = get_image_paths(args.community_val_fake_dir)

    print(
        "Dataset sizes | "
        f"train real={len(train_real_paths)}, train fake={len(train_fake_paths)}, "
        f"SID val real={len(sid_val_real_paths)}, "
        f"SID val fake={len(sid_val_fake_paths)}, "
        f"CompEval val real={len(community_val_real_paths)}, "
        f"CompEval val fake={len(community_val_fake_paths)}"
    )

    train_dataset = AIGCDataset(
        train_real_paths,
        train_fake_paths,
        transform=STALAugmentation(size=224),
    )
    sid_val_dataset = AIGCDataset(
        sid_val_real_paths,
        sid_val_fake_paths,
        transform=None,
    )
    community_val_dataset = AIGCDataset(
        community_val_real_paths,
        community_val_fake_paths,
        transform=None,
    )

    labels = np.asarray([label for _, label in train_dataset.paths])
    class_counts = np.bincount(labels, minlength=2)
    if np.any(class_counts == 0):
        raise ValueError(f"Both classes are required, found counts: {class_counts.tolist()}")

    balance_strategy = args.balance_strategy
    if balance_strategy == "auto":
        balance_strategy = "none" if class_counts[0] == class_counts[1] else "loss"

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

    print(
        f"Class counts: real={class_counts[0]}, fake={class_counts[1]} | "
        f"balancing={balance_strategy}"
    )

    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)
    train_loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker,
        "generator": loader_generator,
        "persistent_workers": args.num_workers > 0,
    }
    evaluation_loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker,
        # Do not retain two extra sets of validation workers in memory.
        "persistent_workers": False,
    }
    train_loader = DataLoader(
        train_dataset,
        sampler=sampler,
        shuffle=sampler is None,
        **train_loader_options,
    )
    sid_val_loader = DataLoader(
        sid_val_dataset,
        shuffle=False,
        **evaluation_loader_options,
    )
    community_val_loader = DataLoader(
        community_val_dataset,
        shuffle=False,
        **evaluation_loader_options,
    )

    model = STALTrainingModel(pretrained=True).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=positive_class_weight)

    backbone_parameters = list(model.spatial_detector.backbone.features.parameters())
    backbone_parameter_ids = {id(parameter) for parameter in backbone_parameters}
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
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    total_steps = args.epochs * len(train_loader)
    global_step = 0
    best_selection_score = -1.0

    for epoch in range(args.epochs):
        model.train()
        loss_sums = {
            "total": 0.0,
            "spatial": 0.0,
            "frequency": 0.0,
            "tail": 0.0,
            "alignment": 0.0,
        }

        for spatial_images, frequency_images, labels in train_loader:
            spatial_images = spatial_images.to(device, non_blocking=True)
            frequency_images = frequency_images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float().unsqueeze(1)
            auxiliary_weight = stal_curriculum_weight(global_step, total_steps)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                outputs = model(spatial_images, frequency_images)
                spatial_loss = criterion(outputs["spatial_logits"], labels)
                frequency_loss = criterion(outputs["frequency_logits"], labels)
                tail_loss = criterion(outputs["tail_logits"], labels)
                alignment_loss = class_balanced_alignment_loss(
                    outputs["spatial_projection"],
                    outputs["teacher_target"],
                    labels,
                )
                auxiliary_loss = (
                    args.frequency_loss_weight * frequency_loss
                    + args.tail_loss_weight * tail_loss
                    + args.alignment_loss_weight * alignment_loss
                )
                loss = spatial_loss + auxiliary_weight * auxiliary_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_size = spatial_images.size(0)
            loss_sums["total"] += loss.item() * batch_size
            loss_sums["spatial"] += spatial_loss.item() * batch_size
            loss_sums["frequency"] += frequency_loss.item() * batch_size
            loss_sums["tail"] += tail_loss.item() * batch_size
            loss_sums["alignment"] += alignment_loss.item() * batch_size
            global_step += 1

        scheduler.step()
        sid_metrics = validate(model, sid_val_loader, criterion, device)
        community_metrics = validate(
            model,
            community_val_loader,
            criterion,
            device,
        )
        selection_score = 0.5 * (
            sid_metrics["balanced_accuracy"]
            + community_metrics["balanced_accuracy"]
        )
        checkpoint_metrics = {
            "selection_score": selection_score,
            "sid_validation": sid_metrics,
            "community_validation": community_metrics,
        }
        denominator = len(train_loader.dataset)
        averaged_losses = {
            name: value / denominator for name, value in loss_sums.items()
        }

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"train total={averaged_losses['total']:.4f}, "
            f"spatial={averaged_losses['spatial']:.4f}, "
            f"freq={averaged_losses['frequency']:.4f}, "
            f"tail={averaged_losses['tail']:.4f}, "
            f"align={averaged_losses['alignment']:.4f} | "
            f"selection score={selection_score:.4f}"
        )
        print(
            "  SID validation | "
            f"loss={sid_metrics['loss']:.4f}, "
            f"balanced acc={sid_metrics['balanced_accuracy']:.4f}, "
            f"real acc={sid_metrics['real_accuracy']:.4f}, "
            f"fake acc={sid_metrics['fake_accuracy']:.4f}"
        )
        print(
            "  CompEval validation | "
            f"loss={community_metrics['loss']:.4f}, "
            f"balanced acc={community_metrics['balanced_accuracy']:.4f}, "
            f"real acc={community_metrics['real_accuracy']:.4f}, "
            f"fake acc={community_metrics['fake_accuracy']:.4f}"
        )

        if selection_score > best_selection_score:
            best_selection_score = selection_score
            save_checkpoints(
                model,
                optimizer,
                epoch + 1,
                checkpoint_metrics,
                args,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a lightweight STAL-style AI-generated image detector"
    )
    parser.add_argument(
        "--train_real_dir",
        default="/home/utakata/LLMs/techjam2026/SID/real",
    )
    parser.add_argument(
        "--train_fake_dir",
        default="/home/utakata/LLMs/techjam2026/SID/ai",
    )
    parser.add_argument(
        "--community_train_real_dir",
        default="data/com/real",
        help="CommunityForensics-Small real-image training directory",
    )
    parser.add_argument(
        "--community_train_fake_dir",
        default="data/com/ai",
        help="CommunityForensics-Small AI-image training directory",
    )
    parser.add_argument(
        "--val_real_dir",
        default="/home/utakata/LLMs/techjam2026/SID/test/real",
        help="SID validation real-image directory",
    )
    parser.add_argument(
        "--val_fake_dir",
        default="/home/utakata/LLMs/techjam2026/SID/test/full_synthetic",
        help="SID validation AI-image directory",
    )
    parser.add_argument(
        "--community_val_real_dir",
        default="data/com/val/real",
        help="CommunityForensics CompEval real-image validation directory",
    )
    parser.add_argument(
        "--community_val_fake_dir",
        default="data/com/val/ai",
        help="CommunityForensics CompEval AI-image validation directory",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--backbone_lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--frequency_loss_weight", type=float, default=0.2)
    parser.add_argument("--tail_loss_weight", type=float, default=0.2)
    parser.add_argument("--alignment_loss_weight", type=float, default=0.1)
    parser.add_argument(
        "--balance_strategy",
        choices=("auto", "none", "loss", "sampler"),
        default="auto",
        help=(
            "Class-imbalance handling. auto uses ordinary shuffling when "
            "balanced and BCE positive-class weighting when unbalanced; "
            "sampler oversamples the minority class with replacement."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--save_path", default="aigc_detector.pth")
    parser.add_argument(
        "--stal_checkpoint_path",
        default=None,
        help="Optional path for the resumable training checkpoint",
    )
    train(parser.parse_args())
