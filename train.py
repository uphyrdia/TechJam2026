"""Train the spatial/frequency fusion classifier and save its best weights.

Each image is presented to the model in two forms: ImageNet-normalized RGB for
the spatial branch and unnormalized ``[0, 1]`` RGB for the radial-spectrum
branch. Validation is reported for every configured dataset, while only the
datasets named in ``SELECTION_VALIDATION_SETS`` participate in best-epoch
selection. Labels use ``0`` for real images and ``1`` for AI-generated images.

Dataset locations and the main defaults are intentionally kept as module-level
configuration so a run's configuration remains visible without a separate
config file. Command-line arguments can override numerical and output settings.
"""

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

from data_augmentation import Augmentation
from model import Fusion

# Training directories are combined within each class. ``get_image_paths``
# scans them recursively and removes duplicate physical paths.
TRAIN_REAL_DIRS = [
    "/home/utakata/LLMs/techjam2026/SID/real",
    "/home/utakata/LLMs/techjam/data/com/real",
]
TRAIN_FAKE_DIRS = [
    "/home/utakata/LLMs/techjam2026/SID/ai",
    "/home/utakata/LLMs/techjam/data/com/ai",
]

# These three validation lists are parallel: entries at the same index form
# one named real/fake validation dataset.
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
# Only these named validation sets determine the best epoch. For example, use
# ["SID"] for in-distribution selection, or ["SID", "CompEval"] to average
# the selected metric across both. Every listed dataset is still evaluated.
SELECTION_VALIDATION_SETS = ["SID", "CompEval"]
# Balanced accuracy uses the fixed 0.5 decision threshold; ROC AUC measures
# ranking quality across all thresholds and is therefore threshold-independent.
SELECTION_METRIC = "roc_auc"  # "balanced_accuracy" or "roc_auc"

# Inference weights and their JSON metadata are always saved on improvement.
# The larger training-state checkpoint is optional because it also contains
# optimizer, scheduler, and gradient-scaler state. This script does not include
# resume-loading logic or save RNG/sampler state for an exact continuation.
SAVE_TRAINING_CHECKPOINT = False
WEIGHTS_PATH = "weights.pth"
TRAINING_CHECKPOINT_PATH = "training_checkpoint.pth"

IMAGE_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 8
NUM_WORKERS = 12
LEARNING_RATE = 1e-4
BACKBONE_LEARNING_RATE = 2e-5
WEIGHT_DECAY = 1e-4
USE_AMP = True

# =============================================================================

# Shared preprocessing constants must match the inference script.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class AIGCDataset(Dataset):
    """Pair image paths with labels and construct synchronized model inputs.

    A training transform is applied once to the PIL image, before either tensor
    is created. Consequently, both branches see the same random crop, flip,
    compression artifact, or other augmentation rather than independently
    transformed versions of the source image. With no transform, validation
    uses only a deterministic square resize.
    """

    def __init__(
        self,
        real_image_paths,
        fake_image_paths,
        image_size,
        transform=None,
    ):
        """Create a real-label-0/fake-label-1 dataset from two path lists."""
        self.paths = [(path, 0) for path in real_image_paths] + [
            (path, 1) for path in fake_image_paths
        ]
        self.transform = transform
        self.eval_geometry = transforms.Resize(
            (image_size, image_size),
            antialias=True,
        )
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        )

    def __len__(self):
        """Return the total number of real and AI-generated examples."""
        return len(self.paths)

    @staticmethod
    def _load_rgb_image(path):
        """Load an image as RGB, compositing transparent pixels over white."""
        with Image.open(path) as source:
            # Force decoding while the file handle is open. Converted images
            # returned below are then independent of the context manager.
            source.load()
            has_transparency = (
                source.mode in {"RGBA", "LA"}
                or (source.mode == "P" and "transparency" in source.info)
            )
            if has_transparency:
                # Direct RGBA-to-RGB conversion discards alpha and can leave an
                # unintended dark background. White is a neutral canvas for the
                # photographs and generated images used by this detector.
                rgba = source.convert("RGBA")
                background = Image.new(
                    "RGBA",
                    rgba.size,
                    (255, 255, 255, 255),
                )
                return Image.alpha_composite(background, rgba).convert("RGB")
            return source.convert("RGB")

    def __getitem__(self, index):
        """Return ``(spatial_tensor, frequency_tensor, integer_label)``."""
        path, label = self.paths[index]
        image = self._load_rgb_image(path)
        view = (
            self.transform(image)
            if self.transform is not None
            else self.eval_geometry(image)
        )

        # The FFT branch sees [0, 1] RGB; ConvNeXt sees ImageNet-normalized RGB.
        # Both tensors are derived from the exact same augmented view.
        frequency_tensor = self.to_tensor(view)
        spatial_tensor = self.normalize(frequency_tensor)
        return spatial_tensor, frequency_tensor, label


def get_image_paths(directory_list):
    """Collect a deterministic, de-duplicated recursive image-path list.

    Sorting makes the base ordering reproducible. Resolving each path before
    de-duplication also prevents the same file from entering twice through a
    symlink or overlapping directory configuration.
    """
    if not directory_list:
        raise ValueError("At least one image directory is required")

    paths = []
    seen = set()
    for directory in directory_list:
        folder = Path(directory).expanduser()
        if not folder.is_dir():
            raise FileNotFoundError(f"Image directory does not exist: {folder}")

        folder_paths = sorted(
            path
            for path in folder.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not folder_paths:
            raise ValueError(f"No supported images found under: {folder}")

        for path in folder_paths:
            identity = path.resolve()
            if identity not in seen:
                seen.add(identity)
                paths.append(path)

    return paths


def _ensure_disjoint(first_paths, second_paths, description):
    """Reject overlapping path collections to prevent leakage or label clash."""
    first = {path.resolve() for path in first_paths}
    overlap = first.intersection(path.resolve() for path in second_paths)
    if overlap:
        examples = sorted(str(path) for path in overlap)[:5]
        raise ValueError(
            f"Data leakage: {description} share {len(overlap)} file(s). "
            f"Examples: {examples}"
        )


def seed_everything(seed):
    """Seed main-process RNGs to improve repeatability between training runs.

    This does not force deterministic GPU kernels, so identical seeds do not by
    themselves guarantee bit-for-bit identical results on every platform.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    """Seed Python and NumPy from PyTorch's unique seed for this loader worker."""
    # PyTorch has already incorporated the worker id into ``initial_seed``;
    # the argument itself is therefore deliberately unused. The modulo maps
    # the 64-bit PyTorch seed into NumPy's supported 32-bit seed range.
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _calculate_metrics(labels, probabilities, threshold=0.5):
    """Compute fixed-threshold class accuracy, balanced accuracy, and ROC AUC.

    ``probabilities`` contains sigmoid scores for label 1 (AI-generated); class
    balancing means they need not be calibrated posterior probabilities. The
    threshold is not optimized on validation data: the default 0.5 rule is used
    for real, fake, and balanced accuracy. ROC AUC instead evaluates score
    ordering and does not depend on that threshold.
    """
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = (probabilities >= threshold).astype(np.int64)

    real_mask = labels == 0
    fake_mask = labels == 1
    if not real_mask.any() or not fake_mask.any():
        raise ValueError("Validation requires at least one real and one fake image")

    real_accuracy = float((predictions[real_mask] == 0).mean())
    fake_accuracy = float((predictions[fake_mask] == 1).mean())

    # Mann-Whitney formulation of ROC AUC. Average ranks make this correct when
    # several images receive exactly the same score, without requiring
    # scikit-learn as an additional training dependency.
    order = np.argsort(probabilities, kind="mergesort")
    sorted_probabilities = probabilities[order]
    sorted_ranks = np.empty(len(probabilities), dtype=np.float64)
    start = 0
    while start < len(probabilities):
        end = start + 1
        while (
            end < len(probabilities)
            and sorted_probabilities[end] == sorted_probabilities[start]
        ):
            end += 1
        # Ranks are one-based; all tied observations receive their mean rank.
        sorted_ranks[start:end] = 0.5 * ((start + 1) + end)
        start = end
    ranks = np.empty_like(sorted_ranks)
    ranks[order] = sorted_ranks
    positive_count = int(fake_mask.sum())
    negative_count = int(real_mask.sum())
    positive_rank_sum = ranks[fake_mask].sum()
    roc_auc = (
        positive_rank_sum
        - positive_count * (positive_count + 1) / 2
    ) / (positive_count * negative_count)

    return {
        "balanced_accuracy": 0.5 * (real_accuracy + fake_accuracy),
        "roc_auc": float(roc_auc),
        "real_accuracy": real_accuracy,
        "fake_accuracy": fake_accuracy,
    }


def validate(model, data_loader, criterion, device):
    """Evaluate fused prediction and both diagnostic branch predictions.

    The validation loss is the fused-head BCE used for the primary task. The
    spatial and frequency heads are still scored separately so developers can
    see whether one branch has collapsed or dominates the fusion.
    """
    # ``eval`` disables this model's dropout and stochastic-depth behavior; it
    # does not alter or reset any learned parameters.
    model.eval()
    loss_sum = 0.0
    labels_all = []
    score_lists = {
        "fused": [],
        "spatial": [],
        "frequency": [],
    }

    # Inference mode avoids autograd bookkeeping and is stricter/lighter than
    # merely disabling gradient calculation.
    with torch.inference_mode():
        for spatial_images, frequency_images, labels in data_loader:
            spatial_images = spatial_images.to(device, non_blocking=True)
            frequency_images = frequency_images.to(device, non_blocking=True)
            target = labels.to(device, non_blocking=True).float().unsqueeze(1)

            outputs = model(spatial_images, frequency_images)
            loss = criterion(outputs["fused_logits"], target)
            # Accumulate per-example loss so a short final batch is weighted
            # correctly when the epoch average is formed.
            loss_sum += loss.item() * spatial_images.size(0)

            labels_all.extend(labels.numpy().tolist())
            for head in score_lists:
                logits = outputs[f"{head}_logits"]
                score_lists[head].extend(
                    torch.sigmoid(logits).reshape(-1).cpu().tolist()
                )

    # Omitting ``threshold`` intentionally applies the documented 0.5 rule to
    # every head; no per-dataset threshold tuning occurs here.
    head_metrics = {
        head: _calculate_metrics(labels_all, scores)
        for head, scores in score_lists.items()
    }
    return {
        "loss": loss_sum / len(data_loader.dataset),
        **head_metrics["fused"],
        "heads": head_metrics,
    }


def _build_balancing(args, class_counts, labels, device):
    """Choose and construct the requested class-imbalance correction.

    The ``loss`` strategy rescales the positive (AI-generated) contribution to
    BCE, whereas ``sampler`` draws examples with inverse-class-frequency
    weights. They are mutually exclusive so imbalance is not corrected twice.
    ``auto`` leaves sufficiently balanced data untouched and otherwise chooses
    loss weighting.
    """
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
        # Sampling with replacement preserves the nominal epoch length while
        # making the two classes equally likely in expectation across draws;
        # individual batches need not be balanced, and examples may repeat or
        # be omitted. The dedicated generator advances to fresh draws whenever
        # the DataLoader starts a new epoch.
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
        # BCEWithLogitsLoss treats label 1 as positive. N_real / N_fake makes
        # the total real and fake loss contributions comparable.
        positive_class_weight = torch.tensor(
            [class_counts[0] / class_counts[1]],
            dtype=torch.float32,
            device=device,
        )

    return balance_strategy, sampler, positive_class_weight, imbalance_ratio


def _print_validation(name, metrics):
    """Print primary fused metrics followed by branch-level diagnostics."""
    heads = metrics["heads"]
    print(
        f"  {name} | loss={metrics['loss']:.4f}, "
        f"fused balanced_accuracy={metrics['balanced_accuracy']:.4f}, "
        f"roc_auc={metrics['roc_auc']:.4f}, "
        f"real_accuracy={metrics['real_accuracy']:.4f}, "
        f"fake_accuracy={metrics['fake_accuracy']:.4f}"
    )
    print(
        "    Diagnostic heads | "
        f"spatial bal_acc={heads['spatial']['balanced_accuracy']:.4f}, "
        f"AUC={heads['spatial']['roc_auc']:.4f}; "
        f"frequency bal_acc={heads['frequency']['balanced_accuracy']:.4f}, "
        f"AUC={heads['frequency']['roc_auc']:.4f}"
    )


def _save_best_artifacts(
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    checkpoint_metrics,
    args,
):
    """Save deployable weights/metadata and, if enabled, training state.

    This function is called only after the configured selection score strictly
    improves. The compact state dict is sufficient for ``detect.py``. The
    optional training checkpoint additionally captures optimizer, scheduler,
    and AMP-scaler state for a separately implemented resume workflow; this
    script does not load that checkpoint or restore RNG/sampler state.
    """
    # Record directory macros as well as the selection rule so the provenance
    # of the chosen weights remains inspectable after training.
    configuration = {
        "train_real_dirs": TRAIN_REAL_DIRS,
        "train_fake_dirs": TRAIN_FAKE_DIRS,
        "validation_names": list(VALIDATION_NAMES),
        "validation_real_dirs": list(VAL_REAL_DIRS),
        "validation_ai_dirs": list(VAL_AI_DIRS),
        "selection_validation_sets": SELECTION_VALIDATION_SETS,
        "selection_metric": args.selection_metric,
    }

    weights_path = Path(args.weights_path).expanduser()
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    # Save a plain model state dict to keep the inference-loading API simple.
    torch.save(model.state_dict(), weights_path)

    # Metadata is deliberately stored beside the weights rather than embedded
    # in them, allowing humans and deployment code to inspect it without
    # deserializing a PyTorch object.
    metadata = {
        "positive_class": "ai_generated",
        "label_mapping": {"0": "real", "1": "ai_generated"},
        "score_meaning": "sigmoid(fused_logit) = P(ai_generated)",
        "decision_rule": "ai_generated if score >= 0.5",
        "preprocessing": {
            "resize": [args.image_size, args.image_size],
            "color_mode": "RGB",
            "spatial_normalization_mean": IMAGENET_MEAN,
            "spatial_normalization_std": IMAGENET_STD,
            "frequency_input_range": [0.0, 1.0],
        },
        "model": {
            "training_method": "direct_spatial_frequency_fusion",
            "frequency_descriptor": "full_radial_log_power_spectrum",
            "image_size": args.image_size,
            "num_frequency_bins": args.num_frequency_bins,
            "frequency_embedding_dim": args.frequency_embedding_dim,
            "fusion_hidden_dim": args.fusion_hidden_dim,
            "tail_branch": False,
            "teacher_student_alignment": False,
        },
        "data_configuration": configuration,
        "selected_epoch": epoch,
        "validation_metrics": checkpoint_metrics,
        "inference_script": "detect.py",
    }
    metadata_path = Path(f"{weights_path}.meta.json")
    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)
    print(f"Saved best inference weights to {weights_path}")

    if args.save_training_checkpoint:
        checkpoint_path = Path(args.training_checkpoint_path).expanduser()
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "validation_metrics": checkpoint_metrics,
                "configuration": configuration,
                "arguments": vars(args),
            },
            checkpoint_path,
        )
        print(f"Saved full training checkpoint to {checkpoint_path}")


def _validate_configuration(args):
    """Fail early on inconsistent macros or invalid selection/loss settings."""
    if args.selection_metric not in {"balanced_accuracy", "roc_auc"}:
        raise ValueError(
            "selection_metric must be 'balanced_accuracy' or 'roc_auc'"
        )
    if not (
        len(VALIDATION_NAMES) == len(VAL_REAL_DIRS) == len(VAL_AI_DIRS)
    ):
        raise ValueError(
            "VALIDATION_NAMES, VAL_REAL_DIRS, and VAL_AI_DIRS must have "
            "the same length"
        )
    if not VALIDATION_NAMES:
        raise ValueError("At least one validation dataset is required")
    if any(not name for name in VALIDATION_NAMES) or len(
        VALIDATION_NAMES
    ) != len(set(VALIDATION_NAMES)):
        raise ValueError("Every validation set needs a unique, non-empty name")
    if not SELECTION_VALIDATION_SETS:
        raise ValueError("At least one selection validation set is required")
    unknown = set(SELECTION_VALIDATION_SETS).difference(VALIDATION_NAMES)
    if unknown:
        raise ValueError(
            f"Unknown selection validation set(s): {sorted(unknown)}; "
            f"available names: {VALIDATION_NAMES}"
        )
    if len(SELECTION_VALIDATION_SETS) != len(
        set(SELECTION_VALIDATION_SETS)
    ):
        raise ValueError("SELECTION_VALIDATION_SETS must not contain duplicates")
    if not 0 < args.balance_tolerance <= 1:
        raise ValueError("balance_tolerance must lie in (0, 1]")
    if args.spatial_aux_loss_weight < 0:
        raise ValueError("spatial_aux_loss_weight must be nonnegative")
    if args.frequency_aux_loss_weight < 0:
        raise ValueError("frequency_aux_loss_weight must be nonnegative")
    if args.save_training_checkpoint:
        if Path(args.weights_path).expanduser().resolve() == Path(
            args.training_checkpoint_path
        ).expanduser().resolve():
            raise ValueError(
                "weights_path and training_checkpoint_path must differ when "
                "training-checkpoint saving is enabled"
            )


def train(args):
    """Build the data/model pipeline, train it, and retain the best epoch."""
    # Validate configuration before any expensive directory scans or model
    # construction, then seed the main process before creating random objects.
    _validate_configuration(args)
    seed_everything(args.seed)
    device = torch.device(args.device)

    # Resolve complete file lists up front so class collisions and train/val
    # leakage cause an explicit error rather than an optimistic validation score.
    train_real_paths = get_image_paths(TRAIN_REAL_DIRS)
    train_fake_paths = get_image_paths(TRAIN_FAKE_DIRS)
    _ensure_disjoint(
        train_real_paths,
        train_fake_paths,
        "training real and fake classes",
    )

    validation_data = {}
    # zip is safe here because _validate_configuration has already required all
    # three parallel validation lists to have identical lengths.
    for name, real_directory, ai_directory in zip(
        VALIDATION_NAMES,
        VAL_REAL_DIRS,
        VAL_AI_DIRS,
    ):
        real_paths = get_image_paths([real_directory])
        fake_paths = get_image_paths([ai_directory])
        _ensure_disjoint(
            real_paths,
            fake_paths,
            f"validation set '{name}' real and AI classes",
        )
        _ensure_disjoint(
            train_real_paths + train_fake_paths,
            real_paths + fake_paths,
            f"training data and validation set '{name}'",
        )
        validation_data[name] = (real_paths, fake_paths)

    print(
        "Training sizes | "
        f"real={len(train_real_paths)}, fake={len(train_fake_paths)}"
    )
    for name, (real_paths, fake_paths) in validation_data.items():
        role = "selection" if name in SELECTION_VALIDATION_SETS else "monitor only"
        print(
            f"Validation '{name}' ({role}) | "
            f"real={len(real_paths)}, fake={len(fake_paths)}"
        )
    print(
        f"Best-epoch rule | metric={args.selection_metric}, "
        f"sets={SELECTION_VALIDATION_SETS}"
    )

    train_transform = Augmentation(size=args.image_size)
    train_dataset = AIGCDataset(
        train_real_paths,
        train_fake_paths,
        image_size=args.image_size,
        transform=train_transform,
    )
    # Validation omits stochastic augmentation and uses AIGCDataset's fixed
    # resize, keeping epoch-to-epoch metrics directly comparable.
    validation_datasets = {
        name: AIGCDataset(
            real_paths,
            fake_paths,
            image_size=args.image_size,
            transform=None,
        )
        for name, (real_paths, fake_paths) in validation_data.items()
    }

    # Counts come from the final de-duplicated file list, not from directory
    # names or assumptions about the source datasets.
    labels = np.asarray([label for _, label in train_dataset.paths])
    class_counts = np.bincount(labels, minlength=2)
    if np.any(class_counts == 0):
        raise ValueError(
            f"Both classes are required, found counts: {class_counts.tolist()}"
        )

    balance_strategy, sampler, pos_weight, imbalance_ratio = _build_balancing(
        args,
        class_counts,
        labels,
        device,
    )
    print(
        f"Class balance | minority/majority={imbalance_ratio:.4f}, "
        f"strategy={balance_strategy}"
    )

    # This generator controls ordinary shuffle order and PyTorch's base worker
    # seeds. The weighted sampler has its own seeded generator above.
    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)
    common_loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        # Pinned host memory lets the non_blocking CUDA transfers below overlap
        # with computation. It provides no benefit for CPU-only training.
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker,
    }
    train_loader = DataLoader(
        train_dataset,
        sampler=sampler,
        # A DataLoader cannot combine shuffle with a sampler. Without weighted
        # sampling, a fresh shuffled order is generated on every iteration.
        shuffle=sampler is None,
        generator=loader_generator,
        # Persistent workers avoid process startup each epoch. Their seeded RNG
        # streams continue advancing, so random augmentation does not repeat.
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

    model = Fusion(
        pretrained=True,
        image_size=args.image_size,
        num_frequency_bins=args.num_frequency_bins,
        frequency_embedding_dim=args.frequency_embedding_dim,
        fusion_hidden_dim=args.fusion_hidden_dim,
        dropout=args.dropout,
    ).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Fine-tune the pretrained spatial backbone conservatively, while training
    # newly introduced branch heads and fusion layers at the main learning rate.
    backbone_parameters = list(
        model.spatial_detector.backbone.features.parameters()
    )
    backbone_parameter_ids = {id(parameter) for parameter in backbone_parameters}
    # Identity checks place every current Parameter in exactly one optimizer
    # group without depending on parameter-name string matching.
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

    # float16 autocast/scaling is enabled only on CUDA. Supplying --amp on CPU
    # safely retains full precision because both contexts become no-ops.
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_selection_score = -float("inf")
    best_epoch = None

    for epoch in range(1, args.epochs + 1):
        # ``train`` enables training-time layer behavior; it does not reinitialize
        # weights. Each epoch therefore continues from the preceding epoch.
        model.train()
        loss_sums = {
            "total": 0.0,
            "fused": 0.0,
            "spatial_aux": 0.0,
            "frequency_aux": 0.0,
        }

        # Starting this for-loop calls iter(train_loader). That creates a fresh
        # shuffle/sample sequence even though the DataLoader object was built once.
        for spatial_images, frequency_images, labels in train_loader:
            spatial_images = spatial_images.to(device, non_blocking=True)
            frequency_images = frequency_images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float().unsqueeze(1)

            optimizer.zero_grad(set_to_none=True)
            # Keep the forward pass and loss inside autocast. GradScaler protects
            # small float16 gradients from underflow and is a no-op without AMP.
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                outputs = model(spatial_images, frequency_images)
                fused_loss = criterion(outputs["fused_logits"], labels)
                spatial_aux_loss = criterion(outputs["spatial_logits"], labels)
                frequency_aux_loss = criterion(
                    outputs["frequency_logits"],
                    labels,
                )
                # Auxiliary losses keep both branches individually predictive;
                # the fused head remains the primary objective with weight 1.
                loss = (
                    fused_loss
                    + args.spatial_aux_loss_weight * spatial_aux_loss
                    + args.frequency_aux_loss_weight * frequency_aux_loss
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_size = spatial_images.size(0)
            loss_sums["total"] += loss.item() * batch_size
            loss_sums["fused"] += fused_loss.item() * batch_size
            loss_sums["spatial_aux"] += spatial_aux_loss.item() * batch_size
            loss_sums["frequency_aux"] += frequency_aux_loss.item() * batch_size

        # The scheduler advances once per completed training epoch. A saved full
        # checkpoint therefore contains the learning-rate state for the next epoch.
        scheduler.step()
        validation_metrics = {
            name: validate(model, loader, criterion, device)
            for name, loader in validation_loaders.items()
        }
        # Average dataset-level scores rather than pooling images. Thus every
        # selected validation dataset has equal influence regardless of its size.
        selection_score = float(
            np.mean(
                [
                    validation_metrics[name][args.selection_metric]
                    for name in SELECTION_VALIDATION_SETS
                ]
            )
        )
        # Preserve every validation result in the sidecar/checkpoint, including
        # monitor-only sets that do not affect best-epoch selection.
        checkpoint_metrics = {
            "selection_metric": args.selection_metric,
            "selection_validation_sets": SELECTION_VALIDATION_SETS,
            "selection_score": selection_score,
            "sets": validation_metrics,
        }

        denominator = len(train_loader.dataset)
        averaged_losses = {
            name: value / denominator for name, value in loss_sums.items()
        }
        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train total={averaged_losses['total']:.4f}, "
            f"fused={averaged_losses['fused']:.4f}, "
            f"spatial_aux={averaged_losses['spatial_aux']:.4f}, "
            f"frequency_aux={averaged_losses['frequency_aux']:.4f} | "
            f"selection_score={selection_score:.4f}"
        )
        for name, metrics in validation_metrics.items():
            _print_validation(name, metrics)

        # Strict comparison keeps the earlier checkpoint when scores tie. The
        # -infinity initializer makes the first finite result an improvement.
        if selection_score > best_selection_score:
            best_selection_score = selection_score
            best_epoch = epoch
            print(f"  New best epoch ({args.selection_metric} improved).")
            _save_best_artifacts(
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                checkpoint_metrics,
                args,
            )

    print(
        f"Training complete | best_epoch={best_epoch}, "
        f"best_{args.selection_metric}={best_selection_score:.4f}"
    )


def parse_args():
    """Parse command-line overrides for model, optimization, and output settings."""
    parser = argparse.ArgumentParser(
        description=(
            "Train a direct spatial/full-radial-spectrum fusion detector "
            "(no STAL and no spectral-tail branch). Dataset paths are the "
            "list macros at the top of train.py."
        )
    )
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--backbone_lr", type=float, default=BACKBONE_LEARNING_RATE)
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--image_size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--num_frequency_bins", type=int, default=64)
    parser.add_argument("--frequency_embedding_dim", type=int, default=128)
    parser.add_argument("--fusion_hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--spatial_aux_loss_weight", type=float, default=0.1)
    parser.add_argument("--frequency_aux_loss_weight", type=float, default=0.1)
    parser.add_argument(
        "--balance_strategy",
        choices=("auto", "none", "loss", "sampler"),
        default="auto",
    )
    parser.add_argument("--balance_tolerance", type=float, default=0.95)
    parser.add_argument(
        "--selection_metric",
        choices=("balanced_accuracy", "roc_auc"),
        default=SELECTION_METRIC,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=USE_AMP,
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--weights_path", default=WEIGHTS_PATH)
    parser.add_argument(
        "--save_training_checkpoint",
        action=argparse.BooleanOptionalAction,
        default=SAVE_TRAINING_CHECKPOINT,
    )
    parser.add_argument(
        "--training_checkpoint_path",
        default=TRAINING_CHECKPOINT_PATH,
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
