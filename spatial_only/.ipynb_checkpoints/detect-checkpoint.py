import argparse
import json
from pathlib import Path

from PIL import Image, ImageOps
import torch
from torchvision import transforms

from model import AIGCClassifier


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def torch_load_weights(path, device):
    """Load tensor-only checkpoints without executing pickled Python code."""
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError as error:
        raise RuntimeError(
            "This script requires a PyTorch version that supports "
            "torch.load(..., weights_only=True). Upgrade PyTorch or convert "
            "the checkpoint in a trusted environment."
        ) from error


def extract_spatial_state_dict(checkpoint):
    """Accept raw spatial weights and older wrapped training checkpoints."""
    if not isinstance(checkpoint, dict):
        raise ValueError("The checkpoint is not a state dictionary")

    for container_key in ("model_state_dict", "spatial_state_dict"):
        candidate = checkpoint.get(container_key)
        if isinstance(candidate, dict):
            checkpoint = candidate
            break

    # DataParallel may add this prefix to every key.
    if checkpoint and all(key.startswith("module.") for key in checkpoint):
        checkpoint = {
            key[len("module.") :]: value
            for key, value in checkpoint.items()
        }

    # Also accept a complete older STAL/fusion checkpoint by extracting only
    # its deployable spatial detector. Training-only branches are ignored.
    prefix = "spatial_detector."
    spatial_items = {
        key[len(prefix) :]: value
        for key, value in checkpoint.items()
        if key.startswith(prefix)
    }
    if spatial_items:
        checkpoint = spatial_items

    if not checkpoint or not all(
        isinstance(key, str) and torch.is_tensor(value)
        for key, value in checkpoint.items()
    ):
        raise ValueError(
            "Could not find a tensor-only spatial model state dictionary"
        )
    return checkpoint


def load_model(model_path, device):
    checkpoint = torch_load_weights(model_path, device)
    state_dict = extract_spatial_state_dict(checkpoint)
    model = AIGCClassifier(pretrained=False)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def load_rgb_image(image_path):
    """Apply EXIF orientation and composite transparency onto white."""
    with Image.open(image_path) as source:
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


def process_image(image_path, model, preprocess, device, threshold):
    image = load_rgb_image(image_path)
    image_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        logit = model(image_tensor)
        ai_probability = float(torch.sigmoid(logit).item())

    predicted_label = int(ai_probability >= threshold)
    return {
        # Retained for compatibility with the user's existing analysis code.
        "pred": ai_probability,
        "ai_probability": ai_probability,
        "predicted_label": predicted_label,
        "predicted_class": (
            "ai_generated" if predicted_label == 1 else "real"
        ),
    }


def collect_image_paths(image_directory, recursive):
    folder = Path(image_directory).expanduser()
    if not folder.is_dir():
        raise FileNotFoundError(
            f"Image directory does not exist: {folder}"
        )

    iterator = folder.rglob("*") if recursive else folder.iterdir()
    paths = sorted(
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        raise ValueError(f"No supported images found under: {folder}")
    return paths


def print_checkpoint_metadata(model_path):
    metadata_path = Path(f"{model_path}.meta.json")
    if not metadata_path.is_file():
        print(
            "Checkpoint metadata not found; this is normal for older "
            "spatial weights."
        )
        return

    try:
        with metadata_path.open("r", encoding="utf-8") as metadata_file:
            metadata = json.load(metadata_file)
    except (OSError, json.JSONDecodeError) as error:
        print(f"Could not read checkpoint metadata: {error}")
        return

    print(
        "Checkpoint | "
        f"epoch={metadata.get('selected_epoch', 'unknown')}, "
        f"selection metric={metadata.get('selection_metric', 'unknown')}"
    )


def print_summary(results, ground_truth):
    if not results:
        print("No images were processed successfully.")
        return

    probabilities = [item["ai_probability"] for item in results]
    predicted_ai_count = sum(
        item["predicted_label"] == 1 for item in results
    )
    print(
        f"Processed {len(results):,} images | "
        f"P(AI) min={min(probabilities):.4f}, "
        f"mean={sum(probabilities) / len(probabilities):.4f}, "
        f"max={max(probabilities):.4f} | "
        f"predicted AI={predicted_ai_count / len(results):.2%}"
    )

    if ground_truth is not None:
        true_label = 1 if ground_truth == "ai_generated" else 0
        correct = sum(
            item["predicted_label"] == true_label for item in results
        )
        print(
            f"Accuracy for ground truth '{ground_truth}': "
            f"{correct / len(results):.2%} ({correct}/{len(results)})"
        )


def build_argument_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Run the spatial-only ConvNeXt AI-image detector. Scores are "
            "P(ai_generated): 0 means real and 1 means AI-generated."
        )
    )
    parser.add_argument("--image_dir", required=True)
    parser.add_argument(
        "--model_path",
        default="weights.pth",
    )
    parser.add_argument("--output_json", default="predictions.json")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classify as AI-generated when P(AI) is at least this value",
    )
    parser.add_argument(
        "--ground_truth",
        choices=("real", "ai_generated"),
        default=None,
        help="If every image has this class, also report accuracy",
    )
    parser.add_argument("--recursive", action="store_true")
    return parser


def main():
    parser = build_argument_parser()
    args = parser.parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0 and 1")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    model = load_model(args.model_path, device)
    print_checkpoint_metadata(args.model_path)
    preprocess = transforms.Compose(
        [
            transforms.Resize(
                (IMAGE_SIZE, IMAGE_SIZE),
                antialias=True,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )

    image_paths = collect_image_paths(args.image_dir, args.recursive)
    results = []
    failed_count = 0
    for image_path in image_paths:
        try:
            prediction = process_image(
                image_path,
                model,
                preprocess,
                device,
                args.threshold,
            )
            results.append({"image_path": str(image_path), **prediction})
        except (OSError, ValueError, RuntimeError) as error:
            failed_count += 1
            print(f"Skipped {image_path}: {error}")

    output_path = Path(args.output_json).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(results, output_file, indent=2)

    print_summary(results, args.ground_truth)
    if failed_count:
        print(f"Skipped {failed_count:,} unreadable/invalid image(s)")
    print(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    main()
