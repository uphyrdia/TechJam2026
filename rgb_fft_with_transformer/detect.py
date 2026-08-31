import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from model import Full2DFrequencyFusionClassifier


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _torch_load_weights(path, device):
    """Use safe weights-only loading when supported."""
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def _extract_checkpoint(checkpoint):
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must be a dictionary")

    if "model_state_dict" not in checkpoint:
        raise ValueError(
            "This is not a full-2D frequency-fusion checkpoint. Train with "
            "train_full2d_frequency_fusion.py."
        )
    if "model_config" not in checkpoint:
        raise ValueError(
            "Checkpoint has no model_config, so the spectral Transformer "
            "architecture cannot be reconstructed safely."
        )

    return checkpoint["model_state_dict"], checkpoint["model_config"]


def load_model(model_path, device):
    checkpoint = _torch_load_weights(model_path, device)
    state_dict, model_config = _extract_checkpoint(checkpoint)

    model = Full2DFrequencyFusionClassifier(
        pretrained=False,
        **model_config,
    )
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model, model_config


def print_checkpoint_metadata(model_path):
    metadata_path = Path(f"{model_path}.meta.json")
    if not metadata_path.exists():
        print(
            "Checkpoint metadata not found. The embedded model_config will "
            "still be used."
        )
        return

    try:
        with metadata_path.open("r", encoding="utf-8") as metadata_file:
            metadata = json.load(metadata_file)
        print(f"Checkpoint score: {metadata.get('score_meaning', 'unknown')}")
        print(f"Checkpoint model: {metadata.get('model', 'unknown')}")
    except (OSError, json.JSONDecodeError) as error:
        print(f"Could not read checkpoint metadata: {error}")


def load_rgb_image(image_path):
    """Load RGB and composite transparent pixels onto white."""
    with Image.open(image_path) as source:
        source.load()
        has_transparency = (
            source.mode in {"RGBA", "LA"}
            or (source.mode == "P" and "transparency" in source.info)
        )
        if has_transparency:
            rgba = source.convert("RGBA")
            background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            return Image.alpha_composite(background, rgba).convert("RGB")
        return source.convert("RGB")


def process_image(
    image_path,
    model,
    resize,
    to_tensor,
    normalize,
    device,
    threshold,
):
    image = resize(load_rgb_image(image_path))
    frequency_tensor = to_tensor(image)
    spatial_tensor = normalize(frequency_tensor)

    spatial_tensor = spatial_tensor.unsqueeze(0).to(device)
    frequency_tensor = frequency_tensor.unsqueeze(0).to(device)

    with torch.inference_mode():
        outputs = model(spatial_tensor, frequency_tensor)
        fused_probability = torch.sigmoid(outputs["fused_logits"]).item()
        spatial_probability = torch.sigmoid(outputs["spatial_logits"]).item()
        frequency_probability = torch.sigmoid(
            outputs["frequency_logits"]
        ).item()
        fusion_gate = outputs["fusion_gate"].item()

    predicted_label = int(fused_probability >= threshold)
    return {
        # Retain "pred" for compatibility with existing ROC/AUC scripts.
        "pred": fused_probability,
        "ai_probability": fused_probability,
        "spatial_ai_probability": spatial_probability,
        "frequency_ai_probability": frequency_probability,
        "fusion_gate": fusion_gate,
        "predicted_label": predicted_label,
        "predicted_class": "ai_generated" if predicted_label else "real",
    }


def collect_image_paths(image_dir, recursive):
    image_dir = Path(image_dir)
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")

    iterator = image_dir.rglob("*") if recursive else image_dir.iterdir()
    return sorted(
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def print_summary(results, ground_truth):
    if not results:
        print("No supported images were processed.")
        return

    fused = [item["ai_probability"] for item in results]
    spatial = [item["spatial_ai_probability"] for item in results]
    frequency = [item["frequency_ai_probability"] for item in results]
    ai_fraction = sum(
        item["predicted_label"] == 1 for item in results
    ) / len(results)

    print(
        f"Processed {len(results)} images | "
        f"mean P(AI): fused={sum(fused) / len(fused):.4f}, "
        f"spatial={sum(spatial) / len(spatial):.4f}, "
        f"frequency={sum(frequency) / len(frequency):.4f} | "
        f"fusion gate={results[0]['fusion_gate']:.4f} | "
        f"predicted AI={ai_fraction:.2%}"
    )

    if ground_truth is not None:
        true_label = 1 if ground_truth == "ai_generated" else 0
        correct = sum(
            item["predicted_label"] == true_label for item in results
        )
        print(
            f"Fused accuracy for ground truth '{ground_truth}': "
            f"{correct / len(results):.2%} ({correct}/{len(results)})"
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "AI-image detection using ConvNeXt and the complete two-dimensional "
            "RGB FFT log-power spectrum"
        )
    )
    parser.add_argument("--image_dir", required=True)
    parser.add_argument(
        "--model_path",
        default="aigid.pth",
    )
    parser.add_argument(
        "--output_json",
        default="predictions.json",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classify as AI-generated when fused P(AI) reaches this value",
    )
    parser.add_argument(
        "--ground_truth",
        choices=("real", "ai_generated"),
        default=None,
        help="If every input has this class, also report fused accuracy",
    )
    parser.add_argument("--recursive", action="store_true")
    args = parser.parse_args()

    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0 and 1")

    device = torch.device(args.device)
    model, model_config = load_model(args.model_path, device)
    print(f"Loaded architecture: {model_config}")
    print_checkpoint_metadata(args.model_path)

    image_size = model_config["image_size"]
    resize = transforms.Resize(
        (image_size, image_size),
        antialias=True,
    )
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
    )

    image_paths = collect_image_paths(args.image_dir, args.recursive)
    results = []
    for image_path in image_paths:
        try:
            prediction = process_image(
                image_path,
                model,
                resize,
                to_tensor,
                normalize,
                device,
                args.threshold,
            )
            results.append({"image_path": str(image_path), **prediction})
        except Exception as error:
            print(f"Error processing {image_path}: {error}")

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(results, output_file, indent=2)

    print_summary(results, args.ground_truth)
    print(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    main()
