import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from model import AIGCClassifier


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def _torch_load_weights(path, device):
    """Use safe weights-only loading when the installed PyTorch supports it."""
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def _extract_spatial_state_dict(checkpoint):
    """Accept old raw weights or the resumable STAL training checkpoint."""
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint is not a state dictionary")

    if "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    elif "spatial_state_dict" in checkpoint:
        checkpoint = checkpoint["spatial_state_dict"]

    # A full STAL checkpoint prefixes deployment weights with
    # 'spatial_detector.'. Remove that prefix and ignore training-only modules.
    prefix = "spatial_detector."
    spatial_items = {
        key[len(prefix) :]: value
        for key, value in checkpoint.items()
        if key.startswith(prefix)
    }
    return spatial_items if spatial_items else checkpoint


def load_model(model_path, device):
    model = AIGCClassifier(pretrained=False)
    checkpoint = _torch_load_weights(model_path, device)
    state_dict = _extract_spatial_state_dict(checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def print_checkpoint_metadata(model_path):
    metadata_path = Path(f"{model_path}.meta.json")
    if not metadata_path.exists():
        print(
            "Checkpoint metadata not found (this is normal for older weights). "
            "Verify manually which dataset produced this checkpoint."
        )
        return
    try:
        with open(metadata_path, "r", encoding="utf-8") as metadata_file:
            metadata = json.load(metadata_file)
        print(f"Checkpoint score: {metadata.get('score_meaning', 'unknown')}")
        training_data = metadata.get("training_data")
        if training_data:
            print(f"Checkpoint training data: {training_data}")
    except (OSError, json.JSONDecodeError) as error:
        print(f"Could not read checkpoint metadata: {error}")


def process_image(image_path, model, preprocess, device, threshold):
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        logit = model(image_tensor)
        ai_probability = torch.sigmoid(logit).item()

    predicted_label = int(ai_probability >= threshold)
    return {
        # 'pred' is retained for compatibility with existing evaluation code.
        "pred": ai_probability,
        "ai_probability": ai_probability,
        "predicted_label": predicted_label,
        "predicted_class": "ai_generated" if predicted_label == 1 else "real",
    }


def collect_image_paths(image_dir, recursive):
    image_dir = Path(image_dir)
    if not image_dir.exists():
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

    probabilities = [item["ai_probability"] for item in results]
    ai_fraction = sum(item["predicted_label"] == 1 for item in results) / len(results)
    print(
        f"Processed {len(results)} images | "
        f"P(AI) min={min(probabilities):.4f}, "
        f"mean={sum(probabilities) / len(probabilities):.4f}, "
        f"max={max(probabilities):.4f} | "
        f"predicted AI={ai_fraction:.2%}"
    )

    if ground_truth is not None:
        true_label = 1 if ground_truth == "ai_generated" else 0
        correct = sum(item["predicted_label"] == true_label for item in results)
        print(
            f"Accuracy for ground truth '{ground_truth}': "
            f"{correct / len(results):.2%} ({correct}/{len(results)})"
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "AI-generated image detection. Scores are P(ai_generated): "
            "0 means real and 1 means AI-generated."
        )
    )
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--model_path", default="./weights/aigc_detector.pth")
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
    args = parser.parse_args()

    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1")

    device = torch.device(args.device)
    model = load_model(args.model_path, device)
    print_checkpoint_metadata(args.model_path)
    preprocess = transforms.Compose(
        [
            transforms.Resize((224, 224), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    image_paths = collect_image_paths(args.image_dir, args.recursive)
    results = []
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
        except Exception as error:
            print(f"Error processing {image_path}: {error}")

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(results, output_file, indent=2)

    print_summary(results, args.ground_truth)
    print(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    main()
