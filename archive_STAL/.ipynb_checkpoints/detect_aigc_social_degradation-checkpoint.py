import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from model import AIGCClassifier


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _load_degradation_class():
    """Load SocialMediaDegradation from either common attachment filename."""
    try:
        from data_augmentation import SocialMediaDegradation

        return SocialMediaDegradation
    except ModuleNotFoundError as error:
        # Do not hide an import failure inside an existing data_augmentation.py.
        if error.name != "data_augmentation":
            raise

    alternate_path = Path(__file__).with_name("data_augmentation(1).py")
    if not alternate_path.is_file():
        raise ModuleNotFoundError(
            "Could not find data_augmentation.py or data_augmentation(1).py "
            "beside this inference script"
        )

    spec = importlib.util.spec_from_file_location(
        "data_augmentation_attachment",
        alternate_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load augmentation module: {alternate_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.SocialMediaDegradation


SocialMediaDegradation = _load_degradation_class()


def _torch_load_weights(path, device):
    """Use safe weights-only loading when the installed PyTorch supports it."""
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def _extract_spatial_state_dict(checkpoint):
    """Accept raw inference weights or a full STAL training checkpoint."""
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint is not a state dictionary")

    if "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    elif "spatial_state_dict" in checkpoint:
        checkpoint = checkpoint["spatial_state_dict"]

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
            "Checkpoint metadata not found (normal for older weights). "
            "Verify manually which dataset produced this checkpoint."
        )
        return

    try:
        with metadata_path.open("r", encoding="utf-8") as metadata_file:
            metadata = json.load(metadata_file)
        print(f"Checkpoint score: {metadata.get('score_meaning', 'unknown')}")
        training_data = metadata.get("training_data")
        if training_data:
            print(f"Checkpoint training data: {training_data}")
    except (OSError, json.JSONDecodeError) as error:
        print(f"Could not read checkpoint metadata: {error}")


def load_rgb_image(image_path):
    """Load an image and composite transparency onto a white background."""
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
    degradation,
    tensor_transform,
    device,
    threshold,
):
    image = load_rgb_image(image_path)

    # Match the spatial training path: establish the model input geometry first,
    # then sample a random social-media transformation at that resolution.
    image = resize(image)
    image = degradation(image)
    image_tensor = tensor_transform(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        logit = model(image_tensor)
        ai_probability = torch.sigmoid(logit).item()

    predicted_label = int(ai_probability >= threshold)
    return {
        "pred": ai_probability,
        "ai_probability": ai_probability,
        "predicted_label": predicted_label,
        "predicted_class": "ai_generated" if predicted_label == 1 else "real",
        "social_media_degradation_sampled": True,
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
            "AI-generated image detection after one independently sampled "
            "SocialMediaDegradation transformation per image."
        )
    )
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--model_path", default="aigc_detector.pth")
    parser.add_argument("--output_json", default="predictions_degraded.json")
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
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Optional seed for repeatable random degradations. Without it, "
            "each execution samples different transformations."
        ),
    )
    parser.add_argument("--recursive", action="store_true")
    args = parser.parse_args()

    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1")

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    device = torch.device(args.device)
    model = load_model(args.model_path, device)
    print_checkpoint_metadata(args.model_path)

    resize = transforms.Resize((224, 224), antialias=True)
    degradation = SocialMediaDegradation()
    tensor_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    print(
        "Random SocialMediaDegradation enabled. Its configured distribution "
        "keeps approximately 25% of samples clean."
    )

    image_paths = collect_image_paths(args.image_dir, args.recursive)
    results = []
    for image_path in image_paths:
        try:
            prediction = process_image(
                image_path,
                model,
                resize,
                degradation,
                tensor_transform,
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
