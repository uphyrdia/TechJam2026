"""Run batch inference with the spatial/radial-spectrum fusion classifier.

The detector discovers supported images in a directory, applies the two input
representations expected by :class:`model.Fusion`, and writes per-image scores
and thresholded labels to JSON. A training-produced metadata sidecar supplies
the image size when available; three head dimensions are inferred from saved
parameter shapes, while the Fusion/ConvNeXt-Small architecture is fixed here.
"""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from model import Fusion

MODEL_PATH = "weights.pth"
OUTPUT_JSON = "predictions.json"
# This threshold converts the fused sigmoid score into a class label only; the
# unmodified score is always retained in the JSON output.
THRESHOLD = 0.5

# =============================================================================

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def _torch_load_weights(path, device):
    """Load a checkpoint onto ``device`` using safe loading when supported."""
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)

    
def _extract_model_state_dict(checkpoint):
    """Return model weights from either supported checkpoint layout.

    Training checkpoints wrap the weights under ``model_state_dict``; exported
    inference files may consist of the state dictionary itself.
    """
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint is not a state dictionary")
    if "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint

    
def _infer_architecture(state_dict):
    """Infer constructor dimensions from the shapes of stored parameters.

    This avoids duplicating architecture-size settings in the command line and
    allows inference to follow the dimensions that were actually trained.
    """
    try:
        frequency_weight = state_dict["frequency_encoder.1.weight"]
        fusion_weight = state_dict["fusion_classifier.1.weight"]
    except KeyError as error:
        raise ValueError(
            "This is not a spatial/radial-spectrum fusion checkpoint. "
            "Train it with the accompanying train.py."
        ) from error

    return {
        # For a Linear layer, weight shape is [out_features, in_features].
        "num_frequency_bins": frequency_weight.shape[1],
        "frequency_embedding_dim": frequency_weight.shape[0],
        "fusion_hidden_dim": fusion_weight.shape[0],
    }

    
def read_checkpoint_metadata(model_path, print_summary=True):
    """Read the optional ``<checkpoint>.meta.json`` training sidecar.

    Missing or unreadable metadata is non-fatal because the state dictionary
    contains the learned weights and most architecture dimensions.  Callers
    receive an empty dictionary and can apply compatibility defaults.
    """
    metadata_path = Path(f"{model_path}.meta.json")
    if not metadata_path.exists():
        if print_summary:
            print(
                "Checkpoint metadata not found; assuming image_size=224. "
                "This is normal for manually moved weights."
            )
        return {}

    try:
        with metadata_path.open("r", encoding="utf-8") as metadata_file:
            metadata = json.load(metadata_file)
        if print_summary:
            model_metadata = metadata.get("model", {})
            print(f"Checkpoint score: {metadata.get('score_meaning', 'unknown')}")
            print(
                "Checkpoint model: direct spatial/radial-spectrum fusion | "
                f"bins={model_metadata.get('num_frequency_bins', 'unknown')}"
            )
        return metadata
    except (OSError, json.JSONDecodeError) as error:
        if print_summary:
            print(f"Could not read checkpoint metadata: {error}")
        return {}


def load_model(model_path, device):
    """Reconstruct a trained ``Fusion`` model and put it in inference mode.

    Returns both the model and its expected square input size so preprocessing
    remains consistent with training.
    """
    checkpoint = _torch_load_weights(model_path, device)
    state_dict = _extract_model_state_dict(checkpoint)
    architecture = _infer_architecture(state_dict)

    metadata = read_checkpoint_metadata(model_path)
    image_size = metadata.get("model", {}).get("image_size")
    # A full training checkpoint stores image size in its saved arguments,
    # whereas deployable weights normally obtain it from the JSON sidecar.
    if image_size is None and "arguments" in checkpoint:
        image_size = checkpoint["arguments"].get("image_size")
    # 224 preserves compatibility with weights for which neither source exists.
    image_size = int(image_size or 224)

    model = Fusion(
        pretrained=False,
        image_size=image_size,
        **architecture,
    )
    # Strict loading catches incompatible or incomplete checkpoints immediately.
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    # Evaluation mode disables this model's dropout and stochastic-depth
    # behavior; it does not alter or reset learned parameters.
    model.eval()
    return model, image_size


def load_rgb_image(image_path):
    """Load an image as RGB, compositing transparent pixels onto white.

    Explicit compositing avoids discarding alpha and exposing arbitrary RGB
    values stored beneath transparent pixels.
    """
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
    """Run both model branches for one image and format its prediction.

    The frequency branch receives unnormalized RGB values in ``[0, 1]`` because
    it computes its spectral representation from image intensities.  The
    spatial branch receives a separately ImageNet-normalized view, matching the
    pretrained spatial encoder's training convention.
    """
    image = resize(load_rgb_image(image_path))
    frequency_tensor = to_tensor(image)
    spatial_tensor = normalize(frequency_tensor)

    # Add the batch dimension expected by the model, then place both inputs on
    # the same device as its parameters.
    spatial_tensor = spatial_tensor.unsqueeze(0).to(device)
    frequency_tensor = frequency_tensor.unsqueeze(0).to(device)
    # inference_mode disables gradient recording and version tracking.  No AMP
    # autocast is used here, so tensors retain their normal model/input dtype.
    with torch.inference_mode():
        outputs = model(spatial_tensor, frequency_tensor)
        # Each head emits a binary logit. Sigmoid maps it to the AI score that
        # this project interprets as P(ai_generated), though the score is not
        # guaranteed to be calibrated as a real-world posterior probability.
        fused_probability = torch.sigmoid(outputs["fused_logits"]).item()
        spatial_probability = torch.sigmoid(outputs["spatial_logits"]).item()
        frequency_probability = torch.sigmoid(outputs["frequency_logits"]).item()

    # Branch scores are diagnostic.  Only the fused score determines the final
    # class, and equality at the boundary is assigned to the AI-generated class.
    predicted_label = int(fused_probability >= threshold)
    return {
        "pred": fused_probability,
        "ai_probability": fused_probability,
        "spatial_ai_probability": spatial_probability,
        "frequency_ai_probability": frequency_probability,
        "predicted_label": predicted_label,
        "predicted_class": "ai_generated" if predicted_label else "real",
    }


def collect_image_paths(image_directory, recursive):
    """Discover supported image files and return them in deterministic order.

    When ``recursive`` is false, only direct children are considered; when true,
    all descendant directories are searched.  Extension matching ignores case.
    """
    folder = Path(image_directory).expanduser()
    if not folder.is_dir():
        raise FileNotFoundError(
            f"Image directory does not exist: {folder}"
        )

    iterator = folder.rglob("*") if recursive else folder.iterdir()
    # Sorting makes output JSON order stable across filesystems and runs.
    paths = sorted(
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        raise ValueError(f"No supported images found under: {folder}")
    return paths


def print_summary(results):
    """Print aggregate fused and branch sigmoid scores for processed images."""
    if not results:
        print("No supported images were processed.")
        return

    fused = [item["ai_probability"] for item in results]
    spatial = [item["spatial_ai_probability"] for item in results]
    frequency = [item["frequency_ai_probability"] for item in results]
    ai_fraction = sum(item["predicted_label"] == 1 for item in results) / len(
        results
    )
    print(
        f"Processed {len(results)} images | "
        f"mean P(AI): fused={sum(fused) / len(fused):.4f}, "
        f"spatial={sum(spatial) / len(spatial):.4f}, "
        f"frequency={sum(frequency) / len(frequency):.4f} | "
        f"predicted AI={ai_fraction:.2%}"
    )

def parse_args():
    """Parse command-line options and validate the decision threshold."""
    parser = argparse.ArgumentParser(
        description=(
            "Detect AI-generated images with direct spatial/full-radial-"
            "spectrum fusion. Scores mean P(ai_generated)."
        )
    )
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--model_path", default=MODEL_PATH)
    parser.add_argument("--output_json", default=OUTPUT_JSON)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    # store_true means recursive discovery is disabled unless this flag appears.
    parser.add_argument("--recursive", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1")
    return args


def main():
    """Load the detector, process discovered images, and write JSON results."""
    args = parse_args()
    # An explicit --device value is honored as-is; otherwise CUDA is selected
    # only when PyTorch reports it as available.
    device = torch.device(args.device)
    model, image_size = load_model(args.model_path, device)

    # Construct stateless transforms once and reuse them for every input image.
    resize = transforms.Resize((image_size, image_size), antialias=True)
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
            # A corrupt/unsupported individual image should not abort the batch.
            print(f"Error processing {image_path}: {error}")

    output_path = Path(args.output_json).expanduser()
    # Permit nested output paths without requiring callers to create them first.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(results, output_file, indent=2)

    print_summary(results)
    print(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    main()
