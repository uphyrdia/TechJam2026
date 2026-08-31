"""Apply the frequency-aware random augmentation pipeline to an image folder.

Edit the constants in the CONFIGURATION section, then run:

    python augment_image_directory.py
"""

from __future__ import annotations

import io
import random
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as F


# ============================= CONFIGURATION =============================

# Directory containing the original images.
INPUT_DIRECTORY = Path("data/clean_for_demo")

# Directory in which augmented images will be saved.
OUTPUT_DIRECTORY = Path("data/severely_transformed_for_demo")

# Force every image to use the severe degradation tier. Change this to
# "random" only if you want the original 25% clean / 55% typical / 20% severe
# mixture instead.
SEVERITY = "severe"

# The attached training augmentation produces 224 x 224 images.
OUTPUT_SIZE = 224

# Set to an integer for repeatable random results, or None for fresh results.
RANDOM_SEED = None

# If False, an existing destination image is left untouched.
OVERWRITE_EXISTING = True

# ========================================================================


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VALID_SEVERITIES = {"random", "clean", "typical", "severe"}


class Degradation:
    """Random post-processing copied from data_augmentation_frequency.py."""

    def __init__(self, severity: str = "severe") -> None:
        severity = severity.lower()
        if severity not in VALID_SEVERITIES:
            choices = ", ".join(sorted(VALID_SEVERITIES))
            raise ValueError(f"SEVERITY must be one of: {choices}")

        self.severity = severity
        self.color_jitter = transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
        )
        self.resize_interpolations = (
            InterpolationMode.BILINEAR,
            InterpolationMode.BICUBIC,
            InterpolationMode.LANCZOS,
        )

    def _choose_severity(self) -> str:
        if self.severity != "random":
            return self.severity

        return random.choices(
            population=("clean", "typical", "severe"),
            weights=(0.25, 0.55, 0.20),
            k=1,
        )[0]

    def __call__(self, image: Image.Image) -> Image.Image:
        severity = self._choose_severity()
        if severity == "clean":
            return image

        if random.random() < 0.5:
            image = self.color_jitter(image)

        if random.random() < (0.15 if severity == "typical" else 0.35):
            sigma = random.uniform(
                0.1,
                0.8 if severity == "typical" else 1.2,
            )
            kernel_size = max(3, 2 * int(3 * sigma) + 1)
            image = F.gaussian_blur(
                image,
                kernel_size=kernel_size,
                sigma=sigma,
            )

        if random.random() < (0.45 if severity == "typical" else 0.80):
            minimum_scale = 0.55 if severity == "typical" else 0.35
            scale = random.uniform(minimum_scale, 0.95)
            interpolation = random.choice(self.resize_interpolations)
            width, height = image.size
            small_height = max(1, round(height * scale))
            small_width = max(1, round(width * scale))
            image = F.resize(
                image,
                [small_height, small_width],
                interpolation=interpolation,
                antialias=True,
            )
            image = F.resize(
                image,
                [height, width],
                interpolation=interpolation,
                antialias=True,
            )

        if random.random() < (0.10 if severity == "typical" else 0.20):
            sigma = random.uniform(
                0.002,
                0.015 if severity == "typical" else 0.03,
            )
            tensor = F.to_tensor(image)
            tensor = torch.clamp(
                tensor + torch.randn_like(tensor) * sigma,
                0,
                1,
            )
            image = F.to_pil_image(tensor)

        jpeg_probability = 0.55 if severity == "typical" else 0.90
        if random.random() < jpeg_probability:
            minimum_quality = 60 if severity == "typical" else 40
            image = self.jpeg_compress(
                image,
                random.randint(minimum_quality, 100),
            )

        return image

    @staticmethod
    def jpeg_compress(image: Image.Image, quality: int) -> Image.Image:
        """Simulate JPEG re-encoding without writing a temporary file."""
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        with Image.open(buffer) as compressed:
            return compressed.convert("RGB")


class Geometric:
    """Create a cropped, right-angle-rotated, and degraded image view."""

    def __init__(self, size: int = 224, severity: str = "severe") -> None:
        self.size = size
        self.degrade = Degradation(severity=severity)

    def __call__(self, image: Image.Image) -> Image.Image:
        top, left, height, width = transforms.RandomResizedCrop.get_params(
            image,
            scale=(0.8, 1.0),
            ratio=(0.9, 1.1),
        )
        view = F.resized_crop(
            image,
            top,
            left,
            height,
            width,
            [self.size, self.size],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
        if random.random() < 0.5:
            view = F.hflip(view)

        # randrange(4) samples 0, 1, 2, and 3 uniformly, so each right-angle
        # rotation has probability 1/4. The square view keeps its dimensions.
        quarter_turns = random.randrange(4)
        if quarter_turns:
            view = F.rotate(
                view,
                angle=90 * quarter_turns,
                interpolation=InterpolationMode.NEAREST,
            )

        return self.degrade(view)


def save_image(image: Image.Image, output_path: Path) -> None:
    """Save an RGB image using the format implied by its filename."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def main() -> None:
    input_directory = INPUT_DIRECTORY.expanduser().resolve()
    output_directory = OUTPUT_DIRECTORY.expanduser().resolve()

    if not input_directory.is_dir():
        raise NotADirectoryError(
            f"INPUT_DIRECTORY does not exist or is not a directory: "
            f"{input_directory}"
        )
    if input_directory == output_directory:
        raise ValueError("INPUT_DIRECTORY and OUTPUT_DIRECTORY must be different.")

    output_directory.mkdir(parents=True, exist_ok=True)

    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)
        torch.manual_seed(RANDOM_SEED)

    augmenter = Geometric(
        size=OUTPUT_SIZE,
        severity=SEVERITY,
    )
    image_paths = sorted(
        path
        for path in input_directory.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not image_paths:
        print(f"No supported images found in {input_directory}")
        return

    saved = 0
    skipped = 0
    failed = 0

    for image_path in image_paths:
        output_path = output_directory / image_path.name

        if output_path.exists() and not OVERWRITE_EXISTING:
            print(f"Skipping existing file: {output_path.name}")
            skipped += 1
            continue

        try:
            with Image.open(image_path) as source:
                # Apply the EXIF orientation before discarding image metadata.
                image = ImageOps.exif_transpose(source).convert("RGB")

            augmented = augmenter(image)
            save_image(augmented, output_path)
            saved += 1
        except Exception as error:
            print(f"Failed to process {image_path.name}: {error}")
            failed += 1

    print(
        f"Finished: {saved} saved, {skipped} skipped, {failed} failed "
        f"out of {len(image_paths)} images."
    )


if __name__ == "__main__":
    main()
