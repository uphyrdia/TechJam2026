import io
import os
import random

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as F


class SocialMediaDegradation:
    """Post-processing applied only to the spatial training view.

    The ranges intentionally include clean and mildly degraded examples. Very
    strong blur/noise/compression combinations can erase all forensic evidence
    and are therefore avoided here.
    """

    def __init__(self):
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

    def __call__(self, img):
        # Keep some samples nearly clean. This prevents the model from seeing
        # only heavily corrupted images during training.
        severity = random.choices(
            population=("clean", "typical", "severe"),
            weights=(0.25, 0.55, 0.20),
            k=1,
        )[0]
        if severity == "clean":
            return img

        if random.random() < 0.5:
            img = self.color_jitter(img)

        # A realistic processing order: optional blur, resampling, small noise,
        # and finally compression.
        if random.random() < (0.15 if severity == "typical" else 0.35):
            sigma = random.uniform(0.1, 0.8 if severity == "typical" else 1.2)
            kernel_size = max(3, 2 * int(3 * sigma) + 1)
            img = F.gaussian_blur(img, kernel_size=kernel_size, sigma=sigma)

        if random.random() < (0.45 if severity == "typical" else 0.80):
            low = 0.55 if severity == "typical" else 0.35
            scale = random.uniform(low, 0.95)
            interpolation = random.choice(self.resize_interpolations)
            width, height = img.size
            small_height = max(1, round(height * scale))
            small_width = max(1, round(width * scale))
            img = F.resize(
                img,
                [small_height, small_width],
                interpolation=interpolation,
                antialias=True,
            )
            img = F.resize(
                img,
                [height, width],
                interpolation=interpolation,
                antialias=True,
            )

        if random.random() < (0.10 if severity == "typical" else 0.20):
            sigma = random.uniform(0.002, 0.015 if severity == "typical" else 0.03)
            tensor = F.to_tensor(img)
            tensor = torch.clamp(tensor + torch.randn_like(tensor) * sigma, 0, 1)
            img = F.to_pil_image(tensor)

        jpeg_probability = 0.55 if severity == "typical" else 0.90
        if random.random() < jpeg_probability:
            minimum_quality = 60 if severity == "typical" else 40
            img = self.jpeg_compress(img, random.randint(minimum_quality, 100))

        return img

    @staticmethod
    def jpeg_compress(img, quality):
        """Simulate JPEG re-encoding in memory."""
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        with Image.open(buffer) as compressed:
            return compressed.convert("RGB")


class STALAugmentation:
    """Create aligned spatial and frequency views for STAL-style training.

    Both views receive exactly the same crop and flip. Only the spatial view is
    exposed to spectrum-destroying social-media degradation. The frequency
    view is used by the training-only frequency teacher.
    """

    def __init__(self, size=224):
        self.size = size
        self.degrade = SocialMediaDegradation()

    def __call__(self, img):
        top, left, height, width = transforms.RandomResizedCrop.get_params(
            img,
            scale=(0.8, 1.0),
            ratio=(0.9, 1.1),
        )
        shared = F.resized_crop(
            img,
            top,
            left,
            height,
            width,
            [self.size, self.size],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
        if random.random() < 0.5:
            shared = F.hflip(shared)

        frequency_view = shared.copy()
        spatial_view = self.degrade(shared.copy())
        return spatial_view, frequency_view


class Augmentation:
    """Backward-compatible single-view augmentation."""

    def __init__(self, size=224):
        self.two_view = STALAugmentation(size=size)

    def __call__(self, img):
        spatial_view, _ = self.two_view(img)
        return spatial_view


if __name__ == "__main__":
    augmenter = STALAugmentation()
    input_folder = "./church/church/church/train/"
    output_folder = "./church/church/church/augmented/"
    os.makedirs(output_folder, exist_ok=True)

    extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    image_files = [
        filename
        for filename in os.listdir(input_folder)
        if filename.lower().endswith(extensions)
    ][:10]

    for filename in image_files:
        image_path = os.path.join(input_folder, filename)
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        spatial, frequency = augmenter(image)
        spatial.save(os.path.join(output_folder, f"spatial_{filename}"))
        frequency.save(os.path.join(output_folder, f"frequency_{filename}"))

    print(f"Saved {len(image_files)} aligned view pairs.")
