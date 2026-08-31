import io
import random

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as F


class SocialMediaDegradation:
    """Random deployment-style post-processing for spatial training.

    Twenty-five percent of samples remain clean. Typical and severe branches
    reproduce the probabilities and ranges used by the previous robust
    augmentation script without creating temporary files.
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

    def __call__(self, image):
        severity = random.choices(
            population=("clean", "typical", "severe"),
            weights=(0.25, 0.55, 0.20),
            k=1,
        )[0]
        if severity == "clean":
            return image

        if random.random() < 0.5:
            image = self.color_jitter(image)

        blur_probability = 0.15 if severity == "typical" else 0.35
        if random.random() < blur_probability:
            maximum_sigma = 0.8 if severity == "typical" else 1.2
            sigma = random.uniform(0.1, maximum_sigma)
            kernel_size = max(3, 2 * int(3 * sigma) + 1)
            image = F.gaussian_blur(
                image,
                kernel_size=kernel_size,
                sigma=sigma,
            )

        resize_probability = 0.45 if severity == "typical" else 0.80
        if random.random() < resize_probability:
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

        noise_probability = 0.10 if severity == "typical" else 0.20
        if random.random() < noise_probability:
            maximum_noise = 0.015 if severity == "typical" else 0.03
            sigma = random.uniform(0.002, maximum_noise)
            tensor = F.to_tensor(image)
            tensor = torch.clamp(
                tensor + torch.randn_like(tensor) * sigma,
                0.0,
                1.0,
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
    def jpeg_compress(image, quality):
        """Simulate JPEG re-encoding entirely in memory."""
        buffer = io.BytesIO()
        image.save(
            buffer,
            format="JPEG",
            quality=quality,
            subsampling=2,
        )
        buffer.seek(0)
        with Image.open(buffer) as compressed:
            compressed.load()
            return compressed.convert("RGB").copy()


class SpatialAugmentation:
    """Crop, flip, and degrade one spatial training view."""

    def __init__(self, size=224):
        self.size = size
        self.degrade = SocialMediaDegradation()

    def __call__(self, image):
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

        return self.degrade(view)


class Augmentation(SpatialAugmentation):
    """Backward-compatible alias for older imports."""
