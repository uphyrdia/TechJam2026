"""Training-time image augmentation for the spatial/spectral classifier.

The pipeline first creates a common geometric view, then optionally simulates
the kinds of quality loss introduced by social-media processing.  It returns a
PIL image so the dataset can derive both model inputs from exactly the same
pixels: an ImageNet-normalized tensor for the spatial branch and an
unnormalized tensor for the FFT branch.
"""

import io
import random

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as F


class Degradation:
    """Randomly simulate post-processing seen in deployed image pipelines.

    A severity tier is sampled independently for every transformed view. Within
    a non-clean tier, each degradation is sampled independently, so several
    effects may be composed in sequence. The input and output are PIL images
    with unchanged spatial dimensions.
    """

    def __init__(self):
        """Create reusable torchvision transforms and interpolation choices."""

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
        """Return a randomly degraded version of ``image``.

        The clean tier skips this entire degradation stage.  The probabilities
        below are conditional on the sampled tier, rather than probabilities
        across the full training distribution.
        """

        # ``random.choices`` normalizes these relative weights internally.
        severity = random.choices(
            population=("clean", "typical", "severe"),
            weights=(0.25, 0.55, 0.20),
            k=1,
        )[0]
        if severity == "clean":
            return image

        # Color jitter uses the same moderate range in both degraded tiers.
        if random.random() < 0.5:
            image = self.color_jitter(image)

        if random.random() < (0.15 if severity == "typical" else 0.35):
            sigma = random.uniform(0.1, 0.8 if severity == "typical" else 1.2)
            # Derive an odd kernel size from sigma, with the minimum size of 3
            # required by the Gaussian blur operator.
            kernel_size = max(3, 2 * int(3 * sigma) + 1)
            image = F.gaussian_blur(
                image,
                kernel_size=kernel_size,
                sigma=sigma,
            )

        if random.random() < (0.45 if severity == "typical" else 0.80):
            scale = random.uniform(0.55 if severity == "typical" else 0.35, 0.95)
            interpolation = random.choice(self.resize_interpolations)
            # PIL reports (width, height), whereas torchvision resize expects
            # its requested output size in [height, width] order.
            width, height = image.size
            small_height = max(1, round(height * scale))
            small_width = max(1, round(width * scale))
            # Downsample and restore the original dimensions to discard detail
            # without changing the shape expected by the downstream model.
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
            sigma = random.uniform(0.002, 0.015 if severity == "typical" else 0.03)
            # Noise is added in float [0, 1] space.  Clamping preserves the
            # valid image range before converting back to PIL for later steps.
            tensor = F.to_tensor(image)
            tensor = torch.clamp(
                tensor + torch.randn_like(tensor) * sigma,
                0,
                1,
            )
            image = F.to_pil_image(tensor)

        jpeg_probability = 0.55 if severity == "typical" else 0.90
        if random.random() < jpeg_probability:
            # Lower JPEG quality produces stronger compression artifacts.
            minimum_quality = 60 if severity == "typical" else 40
            image = self.jpeg_compress(
                image,
                random.randint(minimum_quality, 100),
            )

        return image

    @staticmethod
    def jpeg_compress(image, quality):
        """Round-trip ``image`` through JPEG entirely in memory.

        Converting the decoded result to RGB also detaches it from the
        short-lived buffer-backed PIL image before the context manager closes.
        """

        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        with Image.open(buffer) as compressed:
            return compressed.convert("RGB")


class Augmentation:
    """Create one augmented view shared by the spatial and FFT branches.

    The dataset converts this view into an ImageNet-normalized tensor for the
    spatial branch and an unnormalized [0, 1] tensor for the spectrum branch.
    Sharing the pixels keeps the paired branch inputs spatially aligned.
    """

    def __init__(self, size=224):
        """Configure the square output size and degradation stage."""

        self.size = size
        self.degrade = Degradation()

    def __call__(self, image):
        """Crop, resize, optionally mirror, and degrade one PIL image."""

        # Sampling crop parameters explicitly allows this single transformed
        # view to be reused by both branches instead of augmenting them apart.
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
        # Horizontal mirroring is label-preserving for real-vs-generated image
        # classification and expands geometric variation without interpolation.
        if random.random() < 0.5:
            view = F.hflip(view)

        return self.degrade(view)
