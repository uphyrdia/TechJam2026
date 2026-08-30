import argparse
import io
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF


def load_rgb_image(path):
    """Load RGB while handling palette/RGBA transparency consistently."""
    with Image.open(path) as source:
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


def jpeg_compress(image, quality):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    with Image.open(buffer) as compressed:
        compressed.load()
        return compressed.convert("RGB")


def gaussian_blur(image, sigma):
    kernel_size = max(3, 2 * int(3 * sigma) + 1)
    return TF.gaussian_blur(
        image,
        kernel_size=kernel_size,
        sigma=sigma,
    )


def radial_log_power_spectrum(image, num_bins=64, eps=1e-6):
    """Reproduce RadialLogPowerSpectrum from the STAL training model.

    Returns the raw radial mean log-power and its per-image z-score. Radial
    coordinates are normalized by the center-to-corner FFT distance, exactly
    as in the model implementation.
    """
    images = TF.to_tensor(image).unsqueeze(0).float()
    if images.shape[1] != 3:
        raise ValueError("Expected an RGB image")

    luminance = (
        0.299 * images[:, 0]
        + 0.587 * images[:, 1]
        + 0.114 * images[:, 2]
    )
    luminance = luminance - luminance.mean(dim=(-2, -1), keepdim=True)

    height, width = luminance.shape[-2:]
    window_y = torch.hann_window(
        height,
        periodic=False,
        dtype=images.dtype,
    )
    window_x = torch.hann_window(
        width,
        periodic=False,
        dtype=images.dtype,
    )
    window = window_y[:, None] * window_x[None, :]

    spectrum = torch.fft.fft2(luminance * window, norm="ortho")
    spectrum = torch.fft.fftshift(spectrum, dim=(-2, -1))
    log_power = torch.log(spectrum.abs().square() + 1e-12)

    y = torch.arange(height, dtype=images.dtype) - (height - 1) / 2
    x = torch.arange(width, dtype=images.dtype) - (width - 1) / 2
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    radius = torch.sqrt(yy.square() + xx.square())
    radius = radius / radius.max().clamp_min(eps)

    bin_index = torch.clamp(
        (radius * num_bins).long(),
        min=0,
        max=num_bins - 1,
    ).reshape(1, -1)

    radial_sum = torch.zeros(1, num_bins, dtype=log_power.dtype)
    radial_sum.scatter_add_(1, bin_index, log_power.reshape(1, -1))
    counts = torch.bincount(
        bin_index[0],
        minlength=num_bins,
    ).to(log_power.dtype)
    radial = radial_sum / counts.clamp_min(1).unsqueeze(0)

    normalized = radial - radial.mean(dim=1, keepdim=True)
    normalized = normalized / normalized.std(
        dim=1,
        keepdim=True,
    ).clamp_min(eps)

    return (
        radial.squeeze(0).numpy(),
        normalized.squeeze(0).numpy(),
        counts.numpy(),
    )


def tail_statistics(normalized_spectrum, tail_bin, x_coordinates):
    tail = normalized_spectrum[tail_bin:]
    tail_x = x_coordinates[tail_bin:]
    slope = float(np.polyfit(tail_x, tail, deg=1)[0]) if len(tail) >= 2 else 0.0
    return {
        "mean": float(tail.mean()),
        "std": float(tail.std()),
        "minimum": float(tail.min()),
        "maximum": float(tail.max()),
        "slope": slope,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot the normalized radial FFT log-power spectrum used by the "
            "STAL frequency teacher for one image."
        )
    )
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument(
        "--output",
        default=None,
        help="Output plot path; default is <image_stem>_radial_fft.png",
    )
    parser.add_argument(
        "--output_json",
        default=None,
        help="Optional JSON path for raw and normalized spectrum values",
    )
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--num_bins", type=int, default=64)
    parser.add_argument("--tail_start", type=float, default=0.70)
    parser.add_argument(
        "--compare_degraded",
        action="store_true",
        help="Overlay original, heavy blur, JPEG, and blur+JPEG spectra",
    )
    parser.add_argument(
        "--blur_sigma",
        type=float,
        default=2.0,
        help="Gaussian sigma used by --compare_degraded",
    )
    parser.add_argument(
        "--jpeg_quality",
        type=int,
        default=30,
        help="JPEG quality used by --compare_degraded",
    )
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    if args.size <= 1:
        parser.error("--size must be greater than 1")
    if args.num_bins < 2:
        parser.error("--num_bins must be at least 2")
    if not 0.0 <= args.tail_start < 1.0:
        parser.error("--tail_start must be in [0, 1)")
    if args.blur_sigma <= 0:
        parser.error("--blur_sigma must be positive")
    if not 1 <= args.jpeg_quality <= 100:
        parser.error("--jpeg_quality must be between 1 and 100")

    image_path = Path(args.image)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image does not exist: {image_path}")

    image = load_rgb_image(image_path)
    resize = transforms.Resize((args.size, args.size), antialias=True)
    image = resize(image)

    variants = {"original": image}
    if args.compare_degraded:
        blurred = gaussian_blur(image, args.blur_sigma)
        compressed = jpeg_compress(image, args.jpeg_quality)
        blur_and_jpeg = jpeg_compress(blurred, args.jpeg_quality)
        variants.update(
            {
                f"blur sigma={args.blur_sigma:g}": blurred,
                f"JPEG quality={args.jpeg_quality}": compressed,
                "blur + JPEG": blur_and_jpeg,
            }
        )

    bin_centers = (np.arange(args.num_bins) + 0.5) / args.num_bins
    tail_bin = min(
        args.num_bins - 1,
        int(round(args.num_bins * args.tail_start)),
    )
    tail_boundary = tail_bin / args.num_bins

    results = {}
    fig, axis = plt.subplots(figsize=(9, 5.5))
    for name, variant in variants.items():
        raw, normalized, counts = radial_log_power_spectrum(
            variant,
            num_bins=args.num_bins,
        )
        stats = tail_statistics(normalized, tail_bin, bin_centers)
        results[name] = {
            "raw_radial_log_power": raw.tolist(),
            "normalized_radial_log_power": normalized.tolist(),
            "bin_counts": counts.astype(int).tolist(),
            "tail_statistics": stats,
        }
        axis.plot(
            bin_centers,
            normalized,
            linewidth=2,
            label=name,
        )
        print(
            f"{name}: tail mean={stats['mean']:.4f}, "
            f"std={stats['std']:.4f}, slope={stats['slope']:.4f}"
        )

    axis.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    axis.axvspan(
        tail_boundary,
        1.0,
        color="tab:red",
        alpha=0.08,
        label=f"STAL tail bins {tail_bin}-{args.num_bins - 1}",
    )
    axis.axvline(
        1 / math.sqrt(2),
        color="gray",
        linestyle="--",
        linewidth=1,
        label="axis Nyquist radius",
    )
    axis.set_xlim(0, 1)
    axis.set_xlabel("Normalized radial frequency (0=center, 1=FFT corner)")
    axis.set_ylabel("Normalized radial log-power (z-score across bins)")
    axis.set_title(f"Radial FFT spectrum: {image_path.name}")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8)
    fig.tight_layout()

    output_path = (
        Path(args.output)
        if args.output
        else image_path.with_name(f"{image_path.stem}_radial_fft.png")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Plot saved to {output_path}")

    if args.output_json:
        json_path = Path(args.output_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "image": str(image_path),
            "size": args.size,
            "num_bins": args.num_bins,
            "tail_start_requested": args.tail_start,
            "tail_bin": tail_bin,
            "tail_boundary_actual": tail_boundary,
            "radial_normalization": "center-to-corner",
            "bin_centers": bin_centers.tolist(),
            "variants": results,
        }
        with json_path.open("w", encoding="utf-8") as output_file:
            json.dump(payload, output_file, indent=2)
        print(f"Spectrum values saved to {json_path}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
