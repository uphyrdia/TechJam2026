"""Neural-network components for spatial/frequency AIGC classification.

The detector analyzes two tensors made from the same resized or augmented RGB
image. ``spatial_images`` must contain the ImageNet-normalized view expected by
ConvNeXt, whereas ``frequency_images`` must contain the corresponding raw
``[0, 1]`` RGB view. Keeping these views separate prevents ImageNet
normalization from changing the meaning of luminance in the FFT branch.

``Fusion.forward`` returns raw binary-classification logits, not probabilities.
Training code should pass them directly to ``BCEWithLogitsLoss``; inference code
can apply ``sigmoid`` to obtain the project's AI score. That score is not
necessarily a calibrated posterior probability.
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ConvNeXt_Small_Weights


class SpatialBranch(nn.Module):
    """Extract spatial features and a spatial-only logit with ConvNeXt-Small.

    Input tensors have shape ``[B, 3, H, W]`` and are expected to use ImageNet
    normalization. ``forward_features`` returns one ``feature_dim``-element
    vector per image; the separate classifier method allows the same vector to
    feed both the spatial-only auxiliary head and the fusion head.
    """

    def __init__(self, pretrained=True):
        super().__init__()
        # ``None`` avoids both downloading and loading ImageNet parameters when
        # constructing a model solely for checkpoint restoration or testing.
        weights = ConvNeXt_Small_Weights.DEFAULT if pretrained else None
        self.backbone = models.convnext_small(weights=weights)

        # ConvNeXt's classifier is normalization -> flatten -> linear. Replace
        # only its final multiclass layer with a one-logit binary head, while
        # retaining the flattened vector immediately before that head for
        # cross-branch fusion.
        self.feature_dim = self.backbone.classifier[-1].in_features
        self.backbone.classifier[-1] = nn.Linear(self.feature_dim, 1)

    def forward_features(self, images):
        """Return pooled ConvNeXt embeddings with shape ``[B, feature_dim]``."""

        # Feature maps: [B, feature_dim, h, w]. Adaptive pooling removes the
        # remaining spatial axes, after which the original normalization and
        # flattening layers produce [B, feature_dim].
        features = self.backbone.features(images)
        features = self.backbone.avgpool(features)
        features = self.backbone.classifier[0](features)
        return self.backbone.classifier[1](features)

    def classify_features(self, features):
        """Map ``[B, feature_dim]`` embeddings to raw logits of shape ``[B, 1]``."""

        return self.backbone.classifier[-1](features)

    def forward(self, images):
        """Return the spatial branch's raw binary-classification logits."""

        return self.classify_features(self.forward_features(images))


class RadiallyAveragedLogPowerSpectrum(nn.Module):
    """Radially average the complete luminance FFT log-power spectrum.

    Inputs have shape ``[B, 3, image_size, image_size]`` and are RGB tensors in
    ``[0, 1]``. Mean subtraction suppresses the DC component; a 2-D Hann window
    reduces spectral leakage caused by discontinuities at the square image
    boundary. Angular averaging then converts each 2-D log-power spectrum into
    ``num_bins`` radius-based values. This discards orientation information and
    gives the descriptor approximate in-plane rotation invariance.

    The complete radial descriptor is standardized per image. This is not a
    tail-only or STAL representation.
    """

    def __init__(self, num_bins=64, image_size=224, eps=1e-6):
        super().__init__()
        if num_bins < 2:
            raise ValueError("num_bins must be at least 2")
        if image_size < 2:
            raise ValueError("image_size must be at least 2")

        self.num_bins = num_bins
        self.image_size = image_size
        self.eps = eps

        window, bin_index, counts = self._build_geometry(
            image_size,
            image_size,
            num_bins,
        )
        # Geometry can be rebuilt from the architecture, so it need not be
        # stored in the checkpoint. It still follows the model onto the GPU.
        self.register_buffer("window", window, persistent=False)
        self.register_buffer("bin_index", bin_index, persistent=False)
        self.register_buffer("bin_counts", counts, persistent=False)

    @staticmethod
    def _build_geometry(height, width, num_bins):
        """Precompute the Hann window, pixel-to-radius map, and bin populations."""

        # ``periodic=False`` creates the symmetric window appropriate for a
        # finite image rather than the periodic form commonly used for frames.
        window_y = torch.hann_window(height, periodic=False, dtype=torch.float32)
        window_x = torch.hann_window(width, periodic=False, dtype=torch.float32)
        window = window_y[:, None] * window_x[None, :]

        # ``fftshift`` places zero frequency at index size // 2 on each axis.
        # These coordinates therefore measure Euclidean distance from the exact
        # shifted DC bin for both odd and even image dimensions.
        y = torch.arange(height, dtype=torch.float32) - (height // 2)
        x = torch.arange(width, dtype=torch.float32) - (width // 2)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        radius = torch.sqrt(yy.square() + xx.square())

        # Divide the full radial extent into uniformly spaced bins. Flattening
        # produces a lookup table compatible with the flattened spectrum used
        # by ``scatter_add_`` in ``forward``.
        radius = radius / radius.max().clamp_min(1e-6)
        bin_index = torch.clamp(
            (radius * num_bins).long(),
            min=0,
            max=num_bins - 1,
        ).reshape(1, -1)
        # Bin populations convert accumulated log power into an angular mean.
        counts = torch.bincount(bin_index[0], minlength=num_bins).float()
        return window, bin_index, counts

    def forward(self, images):
        """Return standardized radial descriptors with shape ``[B, num_bins]``."""

        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("frequency images must have shape [B, 3, H, W]")
        if images.shape[-2:] != (self.image_size, self.image_size):
            raise ValueError(
                "frequency images must have spatial shape "
                f"[{self.image_size}, {self.image_size}]"
            )

        # FFT support for float16 is device/shape dependent. Keep the entire
        # inexpensive spectrum calculation in float32, including under AMP.
        with torch.autocast(device_type=images.device.type, enabled=False):
            images = images.float()
            # Convert RGB to a single luminance plane using standard luma
            # coefficients, then remove each image's mean intensity so that
            # overall brightness does not dominate the lowest frequencies.
            luminance = (0.299 * images[:, 0] + 0.587 * images[:, 1] + 0.114 * images[:, 2])
            luminance -= luminance.mean(dim=(-2, -1), keepdim=True)

            # Orthonormal scaling applies symmetric 1/sqrt(H*W) normalization.
            # ``fftshift`` aligns the spectrum with the centered radial-bin
            # geometry built in ``_build_geometry``.
            spectrum = torch.fft.fft2(
                luminance * self.window,
                norm="ortho",
            )
            spectrum = torch.fft.fftshift(spectrum, dim=(-2, -1))
            # Power is |FFT|^2. The small additive constant keeps log(0) finite.
            log_power = torch.log(spectrum.abs().square() + 1e-12)

            batch_size = images.shape[0]
            expanded_bin_index = self.bin_index.expand(batch_size, -1)
            # Accumulate all frequency-plane pixels with the same radial-bin
            # index. This vectorizes radial averaging across the full batch.
            radial_sum = torch.zeros(
                batch_size,
                self.num_bins,
                device=images.device,
                dtype=log_power.dtype,
            )
            radial_sum.scatter_add_(
                1,
                expanded_bin_index,
                log_power.reshape(batch_size, -1),
            )
            radial = radial_sum / self.bin_counts.clamp_min(1).unsqueeze(0)

            # Per-image standardization emphasizes the shape of the spectrum
            # rather than its absolute level. ``eps`` also handles an otherwise
            # constant descriptor without producing NaNs.
            radial -= radial.mean(dim=1, keepdim=True)
            radial /= radial.std(dim=1, keepdim=True, unbiased=False).clamp_min(self.eps)
            return radial


class Fusion(nn.Module):
    """Fuse ConvNeXt spatial features with the full radial FFT descriptor.

    This is a conventional two-branch classifier, not STAL. Both branches are
    retained at inference. Spatial-only and frequency-only logits support
    auxiliary training losses and branch-level diagnostics; ``fused_logits``
    is the deployed prediction. Every ``*_logits`` tensor has shape ``[B, 1]``.
    """

    def __init__(
        self,
        pretrained=True,
        image_size=224,
        num_frequency_bins=64,
        frequency_embedding_dim=128,
        fusion_hidden_dim=256,
        dropout=0.2,
    ):
        super().__init__()
        self.image_size = image_size
        self.num_frequency_bins = num_frequency_bins
        self.frequency_embedding_dim = frequency_embedding_dim
        self.fusion_hidden_dim = fusion_hidden_dim

        self.spatial_detector = SpatialBranch(pretrained=pretrained)
        self.spectrum = RadiallyAveragedLogPowerSpectrum(
            num_bins=num_frequency_bins,
            image_size=image_size,
        )
        # Encode the compact [B, num_frequency_bins] descriptor into a learned
        # [B, frequency_embedding_dim] representation before fusion.
        self.frequency_encoder = nn.Sequential(
            nn.LayerNorm(num_frequency_bins),
            nn.Linear(num_frequency_bins, frequency_embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(frequency_embedding_dim, frequency_embedding_dim),
            nn.GELU(),
        )
        # This auxiliary head exposes what the frequency branch predicts alone.
        self.frequency_classifier = nn.Linear(frequency_embedding_dim, 1)

        fused_dim = self.spatial_detector.feature_dim + frequency_embedding_dim
        # Concatenation preserves both modalities; the following MLP learns how
        # much to rely on each one for the final decision.
        self.fusion_classifier = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, 1),
        )

    def forward(self, spatial_images, frequency_images):
        """Run both aligned image views and return logits plus intermediate features.

        ``spatial_images`` is the ImageNet-normalized view. ``frequency_images``
        is the matching unnormalized ``[0, 1]`` view and must have the configured
        square ``image_size``. The returned feature tensors are intentionally
        exposed for diagnostics and representation analysis.
        """

        # Shapes are [B, spatial_feature_dim], [B, num_frequency_bins], and
        # [B, frequency_embedding_dim], respectively.
        spatial_features = self.spatial_detector.forward_features(spatial_images)
        radial_spectrum = self.spectrum(frequency_images)
        frequency_features = self.frequency_encoder(radial_spectrum)
        # Concatenate along the feature axis; batch entries remain aligned
        # because both inputs must originate from the same transformed pixels.
        fused_features = torch.cat(
            (spatial_features, frequency_features),
            dim=1,
        )

        # Do not apply sigmoid here. Raw logits are numerically stable with
        # BCEWithLogitsLoss and let callers choose their own decision threshold.
        return {
            "fused_logits": self.fusion_classifier(fused_features),
            "spatial_logits": self.spatial_detector.classify_features(
                spatial_features
            ),
            "frequency_logits": self.frequency_classifier(frequency_features),
            "spatial_features": spatial_features,
            "frequency_features": frequency_features,
            "radial_spectrum": radial_spectrum,
        }


if __name__ == "__main__":
    # Lightweight architecture sanity check; no image or checkpoint is needed.
    detector = Fusion(pretrained=False)
    parameters = sum(parameter.numel() for parameter in detector.parameters())
    print(f"Total parameters: {parameters / 1e6:.2f}M")
