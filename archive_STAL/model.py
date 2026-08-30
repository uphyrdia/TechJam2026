import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ConvNeXt_Small_Weights


class AIGCClassifier(nn.Module):
    """Spatial binary classifier based on ConvNeXt-Small.

    This class deliberately keeps the same state-dict key layout as the
    original implementation, so old spatial checkpoints remain loadable.
    """

    def __init__(self, pretrained=True):
        super().__init__()
        weights = ConvNeXt_Small_Weights.DEFAULT if pretrained else None
        self.backbone = models.convnext_small(weights=weights)
        self.feature_dim = self.backbone.classifier[-1].in_features
        self.backbone.classifier[-1] = nn.Linear(self.feature_dim, 1)

    def forward_features(self, x):
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = self.backbone.classifier[0](x)
        x = self.backbone.classifier[1](x)
        return x

    def classify_features(self, features):
        return self.backbone.classifier[-1](features)

    def forward(self, x):
        return self.classify_features(self.forward_features(x))


class RadialLogPowerSpectrum(nn.Module):
    """Convert RGB tensors in [0, 1] to normalized radial log-power spectra."""

    def __init__(self, num_bins=64, eps=1e-6):
        super().__init__()
        self.num_bins = num_bins
        self.eps = eps

    def forward(self, images):
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("frequency images must have shape [B, 3, H, W]")

        # FFT at 224x224 is not guaranteed to support float16 on every GPU, so
        # perform this small branch in float32 even when AMP is enabled.
        images = images.float()
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
            device=images.device,
            dtype=images.dtype,
        )
        window_x = torch.hann_window(
            width,
            periodic=False,
            device=images.device,
            dtype=images.dtype,
        )
        window = window_y[:, None] * window_x[None, :]

        spectrum = torch.fft.fft2(luminance * window, norm="ortho")
        spectrum = torch.fft.fftshift(spectrum, dim=(-2, -1))
        # STAL uses the shape of the radial log-power spectrum. log(power),
        # rather than log(1 + power), preserves differences in weak tail bins.
        log_power = torch.log(spectrum.abs().square() + 1e-12)

        y = torch.arange(height, device=images.device, dtype=images.dtype)
        x = torch.arange(width, device=images.device, dtype=images.dtype)
        y = y - (height - 1) / 2
        x = x - (width - 1) / 2
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        radius = torch.sqrt(yy.square() + xx.square())
        radius = radius / radius.max().clamp_min(self.eps)
        bin_index = torch.clamp(
            (radius * self.num_bins).long(),
            min=0,
            max=self.num_bins - 1,
        ).reshape(1, -1)

        batch_size = images.shape[0]
        bin_index = bin_index.expand(batch_size, -1)
        radial_sum = torch.zeros(
            batch_size,
            self.num_bins,
            device=images.device,
            dtype=log_power.dtype,
        )
        radial_sum.scatter_add_(1, bin_index, log_power.reshape(batch_size, -1))

        counts = torch.bincount(
            bin_index[0],
            minlength=self.num_bins,
        ).to(log_power.dtype)
        radial = radial_sum / counts.clamp_min(1).unsqueeze(0)

        # Preserve the spectrum's shape while suppressing global exposure and
        # contrast as shortcuts.
        radial = radial - radial.mean(dim=1, keepdim=True)
        radial = radial / radial.std(dim=1, keepdim=True).clamp_min(self.eps)
        return radial


class FrequencyTeacher(nn.Module):
    """Lightweight frequency context and spectral-tail teacher."""

    def __init__(self, num_bins=64, embedding_dim=128, tail_start=0.70, beta=0.5):
        super().__init__()
        self.spectrum = RadialLogPowerSpectrum(num_bins=num_bins)
        self.tail_bin = int(round(num_bins * tail_start))
        self.beta = beta

        self.context_encoder = nn.Sequential(
            nn.LayerNorm(num_bins),
            nn.Linear(num_bins, embedding_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.tail_encoder = nn.Sequential(
            nn.LayerNorm(num_bins - self.tail_bin),
            nn.Linear(num_bins - self.tail_bin, embedding_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(embedding_dim, embedding_dim),
        )
        # The teacher target is stop-gradient in the alignment loss, so a fixed
        # (non-affine) normalization avoids unused trainable scale/bias terms.
        self.teacher_norm = nn.LayerNorm(embedding_dim, elementwise_affine=False)
        self.frequency_classifier = nn.Linear(embedding_dim, 1)
        self.tail_classifier = nn.Linear(embedding_dim, 1)

    def forward(self, images):
        radial_spectrum = self.spectrum(images)
        context = self.context_encoder(radial_spectrum)
        tail_embedding = self.tail_encoder(radial_spectrum[:, self.tail_bin :])
        teacher_target = self.teacher_norm(context + self.beta * tail_embedding)
        return {
            "radial_spectrum": radial_spectrum,
            "frequency_features": context,
            "tail_features": tail_embedding,
            "teacher_target": teacher_target,
            "frequency_logits": self.frequency_classifier(context),
            "tail_logits": self.tail_classifier(tail_embedding),
        }


class STALTrainingModel(nn.Module):
    """Lightweight STAL-style model used only while training.

    This implements radial spectral context, an explicit high-frequency tail
    head, and frequency-to-spatial representation alignment. The local-DCT and
    supervised-contrastive components from the full research system are
    intentionally omitted to keep the implementation small and understandable.
    Only ``spatial_detector`` is needed for deployment.
    """

    def __init__(
        self,
        pretrained=True,
        num_frequency_bins=64,
        frequency_embedding_dim=128,
    ):
        super().__init__()
        self.spatial_detector = AIGCClassifier(pretrained=pretrained)
        self.frequency_teacher = FrequencyTeacher(
            num_bins=num_frequency_bins,
            embedding_dim=frequency_embedding_dim,
        )
        self.spatial_projector = nn.Sequential(
            nn.Linear(self.spatial_detector.feature_dim, frequency_embedding_dim),
            nn.GELU(),
            nn.Linear(frequency_embedding_dim, frequency_embedding_dim),
        )

    def forward(self, spatial_images, frequency_images):
        spatial_features = self.spatial_detector.forward_features(spatial_images)
        outputs = self.frequency_teacher(frequency_images)
        outputs.update(
            {
                "spatial_features": spatial_features,
                "spatial_projection": self.spatial_projector(spatial_features),
                "spatial_logits": self.spatial_detector.classify_features(
                    spatial_features
                ),
            }
        )
        return outputs

    def spatial_state_dict(self):
        """Return an inference-compatible AIGCClassifier state dict."""
        return self.spatial_detector.state_dict()


def class_balanced_alignment_loss(spatial_projection, teacher_target, labels):
    """Cosine alignment averaged separately for each class in the batch."""
    per_sample = 1 - F.cosine_similarity(
        spatial_projection,
        teacher_target.detach(),
        dim=1,
    )
    labels = labels.reshape(-1).long()
    class_losses = []
    for label in (0, 1):
        mask = labels == label
        if mask.any():
            class_losses.append(per_sample[mask].mean())
    if not class_losses:
        return per_sample.mean()
    return torch.stack(class_losses).mean()


if __name__ == "__main__":
    model = STALTrainingModel(pretrained=False)
    total_parameters = sum(parameter.numel() for parameter in model.parameters()) / 1e6
    inference_parameters = sum(
        parameter.numel() for parameter in model.spatial_detector.parameters()
    ) / 1e6
    print(f"Training parameters: {total_parameters:.2f}M")
    print(f"Inference parameters: {inference_parameters:.2f}M")
