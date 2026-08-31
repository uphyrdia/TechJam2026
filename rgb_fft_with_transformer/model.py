import math

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ConvNeXt_Small_Weights


class AIGCClassifier(nn.Module):
    """Unchanged spatial ConvNeXt-Small branch."""

    def __init__(self, pretrained=True):
        super().__init__()
        weights = ConvNeXt_Small_Weights.DEFAULT if pretrained else None
        self.backbone = models.convnext_small(weights=weights)
        self.feature_dim = self.backbone.classifier[-1].in_features
        self.backbone.classifier[-1] = nn.Linear(self.feature_dim, 1)

    def forward_features(self, images):
        features = self.backbone.features(images)
        features = self.backbone.avgpool(features)
        features = self.backbone.classifier[0](features)
        return self.backbone.classifier[1](features)

    def classify_features(self, features):
        return self.backbone.classifier[-1](features)

    def forward(self, images):
        return self.classify_features(self.forward_features(images))


class Full2DLogPowerSpectrum(nn.Module):
    """Compute the complete centered RGB log-power spectrum.

    The output has shape [B, 3, H, W]. No radial averaging, tail selection, or
    hand-crafted uplift statistic is used. Each RGB channel is standardized
    independently within each image. This retains the two-dimensional pattern
    while reducing sensitivity to global spectral scale.
    """

    def __init__(self, image_size=224, eps=1e-6):
        super().__init__()
        if image_size < 2:
            raise ValueError("image_size must be at least 2")

        self.image_size = image_size
        self.eps = eps

        window_y = torch.hann_window(image_size, periodic=False)
        window_x = torch.hann_window(image_size, periodic=False)
        window = window_y[:, None] * window_x[None, :]
        self.register_buffer(
            "hann_window",
            window.reshape(1, 1, image_size, image_size),
            persistent=False,
        )

    def forward(self, images):
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("frequency images must have shape [B, 3, H, W]")
        if images.shape[-2:] != (self.image_size, self.image_size):
            raise ValueError(
                "frequency images must have spatial size "
                f"{self.image_size}x{self.image_size}; received "
                f"{tuple(images.shape[-2:])}"
            )

        # FFT at 224x224 is not reliably supported in float16. The outer
        # training loop may use AMP, but this representation stays float32.
        images = images.float()
        centered = images - images.mean(dim=(-2, -1), keepdim=True)
        windowed = centered * self.hann_window.to(
            device=images.device,
            dtype=images.dtype,
        )

        spectrum = torch.fft.fft2(windowed, norm="ortho")
        spectrum = torch.fft.fftshift(spectrum, dim=(-2, -1))
        log_power = torch.log1p(spectrum.abs().square())

        mean = log_power.mean(dim=(-2, -1), keepdim=True)
        std = log_power.std(
            dim=(-2, -1),
            keepdim=True,
            unbiased=False,
        ).clamp_min(self.eps)
        return (log_power - mean) / std


class Full2DSpectralTransformer(nn.Module):
    """Encode every 2-D spectral patch without fixed global averaging.

    A strided convolution converts non-overlapping spectral patches into a
    sequence. A learned CLS token aggregates information through attention.
    The two-dimensional patch positions are retained by learned positional
    embeddings.
    """

    def __init__(
        self,
        image_size=224,
        patch_size=16,
        embedding_dim=256,
        depth=4,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.1,
    ):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")
        if depth < 1:
            raise ValueError("depth must be at least 1")

        self.image_size = image_size
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        patches_per_side = image_size // patch_size
        self.num_patches = patches_per_side**2

        self.patch_embedding = nn.Conv2d(
            3,
            embedding_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.class_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embedding_dim)
        )
        self.embedding_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=round(embedding_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth,
            enable_nested_tensor=False,
        )
        self.output_norm = nn.LayerNorm(embedding_dim)

        nn.init.trunc_normal_(self.class_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        nn.init.trunc_normal_(self.patch_embedding.weight, std=0.02)
        if self.patch_embedding.bias is not None:
            nn.init.zeros_(self.patch_embedding.bias)

    def forward(self, spectrum):
        patches = self.patch_embedding(spectrum)
        tokens = patches.flatten(2).transpose(1, 2)

        class_token = self.class_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat((class_token, tokens), dim=1)
        tokens = tokens + self.position_embedding
        tokens = self.embedding_dropout(tokens)
        tokens = self.transformer(tokens)
        return self.output_norm(tokens[:, 0])


class Full2DFrequencyFusionClassifier(nn.Module):
    """Fuse ConvNeXt spatial evidence with the complete 2-D FFT spectrum.

    The fused prediction is a gated residual correction to the spatial logit:

        fused_logit = spatial_logit + sigmoid(gate_logit) * fusion_delta

    Initializing the gate near 0.1 keeps early training close to the pretrained
    spatial detector. The model can learn to increase or suppress the spectral
    contribution. The frequency-only logit remains an auxiliary diagnostic.
    """

    def __init__(
        self,
        pretrained=True,
        image_size=224,
        patch_size=16,
        frequency_embedding_dim=256,
        frequency_depth=4,
        frequency_num_heads=8,
        frequency_mlp_ratio=4.0,
        fusion_hidden_dim=256,
        dropout=0.1,
        initial_fusion_gate=0.1,
    ):
        super().__init__()
        if not 0.0 < initial_fusion_gate < 1.0:
            raise ValueError("initial_fusion_gate must lie strictly in (0, 1)")

        self.image_size = image_size
        self.patch_size = patch_size
        self.frequency_embedding_dim = frequency_embedding_dim
        self.frequency_depth = frequency_depth
        self.frequency_num_heads = frequency_num_heads
        self.frequency_mlp_ratio = frequency_mlp_ratio
        self.fusion_hidden_dim = fusion_hidden_dim
        self.dropout = dropout

        self.spatial_detector = AIGCClassifier(pretrained=pretrained)
        self.spectrum = Full2DLogPowerSpectrum(image_size=image_size)
        self.frequency_encoder = Full2DSpectralTransformer(
            image_size=image_size,
            patch_size=patch_size,
            embedding_dim=frequency_embedding_dim,
            depth=frequency_depth,
            num_heads=frequency_num_heads,
            mlp_ratio=frequency_mlp_ratio,
            dropout=dropout,
        )
        self.frequency_classifier = nn.Linear(frequency_embedding_dim, 1)

        fused_dim = self.spatial_detector.feature_dim + frequency_embedding_dim
        self.fusion_delta = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, 1),
        )
        # Begin exactly at the spatial prediction. The fusion residual learns
        # from zero instead of perturbing a pretrained detector randomly.
        nn.init.zeros_(self.fusion_delta[-1].weight)
        nn.init.zeros_(self.fusion_delta[-1].bias)

        initial_gate_logit = math.log(
            initial_fusion_gate / (1.0 - initial_fusion_gate)
        )
        self.fusion_gate_logit = nn.Parameter(
            torch.tensor(initial_gate_logit, dtype=torch.float32)
        )

    def model_config(self):
        return {
            "image_size": self.image_size,
            "patch_size": self.patch_size,
            "frequency_embedding_dim": self.frequency_embedding_dim,
            "frequency_depth": self.frequency_depth,
            "frequency_num_heads": self.frequency_num_heads,
            "frequency_mlp_ratio": self.frequency_mlp_ratio,
            "fusion_hidden_dim": self.fusion_hidden_dim,
            "dropout": self.dropout,
        }

    def forward(self, spatial_images, frequency_images):
        """Return fused and diagnostic logits.

        spatial_images must be ImageNet-normalized. frequency_images must be
        the exact same pixels before normalization, with values in [0, 1].
        """
        spatial_features = self.spatial_detector.forward_features(spatial_images)
        spatial_logits = self.spatial_detector.classify_features(spatial_features)

        # Explicitly disable autocast around FFT. The Transformer that follows
        # still uses the caller's AMP context.
        with torch.autocast(
            device_type=frequency_images.device.type,
            enabled=False,
        ):
            full_2d_spectrum = self.spectrum(frequency_images)

        frequency_features = self.frequency_encoder(full_2d_spectrum)
        frequency_logits = self.frequency_classifier(frequency_features)

        fused_features = torch.cat(
            (spatial_features, frequency_features),
            dim=1,
        )
        fusion_delta = self.fusion_delta(fused_features)
        fusion_gate = torch.sigmoid(self.fusion_gate_logit)
        fused_logits = spatial_logits + fusion_gate * fusion_delta

        return {
            "fused_logits": fused_logits,
            "spatial_logits": spatial_logits,
            "frequency_logits": frequency_logits,
            "spatial_features": spatial_features,
            "frequency_features": frequency_features,
            "fusion_delta": fusion_delta,
            "fusion_gate": fusion_gate,
        }


if __name__ == "__main__":
    model = Full2DFrequencyFusionClassifier(pretrained=False)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    frequency_parameters = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if name.startswith(("frequency_encoder", "frequency_classifier"))
    )
    print(f"Total parameters: {total_parameters / 1e6:.2f}M")
    print(f"Frequency-branch parameters: {frequency_parameters / 1e6:.2f}M")
    print(
        "Initial fusion gate: "
        f"{torch.sigmoid(model.fusion_gate_logit).item():.4f}"
    )
