import torch.nn as nn
from torchvision import models
from torchvision.models import ConvNeXt_Small_Weights


class AIGCClassifier(nn.Module):
    """Spatial-only ConvNeXt-Small binary classifier.

    The module and parameter names intentionally match the original spatial
    model, so existing raw spatial state dictionaries remain compatible.
    Output values are logits; apply sigmoid to obtain P(AI-generated).
    """

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


if __name__ == "__main__":
    model = AIGCClassifier(pretrained=False)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Parameters: {parameter_count / 1e6:.2f}M")
