import torch
import torch.nn as nn
import torchvision.models as models

class MyCnn(nn.Module):
    def __init__(self, num_classes=7):
        super(MyCnn, self).__init__()
        self.model = models.resnet18(weights=None)   # 🔥 ResNet18 instead of ResNet50

        # Freeze early layers
        for param in self.model.parameters():
            param.requires_grad = False

        for name, param in self.model.named_parameters():
            if "layer2" in name or "layer3" in name or "layer4" in name:
                param.requires_grad = True

        # Replace classifier
        in_features = self.model.fc.in_features   # ✅ should be 512 now
        self.model.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.model(x)
