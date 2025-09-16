import os
import torch
import torch.nn as nn
from torchvision import models

# --------------------------
# Model Class
# --------------------------
class MyCnn(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.model = models.efficientnet_v2_m(weights="IMAGENET1K_V1")

        # Freeze all layers
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze deeper layers
        for name, param in self.model.named_parameters():
            if "features.6" in name or "features.7" in name:
                param.requires_grad = True

        # Replace classifier head
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.model(x)


# --------------------------
# Load Model
# --------------------------
def load_model(
    checkpoint_path="best_model.pth", num_classes=7, device=None
):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MyCnn(num_classes=num_classes).to(device)

    def strip_module_prefix(state_dict):
        return {
            (k.replace("module.", "") if k.startswith("module.") else k): v
            for k, v in state_dict.items()
        }

    ckpt = torch.load(checkpoint_path, map_location=device)

    if isinstance(ckpt, dict) and (
        "state_dict" in ckpt or "model_state_dict" in ckpt
    ):
        sd = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))
    else:
        sd = ckpt

    if any(k.startswith("module.") for k in sd.keys()):
        sd = strip_module_prefix(sd)

    load_res = model.load_state_dict(sd, strict=False)
    print("Load result ->", load_res)

    model.eval()
    return model, device

