"""MobileNetV3-Small with explicit preprocessing in the model graph."""
import torch
from torch import nn
from torchvision.models import mobilenet_v3_small


class SignClassifier(nn.Module):
    def __init__(self, num_classes=20):
        super().__init__()
        self.net = mobilenet_v3_small(weights=None, num_classes=num_classes)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])

    def forward(self, image):
        # Same normalization for pretraining, frozen supervision and inference.
        return self.net((image-self.mean)/self.std)


def load_classifier(path, device='cpu', frozen=True):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    if ckpt.get('kind') != 'classifier':
        raise ValueError('Not a classifier checkpoint')
    model = SignClassifier(ckpt['num_classes']).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    if frozen:
        model.requires_grad_(False)
    return model
