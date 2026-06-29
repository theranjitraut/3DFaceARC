"""
models/encoder.py
CNN Encoder: ResNet50 backbone → 3DMM coefficient regressor.
Predicts shape, expression, texture, pose, and scale coefficients
from a single RGB image.  No ground-truth annotations needed —
trained entirely with self-supervised losses.
"""

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models
import timm

# Coefficient Regressor Head
class CoeffRegressor(nn.Module):
    """
    MLP head that maps CNN features → 3DMM coefficient vector.
    """
    def __init__(self, in_dim: int, coeff_dims: dict):
        super().__init__()
        total = sum(coeff_dims.values())

        self.fc = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ELU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ELU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, total),
        )
        self.coeff_dims = coeff_dims
        self._init_weights()

    def _init_weights(self):
        for m in self.fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, feat: torch.Tensor) -> dict:
        raw = self.fc(feat)
        idx = 0
        out = {}
        for name, dim in self.coeff_dims.items():
            out[name] = raw[:, idx: idx + dim]
            idx += dim

        # Apply activation constraints
        out['scale']       = torch.sigmoid(out['scale']) * 2.0 + 0.5  # [0.5, 2.5]
        out['translation'] = out['translation'] * 0.1                  # small shifts
        out['angles']      = out['angles'] * 0.3                       # radians

        return out

# Full CNN Encoder
class FaceEncoder(nn.Module):
    """
    Wraps a pretrained ResNet/EfficientNet backbone with a coefficient
    regressor head. The backbone is fine-tuned end-to-end.

    Args:
        backbone_name : 'resnet50' | 'resnet34' | 'efficientnet_b3'
        pretrained    : load ImageNet weights
        coeff_dims    : dict of {coeff_name: dim}
    """

    def __init__(self, backbone_name: str = 'resnet50',
                 pretrained: bool = True, coeff_dims: dict = None):
        super().__init__()
        if coeff_dims is None:
            coeff_dims = {'shape': 80, 'exp': 64, 'tex': 80,
                          'angles': 3, 'translation': 3, 'scale': 1}

        #  backbone 
        if backbone_name == 'resnet50':
            base = tv_models.resnet50(
                weights=tv_models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
            feat_dim = base.fc.in_features
            base.fc = nn.Identity()
            self.backbone = base

        elif backbone_name == 'resnet34':
            base = tv_models.resnet34(
                weights=tv_models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
            feat_dim = base.fc.in_features
            base.fc = nn.Identity()
            self.backbone = base

        elif backbone_name.startswith('efficientnet'):
            base = timm.create_model(backbone_name, pretrained=pretrained,
                                     num_classes=0, global_pool='avg')
            feat_dim = base.num_features
            self.backbone = base

        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        #  regressor head
        self.regressor = CoeffRegressor(feat_dim, coeff_dims)
        self.feat_dim  = feat_dim

    def forward(self, x: torch.Tensor):
        """
        x : (B, 3, H, W)  normalised image tensor
        returns dict of coefficient tensors
        """
        feat   = self.backbone(x)       # (B, feat_dim)
        coeffs = self.regressor(feat)   # dict
        return coeffs, feat

# ArcFace Identity Encoder  (for identity-preservation loss)
class ArcFaceEncoder(nn.Module):
    """
    Frozen identity embedding extractor for the identity-preservation loss.
    Supports two weight formats:
      - ONNX  (.onnx) : loaded via onnxruntime (e.g. InsightFace model.onnx)
      - PyTorch (.pth) : standard state-dict loaded into a ResNet50 backbone

    Input images are automatically resized to the model's expected resolution.
    Output: (B, 512) L2-normalised identity embedding.
    """

    def __init__(self, weights_path: str = None):
        super().__init__()
        from pathlib import Path
        self._ort_session = None   # ONNX runtime session
        self._input_size  = 224   # default; overridden by ONNX model

        path = Path(weights_path) if weights_path else None

        if path and path.exists():
            if path.suffix.lower() == '.onnx':
                # ONNX path 
                try:
                    import onnxruntime as ort
                    providers = (['CUDAExecutionProvider', 'CPUExecutionProvider']
                                 if self._cuda_available() else ['CPUExecutionProvider'])
                    self._ort_session = ort.InferenceSession(str(path),
                                                             providers=providers)
                    # Read expected input size from the model
                    inp = self._ort_session.get_inputs()[0]
                    self._input_name = inp.name
                    shape = inp.shape          # e.g. [None, 3, 112, 112]
                    self._input_size = int(shape[2]) if isinstance(shape[2], int) and shape[2] > 0 else 112
                    # Dummy backbone (unused, just keeps nn.Module happy)
                    self.net = nn.Identity()
                    print(f"[ArcFace] Loaded ONNX model from {path} "
                          f"(input={self._input_size}x{self._input_size})")
                except Exception as e:
                    print(f"[ArcFace] Failed to load ONNX: {e}. Falling back to random init.")
                    self._ort_session = None
                    self._build_pytorch_backbone(None)
            else:
                # PyTorch .pth path 
                self._build_pytorch_backbone(str(path))
        else:
            if weights_path:
                print(f"[ArcFace] Weights not found at '{weights_path}'. "
                      f"Using random init - identity loss will be noisy initially.")
            self._build_pytorch_backbone(None)

    @staticmethod
    def _cuda_available():
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False

    def _build_pytorch_backbone(self, weights_path):
        """Build a ResNet50 backbone and optionally load .pth weights."""
        import torch
        self.net = tv_models.resnet50(weights=None)
        self.net.fc = nn.Linear(self.net.fc.in_features, 512)
        if weights_path:
            state = torch.load(weights_path, map_location='cpu')
            state = {k.replace('module.', ''): v for k, v in state.items()}
            self.net.load_state_dict(state, strict=False)
            print(f"[ArcFace] Loaded PyTorch weights from {weights_path}")
        self.net.eval()
        for p in self.net.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, 3, H, W)  — any spatial size, normalised to ImageNet stats
        returns: (B, 512)  L2-normalised embedding
        """
        import torch.nn.functional as F

        if self._ort_session is not None:
            # Resize to the ONNX model's expected input size
            if x.shape[-1] != self._input_size or x.shape[-2] != self._input_size:
                x_in = F.interpolate(x, size=(self._input_size, self._input_size),
                                     mode='bilinear', align_corners=False)
            else:
                x_in = x
            # Run ONNX inference one image at a time (model exported with
            # fixed batch_size=1; sending a larger batch triggers a shape
            # mismatch warning from onnxruntime every forward pass).
            x_np = x_in.detach().cpu().numpy()
            embs = [
                self._ort_session.run(None, {self._input_name: x_np[i : i + 1]})[0]
                for i in range(x_np.shape[0])
            ]
            emb = torch.from_numpy(np.concatenate(embs, axis=0)).to(x.device)
        else:
            # PyTorch backbone path
            if x.shape[-1] != 224 or x.shape[-2] != 224:
                x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
            emb = self.net(x)

        return nn.functional.normalize(emb, dim=1)

