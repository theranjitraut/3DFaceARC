"""
utils/losses.py
Self-supervised loss functions for FaceARCs.
No ground-truth 3D annotations required — all losses are derived
from the input image itself (photometric, perceptual, symmetry, etc.)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models

# VGG Perceptual Loss
class VGGPerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG16 intermediate feature maps.
    Compares rendered image vs input image in feature space.
    """

    def __init__(self, layers=(3, 8, 15, 22)):
        super().__init__()
        vgg = tv_models.vgg16(weights=tv_models.VGG16_Weights.IMAGENET1K_V1)
        self.slices = nn.ModuleList([
            nn.Sequential(*list(vgg.features.children())[:l]) for l in layers
        ])
        for p in self.parameters():
            p.requires_grad_(False)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std',  torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def _normalise(self, x):
        return (x - self.mean) / self.std

    def forward(self, rendered: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        rendered: (B,3,H,W) in [0,1]
        target  : (B,3,H,W) in [0,1]
        """
        r = self._normalise(rendered.clamp(0, 1))
        t = self._normalise(target.clamp(0, 1))
        loss = torch.tensor(0.0, device=rendered.device)
        for sl in self.slices:
            r_feat = sl(r)
            t_feat = sl(t)
            loss += F.l1_loss(r_feat, t_feat)
        return loss / len(self.slices)


# Photometric Loss
def photometric_loss(rendered: torch.Tensor, target: torch.Tensor,
                     mask: torch.Tensor = None) -> torch.Tensor:
    """
    Pixel-level L1 loss between rendered and target images.
    Optionally masked to the face region (silhouette).
    rendered: (B,3,H,W), target: (B,3,H,W) both in [0,1]
    mask    : (B,1,H,W) binary silhouette mask
    """
    diff = (rendered - target).abs()
    if mask is not None:
        mask = mask.expand_as(diff)
        diff = diff * mask
        denom = mask.sum().clamp(min=1)
        return diff.sum() / denom
    return diff.mean()


# Landmark Loss  (pseudo-landmarks from MediaPipe)
def landmark_loss(pred_verts: torch.Tensor, faces: torch.Tensor,
                  target_lmks: torch.Tensor,
                  lmk_vertex_ids: torch.Tensor = None) -> torch.Tensor:
    """
    Compares predicted 3D landmark vertices (projected to 2D) against
    pseudo-landmarks from MediaPipe.

    pred_verts   : (B, V, 3)
    target_lmks  : (B, 68, 2) normalised [-1,1]
    lmk_vertex_ids: (68,) vertex indices that correspond to 68 landmarks
                    If None, use nearest-vertex approximation.
    """
    B = pred_verts.shape[0]

    if lmk_vertex_ids is not None:
        ids = lmk_vertex_ids.to(pred_verts.device)
        pred_lmks = pred_verts[:, ids, :2]          # (B, 68, 2) — orthographic xy
    else:
        # Approximate: pick 68 evenly-spaced vertices
        V = pred_verts.shape[1]
        step = max(1, V // 68)
        ids = torch.arange(0, 68 * step, step, device=pred_verts.device)[:68]
        pred_lmks = pred_verts[:, ids, :2]

    # Normalise predicted landmarks to [-1,1]
    max_val = pred_lmks.abs().max(dim=1, keepdim=True).values.max(dim=-1, keepdim=True).values
    pred_lmks_norm = pred_lmks / (max_val + 1e-8)

    # Skip samples where MediaPipe returned all-zeros (no face detected)
    valid = (target_lmks.abs().sum(dim=[1, 2]) > 0.1)
    if valid.sum() == 0:
        return torch.tensor(0.0, device=pred_verts.device)

    return F.mse_loss(pred_lmks_norm[valid], target_lmks[valid])

# 3DMM Regularisation Losses
def coeff_regularisation(coeffs: dict, weights: dict = None) -> torch.Tensor:
    """
    L2 regularisation on 3DMM coefficients to keep reconstructions
    close to the statistical model prior (prevents unrealistic faces).
    """
    if weights is None:
        weights = {'shape': 1.0, 'exp': 0.8, 'tex': 0.5}
    loss = torch.tensor(0.0, device=next(iter(coeffs.values())).device)
    for key, w in weights.items():
        if key in coeffs:
            loss += w * (coeffs[key] ** 2).mean()
    return loss

# Mesh Smoothness (Laplacian) Loss
def laplacian_smoothness_loss(verts: torch.Tensor,
                               faces: torch.Tensor) -> torch.Tensor:
    """
    Uniform Laplacian smoothness loss: penalises vertices that deviate
    significantly from the mean of their neighbours.
    verts: (B, V, 3)
    faces: (F, 3)
    """
    B, V, _ = verts.shape
    device   = verts.device

    # Build neighbour sums
    # Each face [a,b,c] contributes: a↔b, b↔c, a↔c
    a, b, c = faces[:, 0], faces[:, 1], faces[:, 2]
    idx_i = torch.cat([a, b, a, c, b, c])
    idx_j = torch.cat([b, a, c, a, c, b])

    deg = torch.zeros(V, device=device)
    deg.scatter_add_(0, idx_i, torch.ones(idx_i.shape[0], device=device))
    deg = deg.clamp(min=1)

    total = torch.tensor(0.0, device=device)
    for batch in range(B):
        v = verts[batch]                                    # (V,3)
        nbr_sum = torch.zeros_like(v)
        nbr_sum.scatter_add_(0, idx_j.unsqueeze(1).expand(-1, 3),
                             v[idx_i])
        laplacian = v - nbr_sum / deg.unsqueeze(1)         # (V,3)
        total += (laplacian ** 2).mean()
    return total / B


# Face Symmetry Loss
def symmetry_loss(rendered: torch.Tensor) -> torch.Tensor:
    """
    Encourages the rendered face to be approximately symmetric
    by comparing the image with its horizontal flip.
    rendered: (B,3,H,W)
    """
    flipped = rendered.flip(dims=[3])
    return F.l1_loss(rendered, flipped)

# Identity Preservation Loss
def identity_loss(embed_rendered: torch.Tensor,
                  embed_input: torch.Tensor) -> torch.Tensor:
    """
    Cosine similarity loss: rendered face should have same identity
    as input face in ArcFace embedding space.
    """
    cos_sim = F.cosine_similarity(embed_rendered, embed_input, dim=1)
    return (1.0 - cos_sim).mean()

# Combined Self-Supervised Loss
class FaceARCsLoss(nn.Module):
    """
    Combines all self-supervised losses into one module.
    Weights are controlled by config.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.w = cfg['training']['loss_weights']
        self.perceptual_net = VGGPerceptualLoss()

    def forward(self, model_out: dict, batch: dict) -> dict:
        device   = model_out['rendered_img'].device
        target   = batch['image_raw'].to(device)         # (B,3,H,W) [0,1]

        rendered   = model_out['rendered_img'].clamp(0, 1)
        silhouette = model_out['silhouette']
        coeffs     = model_out['coeffs']
        verts      = model_out['refined_verts']
        faces      = model_out['faces']
        lmks       = batch['landmarks'].to(device)       # (B,68,2)

        losses = {}

        # 1. Photometric
        losses['photometric'] = photometric_loss(rendered, target, silhouette) \
                                * self.w['photometric']

        # 2. Perceptual
        losses['perceptual'] = self.perceptual_net(rendered, target) \
                               * self.w['perceptual']

        # 3. Landmark (pseudo)
        losses['landmark'] = landmark_loss(verts, faces, lmks) \
                             * self.w['landmark']

        # 4. 3DMM regularisation
        losses['shape_reg'] = coeff_regularisation(coeffs) \
                              * self.w['shape_reg']

        # 5. Mesh smoothness
        losses['smooth'] = laplacian_smoothness_loss(verts, faces) \
                           * self.w['smooth']

        # 6. Symmetry
        losses['symmetry'] = symmetry_loss(rendered) \
                             * self.w['symmetry']

        # 7. Identity
        losses['id_preserve'] = identity_loss(
            model_out['id_embed_rendered'],
            model_out['id_embed_input']
        ) * self.w['id_preserve']

        losses['total'] = sum(losses.values())
        return losses
