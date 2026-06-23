"""
utils/metrics.py
Evaluation metrics for 3D face reconstruction.
All metrics are computed without ground-truth 3D annotations.
"""

import torch
import torch.nn.functional as F
import numpy as np


def ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """
    Structural Similarity Index (SSIM) between two image batches.
    img1, img2: (B, C, H, W) in [0,1]
    Returns scalar SSIM.
    """
    C1, C2 = 0.01**2, 0.03**2
    mu1 = F.avg_pool2d(img1, window_size, 1, window_size//2)
    mu2 = F.avg_pool2d(img2, window_size, 1, window_size//2)
    mu1_sq, mu2_sq = mu1**2, mu2**2
    mu12 = mu1 * mu2
    sig1 = F.avg_pool2d(img1**2, window_size, 1, window_size//2) - mu1_sq
    sig2 = F.avg_pool2d(img2**2, window_size, 1, window_size//2) - mu2_sq
    sig12 = F.avg_pool2d(img1*img2, window_size, 1, window_size//2) - mu12
    num = (2*mu12 + C1) * (2*sig12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (sig1 + sig2 + C2)
    return (num / den).mean().item()


def psnr(rendered: torch.Tensor, target: torch.Tensor) -> float:
    """Peak Signal-to-Noise Ratio."""
    mse = F.mse_loss(rendered.clamp(0,1), target.clamp(0,1)).item()
    if mse < 1e-10:
        return 100.0
    return 10 * np.log10(1.0 / mse)


def nme_2d(pred_lmks: torch.Tensor, gt_lmks: torch.Tensor) -> float:
    """
    Normalised Mean Error on 2D landmarks.
    pred_lmks, gt_lmks: (B, 68, 2)
    NME = mean L2 / inter-ocular distance
    """
    # Skip all-zero samples
    valid = (gt_lmks.abs().sum(dim=[1,2]) > 0.1)
    if valid.sum() == 0:
        return 0.0
    p = pred_lmks[valid]
    g = gt_lmks[valid]
    # Inter-ocular distance: left eye (36) vs right eye (45) in 68-pt scheme
    iod = (g[:, 36] - g[:, 45]).norm(dim=1, keepdim=True).unsqueeze(1) + 1e-8
    err = (p - g).norm(dim=2) / iod.squeeze(-1)
    return err.mean().item()


def mesh_regularity(verts: torch.Tensor, faces: torch.Tensor) -> float:
    """
    Mean edge length variance — lower means more regular mesh.
    verts: (B, V, 3), faces: (F, 3)
    """
    B = verts.shape[0]
    total = 0.0
    for b in range(min(B, 4)):   # sample up to 4 items for speed
        v = verts[b]
        e01 = (v[faces[:,0]] - v[faces[:,1]]).norm(dim=1)
        e12 = (v[faces[:,1]] - v[faces[:,2]]).norm(dim=1)
        e20 = (v[faces[:,2]] - v[faces[:,0]]).norm(dim=1)
        lengths = torch.cat([e01, e12, e20])
        total += lengths.std().item() / (lengths.mean().item() + 1e-8)
    return total / min(B, 4)


def compute_metrics(model_out: dict, batch: dict) -> dict:
    """
    Aggregate all metrics into a single dict.
    """
    device  = model_out['rendered_img'].device
    target  = batch['image_raw'].to(device).clamp(0, 1)
    rendered = model_out['rendered_img'].clamp(0, 1)

    # Resize to match if needed
    if rendered.shape != target.shape:
        rendered = F.interpolate(rendered, size=target.shape[2:],mode='bilinear', align_corners=False)

    return {
        'psnr':psnr(rendered, target),
        'ssim':ssim(rendered, target),
        'mesh_regularity':mesh_regularity(model_out['refined_verts'],model_out['faces']),
    }
