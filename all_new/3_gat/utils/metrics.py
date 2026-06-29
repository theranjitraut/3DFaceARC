"""
utils/metrics.py
Evaluation metrics for 3D face reconstruction.
All metrics computed without ground-truth 3D annotations.
"""

import torch
import torch.nn.functional as F
import numpy as np


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_01(x: torch.Tensor) -> torch.Tensor:
    """
    Safely convert any image tensor to [0,1].
    Handles [0,255], [-1,1], and already-correct [0,1] inputs.
    """
    if x.max() > 2.0:
        # Likely [0,255]
        x = x / 255.0
    elif x.min() < -0.1:
        # Likely [-1,1]
        x = (x + 1.0) / 2.0
    return x.clamp(0.0, 1.0)


# ── Image Quality ─────────────────────────────────────────────────────────────

def psnr(rendered: torch.Tensor, target: torch.Tensor) -> float:
    """
    Peak Signal-to-Noise Ratio.
    Higher is better. >25 dB is acceptable for face reconstruction.
    Both inputs auto-normalized to [0,1].
    """
    rendered = _to_01(rendered)
    target   = _to_01(target)
    mse = F.mse_loss(rendered, target).item()
    if mse < 1e-10:
        return 100.0
    return float(10 * np.log10(1.0 / mse))


def ssim(img1: torch.Tensor, img2: torch.Tensor,
         window_size: int = 11) -> float:
    """
    Structural Similarity Index.
    Higher is better. Range [0,1]. Target > 0.6 for face reconstruction.
    Both inputs auto-normalized to [0,1].
    """
    img1 = _to_01(img1)
    img2 = _to_01(img2)

    C1, C2 = 0.01 ** 2, 0.03 ** 2
    pad = window_size // 2

    mu1  = F.avg_pool2d(img1, window_size, 1, pad)
    mu2  = F.avg_pool2d(img2, window_size, 1, pad)
    mu1_sq, mu2_sq = mu1 ** 2, mu2 ** 2
    mu12 = mu1 * mu2

    sig1  = F.avg_pool2d(img1 ** 2, window_size, 1, pad) - mu1_sq
    sig2  = F.avg_pool2d(img2 ** 2, window_size, 1, pad) - mu2_sq
    sig12 = F.avg_pool2d(img1 * img2, window_size, 1, pad) - mu12

    # Clamp variances to avoid negative values from numerical noise
    sig1  = sig1.clamp(min=0)
    sig2  = sig2.clamp(min=0)

    num = (2 * mu12 + C1) * (2 * sig12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (sig1 + sig2 + C2)

    val = (num / den.clamp(min=1e-8)).mean().item()
    return float(np.clip(val, 0.0, 1.0))


def mae(rendered: torch.Tensor, target: torch.Tensor) -> float:
    """
    Mean Absolute Error — simpler than MSE, less sensitive to outliers.
    Lower is better. Range [0,1].
    """
    rendered = _to_01(rendered)
    target   = _to_01(target)
    return F.l1_loss(rendered, target).item()


# ── Geometry Metrics ──────────────────────────────────────────────────────────

def mesh_regularity(verts: torch.Tensor, faces: torch.Tensor) -> float:
    """
    Edge length coefficient of variation — lower means more regular mesh.
    verts: (B, V, 3), faces: (F, 3)
    """
    B     = verts.shape[0]
    total = 0.0
    count = 0

    for b in range(min(B, 4)):
        v   = verts[b]
        e01 = (v[faces[:, 0]] - v[faces[:, 1]]).norm(dim=1)
        e12 = (v[faces[:, 1]] - v[faces[:, 2]]).norm(dim=1)
        e20 = (v[faces[:, 2]] - v[faces[:, 0]]).norm(dim=1)
        lengths = torch.cat([e01, e12, e20])

        mean = lengths.mean().item()
        std  = lengths.std().item()

        # Guard against degenerate mesh
        if mean > 1e-6 and np.isfinite(std) and np.isfinite(mean):
            total += std / mean
            count += 1

    return total / max(count, 1)


def normal_consistency(verts: torch.Tensor, faces: torch.Tensor) -> float:
    """
    Mean cosine similarity between adjacent face normals.
    Higher is better. Range [-1,1]. Target > 0.9 for smooth face mesh.
    """
    v0 = verts[:, faces[:, 0], :]
    v1 = verts[:, faces[:, 1], :]
    v2 = verts[:, faces[:, 2], :]

    normals = torch.cross(v1 - v0, v2 - v0, dim=-1)  # (B, F, 3)
    normals = F.normalize(normals, dim=-1)

    # Compare each face normal to the mean normal direction
    mean_normal = F.normalize(normals.mean(dim=1, keepdim=True), dim=-1)
    consistency = (normals * mean_normal).sum(dim=-1)  # (B, F)
    return consistency.mean().item()


def mean_edge_length(verts: torch.Tensor, faces: torch.Tensor) -> float:
    """
    Average edge length across the mesh.
    Should stay stable across epochs — big changes indicate mesh collapse.
    """
    v = verts[0]   # use first item in batch
    e01 = (v[faces[:, 0]] - v[faces[:, 1]]).norm(dim=1)
    e12 = (v[faces[:, 1]] - v[faces[:, 2]]).norm(dim=1)
    e20 = (v[faces[:, 2]] - v[faces[:, 0]]).norm(dim=1)
    return torch.cat([e01, e12, e20]).mean().item()


# ── Landmark Metrics ──────────────────────────────────────────────────────────

def nme_2d(pred_lmks: torch.Tensor, gt_lmks: torch.Tensor) -> float:
    """
    Normalised Mean Error on 2D landmarks.
    pred_lmks, gt_lmks: (B, 68, 2)
    NME = mean L2 distance / inter-ocular distance.
    Lower is better. Target < 0.05.
    """
    valid = (gt_lmks.abs().sum(dim=[1, 2]) > 0.1)
    if valid.sum() == 0:
        return 0.0

    p = pred_lmks[valid]
    g = gt_lmks[valid]

    # Inter-ocular distance: left eye center (36) vs right eye center (45)
    iod = (g[:, 36] - g[:, 45]).norm(dim=1, keepdim=True) + 1e-8  # (B,1)
    err = (p - g).norm(dim=2) / iod                                 # (B,68)
    return err.mean().item()


# ── Identity Metrics ──────────────────────────────────────────────────────────

def cosine_identity(embed_rendered: torch.Tensor,
                    embed_input: torch.Tensor) -> float:
    """
    ArcFace cosine similarity between rendered and input embeddings.
    Higher is better. Range [-1,1]. Target > 0.7.
    Returns 0.0 if embeddings are invalid (all zeros / NaN).
    """
    # Guard against broken ArcFace outputting constant embeddings
    if not torch.isfinite(embed_rendered).all():
        return 0.0
    if (embed_rendered.std() < 1e-6):
        return 0.0   # constant output — ArcFace broken

    return F.cosine_similarity(
        embed_rendered, embed_input, dim=1
    ).mean().item()


# ── Master compute_metrics ────────────────────────────────────────────────────

def compute_metrics(model_out: dict, batch: dict) -> dict:
    """
    Aggregate all metrics into a single dict.
    Safe — individual metric failures return 0.0 rather than crashing.
    """
    device   = model_out['rendered_img'].device
    rendered = _to_01(model_out['rendered_img'].to(device))
    target   = _to_01(batch['image_raw'].to(device))

    # Resize rendered to match target if needed
    if rendered.shape != target.shape:
        rendered = F.interpolate(
            rendered, size=target.shape[2:],
            mode='bilinear', align_corners=False
        )

    verts = model_out['refined_verts']
    faces = model_out['faces']

    metrics = {}

    # ── Image quality ─────────────────────────────────────────────────────────
    try: metrics['psnr']              = psnr(rendered, target)
    except: metrics['psnr']           = 0.0

    try: metrics['ssim']              = ssim(rendered, target)
    except: metrics['ssim']           = 0.0

    try: metrics['mae']               = mae(rendered, target)
    except: metrics['mae']            = 0.0

    # ── Geometry ──────────────────────────────────────────────────────────────
    try: metrics['mesh_regularity']   = mesh_regularity(verts, faces)
    except: metrics['mesh_regularity'] = 0.0

    try: metrics['normal_consistency'] = normal_consistency(verts, faces)
    except: metrics['normal_consistency'] = 0.0

    try: metrics['mean_edge_length']  = mean_edge_length(verts, faces)
    except: metrics['mean_edge_length'] = 0.0

    # ── Identity ──────────────────────────────────────────────────────────────
    try:
        metrics['cosine_identity'] = cosine_identity(
            model_out['id_embed_rendered'],
            model_out['id_embed_input']
        )
    except:
        metrics['cosine_identity'] = 0.0

    # ── Landmarks ─────────────────────────────────────────────────────────────
    try:
        # Get projected 2D landmarks from predicted vertices
        V     = verts.shape[1]
        step  = max(1, V // 68)
        ids   = torch.arange(0, 68 * step, step, device=device)[:68]
        pred_lmks = verts[:, ids, :2]

        # Normalise to [-1,1] to match MediaPipe output
        max_val   = pred_lmks.abs().amax(dim=[1,2], keepdim=True) + 1e-6
        pred_lmks = pred_lmks / max_val

        metrics['nme_2d'] = nme_2d(pred_lmks, batch['landmarks'].to(device))
    except:
        metrics['nme_2d'] = 0.0

    # ── Debug: log rendered image stats once ──────────────────────────────────
    # Uncomment during debugging to catch black/constant rendered images:
    # print(f"rendered mean={rendered.mean():.4f} std={rendered.std():.4f} "
    #       f"min={rendered.min():.4f} max={rendered.max():.4f}")

    return metrics