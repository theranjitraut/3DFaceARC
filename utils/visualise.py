"""
utils/visualise.py
Visualisation utilities — save reconstruction grids,
export meshes as OBJ, render turntable GIFs.
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import cv2

# Save reconstruction grid  (input | rendered | depth)
def save_reconstruction_grid(inputs: torch.Tensor,
                              rendered: torch.Tensor,
                              verts: torch.Tensor,
                              faces: torch.Tensor,
                              save_dir: str,
                              epoch: int,
                              n_samples: int = 4):
    """
    Saves a grid: [input image | rendered image | depth map]
    for the first n_samples images in the batch.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    n = min(n_samples, inputs.shape[0])
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for i in range(n):
        # Input
        inp = inputs[i].cpu().permute(1, 2, 0).numpy().clip(0, 1)
        axes[i, 0].imshow(inp)
        axes[i, 0].set_title("Input")
        axes[i, 0].axis('off')

        # Rendered
        rend = rendered[i].detach().cpu().permute(1, 2, 0).numpy().clip(0, 1)
        axes[i, 1].imshow(rend)
        axes[i, 1].set_title("Rendered")
        axes[i, 1].axis('off')

        # Depth (z-coordinate of refined verts, projected to image)
        depth = verts[i, :, 2].detach().cpu().numpy()
        d_min, d_max = depth.min(), depth.max()
        depth_norm = (depth - d_min) / (d_max - d_min + 1e-8)
        axes[i, 2].scatter(
            verts[i, :, 0].detach().cpu().numpy()[::20],
            verts[i, :, 1].detach().cpu().numpy()[::20],
            c=depth_norm[::20], cmap='plasma', s=0.5
        )
        axes[i, 2].set_title("Depth (Z)")
        axes[i, 2].axis('off')
        axes[i, 2].invert_yaxis()

    plt.suptitle(f"Epoch {epoch}", fontsize=14)
    plt.tight_layout()
    save_path = save_dir / f'reconstruction_epoch_{epoch:03d}.png'
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()


# Export mesh as .OBJ
def export_obj(verts: np.ndarray, faces: np.ndarray, colors: np.ndarray,
               path: str):
    """
    Export a 3D mesh as a Wavefront OBJ + MTL file with vertex colours.
    verts  : (V, 3)
    faces  : (F, 3) 0-indexed
    colors : (V, 3) [0,1]
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        f.write("# FaceARCs exported mesh\n")
        for i, (v, c) in enumerate(zip(verts, colors)):
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} "
                    f"{c[0]:.4f} {c[1]:.4f} {c[2]:.4f}\n")
        for face in faces:
            # OBJ is 1-indexed
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    print(f"[Export] Mesh saved → {path}")


# Turntable visualisation (rotate face through yaw angles)
def visualise_turntable(verts: np.ndarray, faces: np.ndarray,
                         colors: np.ndarray, save_path: str,
                         n_frames: int = 36):
    """
    Renders a simple turntable animation by rotating the mesh
    around the Y axis and projecting orthographically.
    Saves frames as PNGs in a folder.
    """
    out_dir = Path(save_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    angles = np.linspace(0, 2 * np.pi, n_frames, endpoint=False)

    for fi, angle in enumerate(angles):
        # Rotation matrix around Y
        c, s = np.cos(angle), np.sin(angle)
        Ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        v_rot = (Ry @ verts.T).T          # (V,3)

        # Orthographic projection
        xs = v_rot[:, 0]
        ys = v_rot[:, 1]
        zs = v_rot[:, 2]

        # Normalise to [0,1]
        xs = (xs - xs.min()) / (xs.max() - xs.min() + 1e-8)
        ys = (ys - ys.min()) / (ys.max() - ys.min() + 1e-8)

        canvas = np.ones((256, 256, 3))
        # Sort faces by mean Z (painter's algorithm)
        z_mean = zs[faces].mean(axis=1)
        order = np.argsort(z_mean)

        for fi2 in order:
            f = faces[fi2]
            pts = np.stack([
                xs[f] * 255,
                ys[f] * 255
            ], axis=-1).astype(np.int32)
            col = colors[f].mean(axis=0)
            cv2.fillPoly(canvas, [pts], color=col.tolist())

        frame_path = out_dir / f'frame_{fi:03d}.png'
        cv2.imwrite(str(frame_path),
                    (canvas * 255).clip(0, 255).astype(np.uint8))

    print(f"[Turntable] {n_frames} frames saved → {out_dir}")
