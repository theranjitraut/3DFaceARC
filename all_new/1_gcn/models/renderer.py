"""
models/renderer.py
Differentiable mesh renderer for photometric self-supervised loss.
Uses PyTorch3D when available, falls back to a simple z-buffer renderer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Try importing PyTorch3D
try:
    from pytorch3d.structures import Meshes
    from pytorch3d.renderer import (
        FoVOrthographicCameras, look_at_view_transform,
        RasterizationSettings, MeshRenderer, MeshRasterizer,
        SoftPhongShader, SoftSilhouetteShader,
        PointLights, AmbientLights,
        TexturesVertex
    )
    P3D_AVAILABLE = True
except ImportError:
    print("[Renderer] PyTorch3D not found. Using fallback renderer.")
    P3D_AVAILABLE = False


# Fallback renderer: simple orthographic z-buffer
class FallbackRenderer(nn.Module):
    """
    Minimal differentiable renderer.
    Renders vertex colours via barycentric interpolation (approx).
    Not as accurate as PyTorch3D but trains without extra deps.
    """

    def __init__(self, image_size: int = 224):
        super().__init__()
        self.H = image_size
        self.W = image_size

    def forward(self, verts: torch.Tensor, faces: torch.Tensor,
                colors: torch.Tensor) -> torch.Tensor:
        """
        verts  : (B, V, 3) — normalised to [-1, 1] in x,y
        faces  : (F, 3)
        colors : (B, V, 3) — [0,1]
        returns: (B, 3, H, W) rendered image
        """
        B, V, _ = verts.shape
        device = verts.device
        canvas = torch.zeros(B, 3, self.H, self.W, device=device)

        # Map x,y from [-1,1] to pixel coords
        px = ((verts[:, :, 0] + 1) * 0.5 * (self.W - 1)).long().clamp(0, self.W - 1)
        py = ((verts[:, :, 1] + 1) * 0.5 * (self.H - 1)).long().clamp(0, self.H - 1)

        # Vectorised point rendering with simple z-buffering (painter's algorithm).
        # We sort vertices by z (depth) so closer vertices overwrite farther ones.
        # PyTorch3D NDC has +Z pointing into the screen, so larger Z is further away.
        # Thus, we draw larger Z first (descending order).
        for b in range(B):
            idx = verts[b, :, 2].argsort(descending=True)
            canvas[b, :, py[b, idx], px[b, idx]] = colors[b, idx].T

        return canvas


# PyTorch3D renderer (preferred)
class PyTorch3DRenderer(nn.Module):
    """
    Differentiable renderer using PyTorch3D's SoftPhongShader.
    Supports vertex colours and basic Phong shading.
    """

    def __init__(self, image_size: int = 224, device: str = 'cuda'):
        super().__init__()
        self.image_size = image_size
        self.device_str = device

        raster_settings = RasterizationSettings(
            image_size=image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
        )

        # Fixed front-facing camera
        R, T = look_at_view_transform(dist=2.7, elev=0, azim=0)
        cameras = FoVOrthographicCameras(device=device, R=R, T=T)

        lights = PointLights(
            device=device,
            location=[[0.0, 0.0, 3.0]],
            ambient_color=[[0.6, 0.6, 0.6]],
            diffuse_color=[[0.4, 0.4, 0.4]],
            specular_color=[[0.05, 0.05, 0.05]]
        )

        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras,
                                      raster_settings=raster_settings),
            shader=SoftPhongShader(device=device, cameras=cameras, lights=lights)
        )

        # Silhouette renderer (for mask loss)
        sil_settings = RasterizationSettings(
            image_size=image_size,
            blur_radius=np.log(1.0 / 1e-4 - 1.0) * 1e-5,
            faces_per_pixel=50
        )
        self.sil_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras,
                                      raster_settings=sil_settings),
            shader=SoftSilhouetteShader()
        )

    def forward(self, verts: torch.Tensor, faces: torch.Tensor,
                colors: torch.Tensor) -> dict:
        """
        verts  : (B, V, 3)
        faces  : (F, 3)
        colors : (B, V, 3)
        returns: dict with 'image' (B,3,H,W) and 'silhouette' (B,1,H,W)
        """
        B = verts.shape[0]
        # Repeat faces for each batch item
        faces_list  = [faces for _ in range(B)]
        verts_list  = [verts[b] for b in range(B)]

        textures = TexturesVertex(verts_features=colors)
        meshes = Meshes(verts=verts_list, faces=faces_list, textures=textures)

        # Render colour image
        rendered = self.renderer(meshes)            # (B, H, W, 4) RGBA
        img = rendered[..., :3].permute(0, 3, 1, 2)   # (B, 3, H, W)

        # Render silhouette
        sil = self.sil_renderer(meshes)[..., 3:4]  # (B, H, W, 1)
        sil = sil.permute(0, 3, 1, 2)              # (B, 1, H, W)

        return {'image': img, 'silhouette': sil}


# Renderer Factory
def build_renderer(image_size: int = 224, device: str = 'cuda') -> nn.Module:
    if P3D_AVAILABLE:
        print("[Renderer] Using PyTorch3D differentiable renderer.")
        return PyTorch3DRenderer(image_size=image_size, device=device)
    else:
        print("[Renderer] Using fallback z-buffer renderer.")
        return FallbackRenderer(image_size=image_size)


# Normalise verts to renderer coordinate space
def normalise_verts(verts: torch.Tensor) -> torch.Tensor:
    """
    Normalise vertex positions to fit in [-1,1]^3 for rendering.
    verts: (B, V, 3)
    """
    center = verts.mean(dim=1, keepdim=True)
    verts  = verts - center
    scale  = verts.abs().max(dim=1, keepdim=True).values.max(dim=-1, keepdim=True).values
    verts  = verts / (scale + 1e-8)
    return verts
