"""
models/bfm.py
Basel Face Model (3DMM) utilities.
Handles shape/expression/texture basis loading and mesh generation.
If the BFM .mat file is absent, a synthetic mini-BFM is generated for testing.
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

# Synthetic BFM (used when the real .mat file is not present)
def _make_synthetic_bfm(n_shape=80, n_exp=64, n_tex=80, n_verts=10698):
    """
    Creates a minimal synthetic BFM dictionary for unit-testing / demo.
    In production, replace with the real Basel Face Model .mat file.
    """
    rng = np.random.default_rng(0)
    mean_shape = rng.standard_normal((n_verts * 3,)).astype(np.float32) * 30
    shape_basis = rng.standard_normal((n_verts * 3, n_shape)).astype(np.float32) * 5
    shape_std   = np.ones(n_shape, dtype=np.float32)

    exp_basis   = rng.standard_normal((n_verts * 3, n_exp)).astype(np.float32) * 3
    exp_std     = np.ones(n_exp, dtype=np.float32)

    mean_tex    = (rng.random((n_verts * 3,)) * 255).astype(np.float32)
    tex_basis   = rng.standard_normal((n_verts * 3, n_tex)).astype(np.float32) * 10
    tex_std     = np.ones(n_tex, dtype=np.float32)

    # Simple triangular faces (strip)
    faces = []
    for i in range(n_verts - 2):
        if i % 2 == 0:
            faces.append([i, i+1, i+2])
        else:
            faces.append([i+1, i, i+2])
    faces = np.array(faces[:n_verts*2], dtype=np.int64)

    return {
        'mean_shape': mean_shape,
        'shape_basis': shape_basis,
        'shape_std':   shape_std,
        'exp_basis':   exp_basis,
        'exp_std':     exp_std,
        'mean_tex':    mean_tex,
        'tex_basis':   tex_basis,
        'tex_std':     tex_std,
        'faces':       faces,
        'n_verts':     n_verts,
    }

# BFM Loader — supports .h5 (BFM2019) and .mat (BFM09)
# def load_bfm(model_path: str, n_shape=80, n_exp=64, n_tex=80, n_verts=10698):
def load_bfm(model_path: str, n_shape=80, n_exp=64, n_tex=80, n_verts=10698):
    """
    Load the Basel Face Model from a .h5 (BFM2019) or .mat (BFM09) file.
    Falls back to a synthetic model if the file is not found.
    """
    path = Path(model_path)
    if not path.exists():
        print(f"[BFM] WARNING: '{model_path}' not found. Using synthetic BFM for testing.")
        print("[BFM] Download the real model from: https://faces.dmi.unibas.ch/bfm/")
        return _make_synthetic_bfm(n_shape, n_exp, n_tex)

    suffix = path.suffix.lower()

    # HDF5 / BFM 2019
    if suffix in ('.h5', '.hdf5'):
        try:
            import h5py
            with h5py.File(str(path), 'r') as f:
                mean_shape  = f['shape/model/mean'][:].astype(np.float32)
                shape_basis = f['shape/model/pcaBasis'][:].astype(np.float32)
                shape_var   = f['shape/model/pcaVariance'][:].astype(np.float32)

                mean_exp    = f['expression/model/mean'][:].astype(np.float32)
                exp_basis   = f['expression/model/pcaBasis'][:].astype(np.float32)
                exp_var     = f['expression/model/pcaVariance'][:].astype(np.float32)

                mean_tex    = f['color/model/mean'][:].astype(np.float32)
                tex_basis   = f['color/model/pcaBasis'][:].astype(np.float32)
                tex_var     = f['color/model/pcaVariance'][:].astype(np.float32)

                # Faces stored as (3, F) column-major; transpose to (F, 3)
                faces = f['shape/representer/cells'][:].T.astype(np.int64)

            n_verts = mean_shape.shape[0] // 3
            n_shape = min(n_shape, shape_basis.shape[1])
            n_exp   = min(n_exp,   exp_basis.shape[1])
            n_tex   = min(n_tex,   tex_basis.shape[1])

            bfm = {
                'mean_shape':  mean_shape,
                'shape_basis': shape_basis[:, :n_shape],
                'shape_std':   np.sqrt(shape_var[:n_shape]),
                'exp_basis':   exp_basis[:, :n_exp],
                'exp_std':     np.sqrt(exp_var[:n_exp]),
                'mean_tex':    mean_tex * 255.0,
                'tex_basis':   tex_basis[:, :n_tex] * 255.0,
                'tex_std':     np.sqrt(tex_var[:n_tex]),
                'faces':       faces,
                'n_verts':     n_verts,
            }
            print(f"[BFM] Loaded BFM2019 (.h5): {n_verts} vertices, "
                  f"{len(faces)} faces  (shape={n_shape}, exp={n_exp}, tex={n_tex})")
            return bfm
        except Exception as e:
            print(f"[BFM] Error loading .h5: {e}. Falling back to synthetic BFM.")
            return _make_synthetic_bfm(n_shape, n_exp, n_tex)

    # MAT / BFM 09 
    try:
        import scipy.io as sio
        raw = sio.loadmat(str(path))
        bfm = {
            'mean_shape':  raw['meanshape'].flatten().astype(np.float32),
            'shape_basis': raw['idBase'].astype(np.float32)[:, :n_shape],
            'shape_std':   raw['idStd'].flatten().astype(np.float32)[:n_shape],
            'exp_basis':   raw['exBase'].astype(np.float32)[:, :n_exp],
            'exp_std':     raw['exStd'].flatten().astype(np.float32)[:n_exp],
            'mean_tex':    raw['meantex'].flatten().astype(np.float32),
            'tex_basis':   raw['texBase'].astype(np.float32)[:, :n_tex],
            'tex_std':     raw['texStd'].flatten().astype(np.float32)[:n_tex],
            'faces':       raw['tri'].astype(np.int64) - 1,  # 1-indexed to 0-indexed
            'n_verts':     raw['meanshape'].shape[0] // 3,
        }
        print(f"[BFM] Loaded BFM09 (.mat): {bfm['n_verts']} vertices, "
              f"{len(bfm['faces'])} faces")
        return bfm
    except Exception as e:
        print(f"[BFM] Error loading .mat: {e}. Falling back to synthetic BFM.")
        return _make_synthetic_bfm(n_shape, n_exp, n_tex)


# 3DMM Differentiable Layer
class BFMLayer(nn.Module):
    """
    Differentiable 3DMM decoder.
    Converts coefficient vectors → 3D vertex positions and colours.

    Input coefficients:
        shape_coeff : (B, n_shape)
        exp_coeff   : (B, n_exp)
        tex_coeff   : (B, n_tex)
        angles      : (B, 3)   Euler angles (pitch, yaw, roll)
        translation : (B, 3)
        scale       : (B, 1)

    Outputs:
        vertices    : (B, n_verts, 3)   3D positions
        colors      : (B, n_verts, 3)   RGB colours [0,1]
    """

    def __init__(self, bfm_dict: dict):
        super().__init__()
        bfm = bfm_dict

        # Register non-trainable buffers
        self.register_buffer('mean_shape',  torch.from_numpy(bfm['mean_shape']))
        self.register_buffer('shape_basis', torch.from_numpy(bfm['shape_basis']))
        self.register_buffer('shape_std',   torch.from_numpy(bfm['shape_std']))
        self.register_buffer('exp_basis',   torch.from_numpy(bfm['exp_basis']))
        self.register_buffer('exp_std',     torch.from_numpy(bfm['exp_std']))
        self.register_buffer('mean_tex',    torch.from_numpy(bfm['mean_tex']))
        self.register_buffer('tex_basis',   torch.from_numpy(bfm['tex_basis']))
        self.register_buffer('tex_std',     torch.from_numpy(bfm['tex_std']))

        faces = torch.from_numpy(bfm['faces'].astype(np.int64))
        self.register_buffer('faces', faces)
        self.n_verts = bfm['n_verts']

    # geometry
    def decode_shape(self, shape_coeff, exp_coeff):
        """
        shape_coeff: (B, n_shape)
        exp_coeff  : (B, n_exp)
        returns    : (B, n_verts, 3)
        """
        B = shape_coeff.shape[0]
        # (B, 3*V)
        shape_offset = torch.matmul(shape_coeff * self.shape_std, self.shape_basis.T)
        exp_offset   = torch.matmul(exp_coeff   * self.exp_std,   self.exp_basis.T)
        verts = self.mean_shape + shape_offset + exp_offset   # (B, 3V)
        return verts.view(B, self.n_verts, 3)

    # texture
    def decode_texture(self, tex_coeff):
        """
        tex_coeff: (B, n_tex)
        returns  : (B, n_verts, 3) in [0,1]
        """
        B = tex_coeff.shape[0]
        tex_offset = torch.matmul(tex_coeff * self.tex_std, self.tex_basis.T)
        colors = (self.mean_tex + tex_offset).view(B, self.n_verts, 3)
        return colors.clamp(0, 255) / 255.0

    # rigid transform
    @staticmethod
    def euler_to_rotation(angles):
        """
        angles: (B, 3) → rotation matrix (B, 3, 3)
        """
        pitch, yaw, roll = angles[:, 0], angles[:, 1], angles[:, 2]
        cos_p, sin_p = torch.cos(pitch), torch.sin(pitch)
        cos_y, sin_y = torch.cos(yaw),   torch.sin(yaw)
        cos_r, sin_r = torch.cos(roll),  torch.sin(roll)

        zeros = torch.zeros_like(pitch)
        ones  = torch.ones_like(pitch)

        Rx = torch.stack([ones,  zeros,  zeros,
                          zeros, cos_p, -sin_p,
                          zeros, sin_p,  cos_p], dim=1).view(-1, 3, 3)
        Ry = torch.stack([ cos_y, zeros, sin_y,
                           zeros, ones,  zeros,
                          -sin_y, zeros, cos_y], dim=1).view(-1, 3, 3)
        Rz = torch.stack([cos_r, -sin_r, zeros,
                          sin_r,  cos_r, zeros,
                          zeros,  zeros, ones], dim=1).view(-1, 3, 3)
        return torch.bmm(Rz, torch.bmm(Ry, Rx))

    def apply_rigid(self, verts, angles, translation, scale):
        """
        verts      : (B, V, 3)
        angles     : (B, 3)
        translation: (B, 3)
        scale      : (B, 1)
        """
        R = self.euler_to_rotation(angles)                    # (B,3,3)
        verts_r = torch.bmm(verts, R.transpose(1, 2))        # (B,V,3)
        verts_r = verts_r * scale.unsqueeze(-1)               # scale
        verts_r = verts_r + translation.unsqueeze(1)          # translate
        return verts_r

    # forward
    def forward(self, coeffs: dict):
        """
        coeffs dict keys: shape, exp, tex, angles, translation, scale
        Returns: vertices (B,V,3), colors (B,V,3), faces (F,3)
        """
        verts  = self.decode_shape(coeffs['shape'], coeffs['exp'])
        colors = self.decode_texture(coeffs['tex'])
        verts  = self.apply_rigid(verts, coeffs['angles'],
                                  coeffs['translation'], coeffs['scale'])
        return verts, colors, self.faces
