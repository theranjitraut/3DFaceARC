"""
models/nerf.py
Face-conditioned NeRF renderer for FaceARCs.
Replaces Stage 8 (mesh renderer) with a volumetric neural radiance field
conditioned on 3DMM coefficients and GCN-refined vertex positions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, num_freqs=10, include_input=True):
        super().__init__()
        self.num_freqs     = num_freqs
        self.include_input = include_input
        freqs = 2.0 ** torch.linspace(0, num_freqs-1, num_freqs)
        self.register_buffer('freqs', freqs)

    @property
    def out_dim(self):
        d = self.num_freqs * 2
        if self.include_input: d += 1
        return d

    def forward(self, x):
        parts = [x] if self.include_input else []
        for freq in self.freqs:
            parts.append(torch.sin(freq * x))
            parts.append(torch.cos(freq * x))
        return torch.cat(parts, dim=-1)


class CoeffEmbedder(nn.Module):
    def __init__(self, coeff_dim=231, embed_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(coeff_dim, 256), nn.ELU(inplace=True),
            nn.Linear(256, 128),       nn.ELU(inplace=True),
            nn.Linear(128, embed_dim),
        )
    def forward(self, coeffs):
        coeff_vec = torch.cat([v for v in coeffs.values()], dim=-1)
        return self.net(coeff_vec)


class FaceNeRFMLP(nn.Module):
    def __init__(self, pos_enc_freqs=10, dir_enc_freqs=4,
                 hidden_dim=256, coeff_embed_dim=64, skip_layer=4):
        super().__init__()
        self.skip_layer = skip_layer
        self.pos_enc    = PositionalEncoding(pos_enc_freqs)
        self.dir_enc    = PositionalEncoding(dir_enc_freqs)
        pos_dim = 3 * self.pos_enc.out_dim
        dir_dim = 3 * self.dir_enc.out_dim
        self.density_layers = nn.ModuleList()
        in_dim = pos_dim + coeff_embed_dim
        for i in range(8):
            if i == skip_layer:
                in_dim += pos_dim + coeff_embed_dim
            self.density_layers.append(
                nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(inplace=True)))
            in_dim = hidden_dim
        self.sigma_head  = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Softplus())
        self.colour_feat = nn.Linear(hidden_dim, hidden_dim//2)
        self.colour_head = nn.Sequential(
            nn.Linear(hidden_dim//2+dir_dim, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 3), nn.Sigmoid())

    def forward(self, xyz, view_dir, coeff_emb):
        xyz_enc = self.pos_enc(xyz).view(xyz.shape[0], -1)
        dir_enc = self.dir_enc(view_dir).view(xyz.shape[0], -1)
        h = torch.cat([xyz_enc, coeff_emb], dim=-1)
        inp = h.clone()
        for i, layer in enumerate(self.density_layers):
            if i == self.skip_layer:
                h = torch.cat([h, inp], dim=-1)
            h = layer(h)
        sigma = self.sigma_head(h)
        feat  = self.colour_feat(h)
        rgb   = self.colour_head(torch.cat([feat, dir_enc], dim=-1))
        return rgb, sigma


def generate_rays(H, W, focal, c2w):
    B = c2w.shape[0]; device = c2w.device
    i, j = torch.meshgrid(
        torch.arange(W, dtype=torch.float32, device=device),
        torch.arange(H, dtype=torch.float32, device=device), indexing='xy')
    dirs = torch.stack([(i-W*0.5)/focal, -(j-H*0.5)/focal, -torch.ones_like(i)], dim=-1)
    dirs = dirs.unsqueeze(0).expand(B,-1,-1,-1)
    R    = c2w[:,:3,:3]
    ray_dirs = (R.unsqueeze(1).unsqueeze(1) @ dirs.unsqueeze(-1)).squeeze(-1)
    ray_dirs = F.normalize(ray_dirs, dim=-1)
    ray_origins = c2w[:,:3,3].unsqueeze(1).unsqueeze(1).expand_as(ray_dirs)
    return ray_origins, ray_dirs


def volume_render(rgb_samples, sigma_samples, z_vals, ray_dirs):
    dists = z_vals[...,1:] - z_vals[...,:-1]
    last  = torch.full((*z_vals.shape[:-1],1), 1e10, device=z_vals.device)
    dists = torch.cat([dists, last], dim=-1)
    dists = dists * ray_dirs.norm(dim=-1, keepdim=True).unsqueeze(-1)
    alpha = 1.0 - torch.exp(-F.relu(sigma_samples[...,0]) * dists)
    ones  = torch.ones((*alpha.shape[:-1],1), device=alpha.device)
    trans = torch.cat([ones, (1-alpha+1e-10)[...,:-1]], dim=-1)
    T       = torch.cumprod(trans, dim=-1)
    weights = alpha * T
    rgb_map   = (weights.unsqueeze(-1) * rgb_samples).sum(dim=-2)
    depth_map = (weights * z_vals).sum(dim=-1)
    acc_map   = weights.sum(dim=-1)
    return {'rgb_map':rgb_map, 'depth_map':depth_map, 'acc_map':acc_map, 'weights':weights}


class FaceNeRFRenderer(nn.Module):
    def __init__(self, image_size=64, n_samples=64, near=0.5, far=3.0,
                 coeff_embed_dim=64, hidden_dim=256):
        super().__init__()
        self.H=image_size; self.W=image_size
        self.n_samples=n_samples; self.near=near; self.far=far
        self.focal = image_size * 1.2
        self.coeff_embedder = CoeffEmbedder(coeff_dim=231, embed_dim=coeff_embed_dim)
        self.nerf_mlp       = FaceNeRFMLP(hidden_dim=hidden_dim, coeff_embed_dim=coeff_embed_dim)
        self.upsampler      = nn.Sequential(
            nn.Conv2d(3,64,3,padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64,32,3,padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32,3,3,padding=1), nn.Sigmoid())

    def _default_c2w(self, B, device):
        c2w = torch.eye(4, device=device).unsqueeze(0).expand(B,-1,-1).clone()
        c2w[:,2,3] = 2.7
        return c2w

    def forward(self, coeffs, verts, c2w=None, target_size=224):
        B = verts.shape[0]; device = verts.device
        if c2w is None: c2w = self._default_c2w(B, device)
        coeff_emb = self.coeff_embedder(coeffs)
        ray_o, ray_d = generate_rays(self.H, self.W, self.focal, c2w)
        N_rays = self.H * self.W
        ray_o  = ray_o.view(B, N_rays, 3)
        ray_d  = ray_d.view(B, N_rays, 3)
        z_vals = torch.linspace(self.near, self.far, self.n_samples, device=device)
        z_vals = z_vals.unsqueeze(0).unsqueeze(0).expand(B, N_rays, -1)
        if self.training:
            z_vals = z_vals + torch.rand_like(z_vals) * (self.far-self.near) / self.n_samples
        pts = ray_o.unsqueeze(-2) + ray_d.unsqueeze(-2) * z_vals.unsqueeze(-1)
        coeff_exp = coeff_emb.unsqueeze(1).unsqueeze(1).expand(B, N_rays, self.n_samples, -1)
        dirs_exp  = ray_d.unsqueeze(-2).expand(B, N_rays, self.n_samples, 3)
        pts_flat  = pts.reshape(-1, 3)
        dirs_flat = dirs_exp.reshape(-1, 3)
        coeff_flat= coeff_exp.reshape(-1, coeff_emb.shape[-1])
        chunk = 4096
        rgbs, sigmas = [], []
        for i in range(0, pts_flat.shape[0], chunk):
            r, s = self.nerf_mlp(pts_flat[i:i+chunk], dirs_flat[i:i+chunk], coeff_flat[i:i+chunk])
            rgbs.append(r); sigmas.append(s)
        rgb_samples   = torch.cat(rgbs).view(B, N_rays, self.n_samples, 3)
        sigma_samples = torch.cat(sigmas).view(B, N_rays, self.n_samples, 1)
        render_out = volume_render(rgb_samples, sigma_samples, z_vals, ray_d)
        rgb_img    = render_out['rgb_map'].view(B, self.H, self.W, 3).permute(0,3,1,2)
        depth_img  = render_out['depth_map'].view(B, 1, self.H, self.W)
        if self.H != target_size:
            rgb_img   = F.interpolate(rgb_img,   size=(target_size,target_size), mode='bilinear', align_corners=False)
            rgb_img   = self.upsampler(rgb_img)
            depth_img = F.interpolate(depth_img, size=(target_size,target_size), mode='bilinear', align_corners=False)
        return {'image':rgb_img, 'depth':depth_img,
                'acc_map':render_out['acc_map'].view(B,1,self.H,self.W)}