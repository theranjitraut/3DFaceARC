"""
models/gaussian_splatting.py
3D Gaussian Splatting renderer for FaceARCs.
Anchors learnable Gaussians to GCN-refined mesh vertices.
Each Gaussian encodes position, covariance, opacity, and view-dependent colour (SH).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

SH_C0 = 0.28209479177387814

def eval_sh_deg1(sh, dirs):
    C1 = 0.4886025119029199
    x,y,z = dirs[...,0:1], dirs[...,1:2], dirs[...,2:3]
    sh = sh.view(*sh.shape[:-1], 4, 3)
    return (SH_C0*sh[...,0,:] - C1*y*sh[...,1,:] + C1*z*sh[...,2,:] - C1*x*sh[...,3,:])


class GaussianAttributePredictor(nn.Module):
    def __init__(self, in_dim=67, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ELU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim//2), nn.ELU(inplace=True))
        self.pos_head     = nn.Linear(hidden_dim//2, 3)
        self.scale_head   = nn.Linear(hidden_dim//2, 3)
        self.rot_head     = nn.Linear(hidden_dim//2, 4)
        self.opacity_head = nn.Linear(hidden_dim//2, 1)
        self.sh_head      = nn.Linear(hidden_dim//2, 12)
        for head in [self.pos_head, self.scale_head, self.rot_head, self.sh_head]:
            nn.init.uniform_(head.weight, -1e-3, 1e-3); nn.init.zeros_(head.bias)
        nn.init.constant_(self.opacity_head.bias, -2.0)

    def forward(self, node_feats):
        h = self.net(node_feats)
        q = F.normalize(self.rot_head(h), dim=-1)
        return {'pos_offset': self.pos_head(h),
                'log_scale':  self.scale_head(h).clamp(-8, 2),
                'rotation':   q,
                'opacity':    torch.sigmoid(self.opacity_head(h)),
                'sh':         self.sh_head(h)}


def build_covariance_3d(scales, quats):
    s  = torch.exp(scales)
    S  = torch.diag_embed(s)
    w,x,y,z = quats[...,0], quats[...,1], quats[...,2], quats[...,3]
    R  = torch.stack([
        1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y),
        2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x),
        2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)
    ], dim=-1).view(*quats.shape[:-1], 3, 3)
    RS = torch.matmul(R, S)
    return torch.matmul(RS, RS.transpose(-1,-2))


def project_gaussians_2d(means3d, covs3d, K, W2C, image_size):
    B, N = means3d.shape[:2]
    ones  = torch.ones(B, N, 1, device=means3d.device)
    pts_h = torch.cat([means3d, ones], dim=-1)
    pts_c = torch.bmm(pts_h, W2C.transpose(1,2))[...,:3]
    depths= pts_c[...,2].clamp(min=0.01)
    fx = K[:,0,0].unsqueeze(1); fy = K[:,1,1].unsqueeze(1)
    cx = K[:,0,2].unsqueeze(1); cy = K[:,1,2].unsqueeze(1)
    u = (pts_c[...,0]/depths)*fx + cx
    v = (pts_c[...,1]/depths)*fy + cy
    means2d = torch.stack([u,v], dim=-1)
    fx_b = K[:,0,0].view(B,1,1); fy_b = K[:,1,1].view(B,1,1)
    d    = depths.unsqueeze(-1)
    J    = torch.zeros(B, N, 2, 3, device=means3d.device)
    J[...,0,0] = (fx_b/d).squeeze(-1)
    J[...,1,1] = (fy_b/d).squeeze(-1)
    J[...,0,2] = (-fx_b*pts_c[...,0:1]/(d**2)).squeeze(-1)
    J[...,1,2] = (-fy_b*pts_c[...,1:2]/(d**2)).squeeze(-1)
    R_cam = W2C[:,:3,:3]
    R_exp = R_cam.unsqueeze(1).expand(B,N,3,3)
    cov_c = torch.matmul(R_exp, torch.matmul(covs3d, R_exp.transpose(-1,-2)))
    covs2d= torch.matmul(J, torch.matmul(cov_c, J.transpose(-1,-2)))
    return {'means2d':means2d, 'covs2d':covs2d, 'depths':depths, 'pts_cam':pts_c}


def rasterise_gaussians(means2d, covs2d, colours, opacities, depths, H, W):
    B, N  = means2d.shape[:2]
    device= means2d.device
    canvas      = torch.zeros(B, H, W, 3, device=device)
    canvas_alpha= torch.zeros(B, H, W, 1, device=device)
    sort_idx    = depths.argsort(dim=1, descending=True)
    py = torch.arange(H, device=device).float()
    px = torch.arange(W, device=device).float()
    grid_y, grid_x = torch.meshgrid(py, px, indexing='ij')
    pixel_coords = torch.stack([grid_x, grid_y], dim=-1)
    chunk_size = min(N, 512)
    for b in range(B):
        idx = sort_idx[b]
        m   = means2d[b][idx]; c2 = covs2d[b][idx]
        col = colours[b][idx]; opa= opacities[b][idx].squeeze(-1)
        for start in range(0, N, chunk_size):
            end   = min(start+chunk_size, N)
            m_c   = m[start:end]; c2_c = c2[start:end]
            col_c = col[start:end]; opa_c= opa[start:end]
            C     = m_c.shape[0]
            diff  = pixel_coords.unsqueeze(2) - m_c.view(1,1,C,2)
            try:
                prec = torch.linalg.inv(c2_c + 1e-6*torch.eye(2,device=device))
            except Exception:
                prec = torch.eye(2,device=device).unsqueeze(0).expand(C,-1,-1)
            diff_t = diff.unsqueeze(-2)
            maha   = (diff_t @ prec.view(1,1,C,2,2) @ diff.unsqueeze(-1)).squeeze(-1).squeeze(-1)
            gauss  = torch.exp(-0.5*maha.clamp(max=20))
            alpha  = opa_c.view(1,1,C) * gauss
            contrib= alpha.unsqueeze(-1) * col_c.view(1,1,C,3)
            canvas[b]      += contrib.sum(dim=2) * (1-canvas_alpha[b])
            canvas_alpha[b] = (canvas_alpha[b]+alpha.sum(dim=2,keepdim=True)*(1-canvas_alpha[b])).clamp(0,1)
    return canvas.permute(0,3,1,2).clamp(0,1)


class GaussianSplattingRenderer(nn.Module):
    def __init__(self, node_feat_dim=67, image_size=224):
        super().__init__()
        self.H = image_size; self.W = image_size
        self.attr_predictor = GaussianAttributePredictor(in_dim=node_feat_dim, hidden_dim=128)

    def _default_camera(self, B, device):
        focal = self.H * 1.2
        K = torch.tensor([[focal,0,self.W/2],[0,focal,self.H/2],[0,0,1]],
                          device=device).unsqueeze(0).expand(B,-1,-1).float()
        W2C = torch.eye(4, device=device).unsqueeze(0).expand(B,-1,-1).clone()
        W2C[:,2,3] = -2.7
        return K, W2C

    def forward(self, verts, node_feats, K=None, W2C=None):
        B, V, _ = verts.shape; device = verts.device
        if K is None or W2C is None:
            K, W2C = self._default_camera(B, device)
        attrs   = self.attr_predictor(node_feats)
        means3d = verts + attrs['pos_offset']
        covs3d  = build_covariance_3d(attrs['log_scale'], attrs['rotation'])
        cam_pos = W2C[:,:3,3].unsqueeze(1).expand(B,V,3)
        view_dirs = F.normalize(means3d - cam_pos, dim=-1)
        colours   = eval_sh_deg1(attrs['sh'], view_dirs).clamp(0,1)
        proj      = project_gaussians_2d(means3d, covs3d, K, W2C, self.H)
        rendered  = rasterise_gaussians(
            proj['means2d'], proj['covs2d'], colours,
            attrs['opacity'], proj['depths'], self.H, self.W)
        return {'image':rendered, 'means3d':means3d, 'covs3d':covs3d,
                'opacities':attrs['opacity'], 'depths':proj['depths']}