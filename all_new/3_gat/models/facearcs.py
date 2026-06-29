"""
models/facearcs.py
FaceARCs — Full model supporting three renderer modes:
    'mesh'     : Original PyTorch3D / z-buffer renderer
    'nerf'     : Neural Radiance Field renderer
    'gaussian' : 3D Gaussian Splatting renderer

Set mode in configs/config.yaml under renderer.mode
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoder            import FaceEncoder, ArcFaceEncoder
from models.bfm                import BFMLayer, load_bfm
from models.gcn                import BatchedGCNRefiner, VertexFeatureSampler
from models.renderer           import build_renderer, normalise_verts
from models.nerf               import FaceNeRFRenderer
from models.gaussian_splatting import GaussianSplattingRenderer


class FaceARCs(nn.Module):
    """
    FaceARCs v2 — End-to-end 3D Face Reconstruction.

    Pipeline:
      1.  CNN backbone        → feature vector + intermediate feature map
      2.  MLP regressor       → 3DMM coefficients (shape, exp, tex, pose)
      3.  BFM decoder         → coarse 3D mesh (verts, colors, faces)
      4.  VertexSampler       → per-vertex CNN features (B, V, 64)
      5.  GCN refiner         → fine-grained vertex displacements
      6a. [mesh]     Renderer → re-projected 2D image via rasterisation
      6b. [nerf]     Renderer → volumetric radiance field render
      6c. [gaussian] Renderer → 3D Gaussian splatting render
      7.  ArcFace             → identity embeddings for ID loss
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg         = cfg
        self.render_mode = cfg['renderer'].get('mode', 'mesh')
        dev              = cfg['training'].get('device', 'cuda')

        # 1. CNN Encoder
        self.encoder = FaceEncoder(
            backbone_name=cfg['model']['backbone'],
            pretrained=cfg['model']['pretrained'],
            coeff_dims=cfg['model']['coeff_dims']
        )

        # 2. BFM — pass n_verts from config so all 10698 vertices are loaded
        bfm_dict = load_bfm(
            cfg['bfm']['model_path'],
            n_shape=cfg['bfm']['n_shape'],
            n_exp=cfg['bfm']['n_exp'],
            n_tex=cfg['bfm']['n_tex'],
            n_verts=cfg['bfm']['n_vertices'],       # ← was missing; fixes synthetic fallback too
        )
        self.bfm     = BFMLayer(bfm_dict)
        self.n_verts = bfm_dict['n_verts']          # 10698

        # 3. Vertex Feature Sampler
        # vertex_feat_channels from config (default 64); node_feat_dim = 3 + 64 = 67
        self.feat_map_channels  = 1024              # ResNet50 layer3 output channels
        self.vertex_feat_out    = cfg['model'].get('vertex_feat_channels', 64)
        self.vertex_sampler     = VertexFeatureSampler(
            feat_channels=self.feat_map_channels,
            out_channels=self.vertex_feat_out       # 64
        )

        # 4. GCN Refiner
        # node_feat_dim = 3 (xyz) + vertex_feat_out (64) = 67
        # max_verts from config — must equal n_vertices (10698) to refine ALL verts
        gcn_cfg = cfg['model']['gcn']
        self.gcn_refiner = BatchedGCNRefiner(
            node_feat_dim=3 + self.vertex_feat_out, # 67
            hidden_dim=gcn_cfg['hidden_dim'],
            num_layers=gcn_cfg['num_layers'],
            arch=gcn_cfg.get('arch', 'gat'),
            heads=gcn_cfg.get('heads', 4),
            dropout=gcn_cfg['dropout'],
            residual=gcn_cfg['residual'],
            max_verts=gcn_cfg['max_verts'],         # ← was missing; now 10698 from config
        )

        # 5. Renderer (switchable)
        img_size = cfg['renderer']['image_size']
        if self.render_mode == 'nerf':
            print("[FaceARCs] Renderer → NeRF")
            self.renderer = FaceNeRFRenderer(
                image_size=cfg['renderer'].get('nerf_size', 64),
                n_samples=cfg['renderer'].get('nerf_samples', 64),
                near=cfg['renderer'].get('nerf_near', 0.5),
                far=cfg['renderer'].get('nerf_far', 3.0),
                coeff_embed_dim=cfg['renderer'].get('nerf_coeff_embed', 64),
                hidden_dim=cfg['renderer'].get('nerf_hidden', 256),
            )
        elif self.render_mode == 'gaussian':
            print("[FaceARCs] Renderer → 3D Gaussian Splatting")
            self.renderer = GaussianSplattingRenderer(
                node_feat_dim=3 + self.vertex_feat_out,
                image_size=img_size
            )
        else:
            print("[FaceARCs] Renderer → Mesh (PyTorch3D / fallback)")
            self.renderer = build_renderer(image_size=img_size, device=dev)

        # 6. ArcFace (frozen)
        self.arcface = ArcFaceEncoder(
            weights_path=cfg['paths'].get('pretrained_arcface', ''))

        # 7. Feature map hook
        self._feat_map = None
        if hasattr(self.encoder.backbone, 'layer3'):
            def hook(module, input, output):
                self._feat_map = output
            self.encoder.backbone.layer3.register_forward_hook(hook)

    def forward(self, x: torch.Tensor) -> dict:
        """
        x : (B, 3, H, W) normalised input image
        Returns dict with all intermediate and final outputs.
        """
        B = x.shape[0]

        # Step 1: CNN → coefficients
        coeffs, global_feat = self.encoder(x)

        # Step 2: BFM → coarse mesh
        coarse_verts, colors, faces = self.bfm(coeffs)

        # Step 3: Vertex feature sampling
        # Sample ALL n_verts (10698) so GCN receives real features for every vertex.
        # If GPU memory is tight, reduce gcn.max_verts in config — the GCN will
        # subsample internally and scatter deltas back to full resolution.
        if self._feat_map is not None:
            feat_map = F.interpolate(self._feat_map, size=(56, 56),
                                     mode='bilinear', align_corners=False)
            if feat_map.shape[1] > self.feat_map_channels:
                feat_map = feat_map[:, :self.feat_map_channels]

            vertex_feats = self.vertex_sampler(
                normalise_verts(coarse_verts), feat_map   # all n_verts, not capped at 5000
            )                                             # (B, n_verts, 64)
            self._feat_map = None                         # free GPU memory
        else:
            vertex_feats = torch.zeros(
                B, self.n_verts, self.vertex_feat_out, device=x.device)

        # Step 4: GCN → refined mesh
        # BatchedGCNRefiner handles subsampling internally via max_verts
        refined_verts = self.gcn_refiner(coarse_verts, vertex_feats, faces)

        # Step 5: Render (mode-dependent)
        if self.render_mode == 'nerf':
            render_out   = self.renderer(coeffs, refined_verts, target_size=x.shape[-1])
            rendered_img = render_out['image']
            silhouette   = render_out.get('acc_map', None)

        elif self.render_mode == 'gaussian':
            node_feats_gs = torch.cat(
                [normalise_verts(refined_verts), vertex_feats], dim=-1)
            render_out    = self.renderer(refined_verts, node_feats_gs)
            rendered_img  = render_out['image']
            silhouette    = None

        else:
            verts_norm = normalise_verts(refined_verts)
            render_out = self.renderer(verts_norm, faces, colors)
            if isinstance(render_out, dict):
                rendered_img = render_out['image']
                silhouette   = render_out.get('silhouette', None)
            else:
                rendered_img = render_out
                silhouette   = None

        # Step 6: ArcFace embeddings
        r_res = F.interpolate(rendered_img.clamp(0, 1), size=(224, 224),
                              mode='bilinear', align_corners=False)
        x_res = F.interpolate(x, size=(224, 224),
                              mode='bilinear', align_corners=False)
        with torch.no_grad():
            id_rendered = self.arcface(r_res)
            id_input    = self.arcface(x_res)

        return {
            'coeffs':            coeffs,
            'coarse_verts':      coarse_verts,
            'refined_verts':     refined_verts,
            'colors':            colors,
            'faces':             faces,
            'rendered_img':      rendered_img,
            'silhouette':        silhouette,
            'id_embed_input':    id_input,
            'id_embed_rendered': id_rendered,
            'vertex_feats':      vertex_feats,
            'render_mode':       self.render_mode,
        }