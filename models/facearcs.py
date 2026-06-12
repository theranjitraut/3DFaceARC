"""
models/facearcs.py
FaceARCs — Full model:
  CNN Encoder → 3DMM Coefficients → BFM Mesh → GCN Refinement → Renderer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoder  import FaceEncoder, ArcFaceEncoder
from models.bfm      import BFMLayer, load_bfm
from models.gcn      import BatchedGCNRefiner, VertexFeatureSampler
from models.renderer import build_renderer, normalise_verts


class FaceARCs(nn.Module):
    """
    End-to-end 3D Face Reconstruction model.

    Pipeline:
      1. CNN backbone  → feature vector + intermediate feature map
      2. MLP regressor → 3DMM coefficients (shape, exp, tex, pose)
      3. BFM decoder   → coarse 3D mesh (verts, colors, faces)
      4. VertexSampler → per-vertex CNN features
      5. GCN refiner   → fine-grained vertex displacements
      6. Renderer      → re-projected 2D image for photometric loss
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        dev = cfg['training'].get('device', 'cuda')

        # 1. CNN encoder
        self.encoder = FaceEncoder(
            backbone_name=cfg['model']['backbone'],
            pretrained=cfg['model']['pretrained'],
            coeff_dims=cfg['model']['coeff_dims']
        )

        # 2. BFM
        bfm_dict = load_bfm(
            cfg['bfm']['model_path'],
            n_shape=cfg['bfm']['n_shape'],
            n_exp=cfg['bfm']['n_exp'],
            n_tex=cfg['bfm']['n_tex'],
        )
        self.bfm = BFMLayer(bfm_dict)
        self.n_verts = bfm_dict['n_verts']

        # 3. Vertex feature sampler
        # We hook into an intermediate layer of the backbone to get a feature map
        self.feat_map_channels = 256           # ResNet layer3 output channels
        self.vertex_sampler = VertexFeatureSampler(
            feat_channels=self.feat_map_channels,
            out_channels=64
        )

        # 4. GCN refiner
        # Node features: 3 (xyz) + 64 (CNN features)
        gcn_cfg = cfg['model']['gcn']
        self.gcn_refiner = BatchedGCNRefiner(
            node_feat_dim=3 + 64,
            hidden_dim=gcn_cfg['hidden_dim'],
            num_layers=gcn_cfg['num_layers'],
            heads=gcn_cfg.get('heads', 4),
            dropout=gcn_cfg['dropout'],
            residual=gcn_cfg['residual']
        )

        # 5. Renderer
        self.renderer = build_renderer(
            image_size=cfg['renderer']['image_size'],
            device=dev
        )

        # 6. ArcFace (frozen, for ID loss)
        self.arcface = ArcFaceEncoder(
            weights_path=cfg['paths'].get('pretrained_arcface', '')
        )

        # 7. Intermediate feature-map hook 
        self._feat_map = None
        self._register_feature_hook()

    def _register_feature_hook(self):
        """Hook into backbone layer3 to capture intermediate feature map."""
        backbone_name = self.cfg['model']['backbone']
        if hasattr(self.encoder.backbone, 'layer3'):
            # ResNet
            def hook(module, input, output):
                self._feat_map = output
            self.encoder.backbone.layer3.register_forward_hook(hook)
            self.feat_map_channels = 1024  # ResNet layer3
        else:
            # EfficientNet / other — skip feature map sampling
            self._feat_map = None

    def forward(self, x: torch.Tensor) -> dict:
        """
        x : (B, 3, H, W) normalised input image
        Returns a dict with all intermediate and final outputs.
        """
        B = x.shape[0]

        # Step 1: CNN → coefficients
        coeffs, global_feat = self.encoder(x)    # coeffs dict, (B, feat_dim)

        # Step 2: BFM → coarse mesh
        coarse_verts, colors, faces = self.bfm(coeffs)
        # coarse_verts: (B, V, 3), colors: (B, V, 3), faces: (F, 3)

        # Step 3: Vertex feature sampling
        if self._feat_map is not None:
            # Resize feature map to spatial dims
            feat_map = F.interpolate(self._feat_map,
                                     size=(self.cfg['renderer']['image_size'] // 4,
                                           self.cfg['renderer']['image_size'] // 4),
                                     mode='bilinear', align_corners=False)
            # Down-project channels if needed
            if feat_map.shape[1] != self.feat_map_channels:
                feat_map = feat_map[:, :self.feat_map_channels]
            # Only sample up to 5k vertices for memory efficiency
            sample_v = min(self.n_verts, 5000)
            verts_sample = coarse_verts[:, :sample_v, :]
            vertex_feats = self.vertex_sampler(
                normalise_verts(verts_sample), feat_map
            )                                   # (B, sample_v, 64)
            # Pad remaining vertices with zeros
            if sample_v < self.n_verts:
                pad = torch.zeros(B, self.n_verts - sample_v, 64,
                                  device=x.device)
                vertex_feats = torch.cat([vertex_feats, pad], dim=1)
        else:
            vertex_feats = torch.zeros(B, self.n_verts, 64, device=x.device)

        # Step 4: GCN → refined mesh
        refined_verts = self.gcn_refiner(coarse_verts, vertex_feats, faces)
        # refined_verts: (B, V, 3)

        # Step 5: Render
        verts_norm = normalise_verts(refined_verts)
        render_out = self.renderer(verts_norm, faces, colors)
        if isinstance(render_out, dict):
            rendered_img  = render_out['image']
            silhouette    = render_out.get('silhouette', None)
        else:
            # Fallback renderer returns tensor directly
            rendered_img  = render_out
            silhouette    = None

        # Step 6: ArcFace identity embedding 
        rendered_resized = F.interpolate(rendered_img, size=(224, 224),
                                         mode='bilinear', align_corners=False)
        with torch.no_grad():
            id_embed_rendered = self.arcface(rendered_resized)

        x_resized = F.interpolate(x, size=(224, 224),
                                   mode='bilinear', align_corners=False)
        with torch.no_grad():
            id_embed_input = self.arcface(x_resized)

        return {
            # Coefficients
            'coeffs':         coeffs,
            # Meshes
            'coarse_verts':   coarse_verts,     # (B,V,3)
            'refined_verts':  refined_verts,    # (B,V,3)
            'colors':         colors,           # (B,V,3)
            'faces':          faces,            # (F,3)
            # Rendered outputs
            'rendered_img':   rendered_img,     # (B,3,H,W)
            'silhouette':     silhouette,       # (B,1,H,W) or None
            # Identity embeddings
            'id_embed_input':    id_embed_input,
            'id_embed_rendered': id_embed_rendered,
        }
