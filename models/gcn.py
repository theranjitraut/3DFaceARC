"""
models/gcn.py
GCN Mesh Refinement Network.
Takes a coarse 3DMM mesh + per-vertex CNN features and predicts
vertex displacements to produce a fine-grained 3D face mesh.

Uses PyTorch Geometric (torch_geometric) for graph convolutions.
Falls back to a pure-PyTorch approximate GCN if torch_geometric is absent.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Try importing torch_geometric; fall back gracefully
try:
    from torch_geometric.nn import GATv2Conv, GCNConv, global_mean_pool
    from torch_geometric.data import Data as GeoData, Batch as GeoBatch
    PYG_AVAILABLE = True
except ImportError:
    print("[GCN] torch_geometric not found. Using built-in approx GCN.")
    PYG_AVAILABLE = False

# Graph construction utilities
def faces_to_edge_index(faces: torch.Tensor) -> torch.Tensor:
    """
    Convert face indices (F,3) to undirected edge_index (2, E).
    """
    e01 = faces[:, [0, 1]]
    e12 = faces[:, [1, 2]]
    e20 = faces[:, [2, 0]]
    edges = torch.cat([e01, e12, e20], dim=0)           # (E, 2)
    edges = torch.cat([edges, edges.flip(1)], dim=0)    # undirected
    edges = torch.unique(edges, dim=0)                  # deduplicate
    return edges.T.contiguous()                         # (2, E)


def build_adj_matrix(edge_index: torch.Tensor, n_verts: int) -> torch.Tensor:
    """
    Build sparse normalised adjacency matrix for the fallback GCN.
    D^{-1/2} A D^{-1/2}  (symmetric normalisation)
    """
    i, j = edge_index
    # Add self-loops
    idx = torch.arange(n_verts, device=edge_index.device)
    i   = torch.cat([i, idx])
    j   = torch.cat([j, idx])

    # Degree
    deg = torch.zeros(n_verts, device=edge_index.device)
    deg.scatter_add_(0, i, torch.ones(i.shape[0], device=edge_index.device))
    deg_inv_sqrt = deg.pow(-0.5).clamp(max=1e5)

    val = deg_inv_sqrt[i] * deg_inv_sqrt[j]
    adj = torch.sparse_coo_tensor(
        torch.stack([i, j]), val, (n_verts, n_verts)
    ).coalesce()
    return adj

# Fallback pure-PyTorch GCN layer
class SimpleGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=False)
        self.bn  = nn.BatchNorm1d(out_dim)

    def forward(self, x, adj):
        # adj: sparse (V,V), x: (V, C)
        x = torch.sparse.mm(adj, x)
        x = self.lin(x)
        x = self.bn(x)
        return F.elu(x)


class FallbackGCN(nn.Module):
    """Pure-PyTorch GCN — used when torch_geometric is unavailable."""

    def __init__(self, in_dim, hidden_dim, out_dim=3, num_layers=4, dropout=0.1):
        super().__init__()
        layers = [SimpleGCNLayer(in_dim, hidden_dim)]
        for _ in range(num_layers - 2):
            layers.append(SimpleGCNLayer(hidden_dim, hidden_dim))
        self.layers  = nn.ModuleList(layers)
        self.out_lin = nn.Linear(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x, adj):
        for layer in self.layers:
            x = layer(x, adj)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.out_lin(x)


# PyG GAT-based GCN (preferred)
class GATBlock(nn.Module):
    def __init__(self, in_dim, out_dim, heads=4, dropout=0.1, residual=True):
        super().__init__()
        assert out_dim % heads == 0, "out_dim must be divisible by heads"
        self.conv = GATv2Conv(in_dim, out_dim // heads, heads=heads,
                              dropout=dropout, concat=True)
        self.norm = nn.LayerNorm(out_dim)
        self.act  = nn.ELU(inplace=True)
        self.residual = residual
        self.proj = nn.Linear(in_dim, out_dim) if (in_dim != out_dim and residual) else None

    def forward(self, x, edge_index):
        h = self.conv(x, edge_index)
        if self.residual:
            res = self.proj(x) if self.proj is not None else x
            h = h + res
        return self.act(self.norm(h))


class GATRefinementNetwork(nn.Module):
    """
    GAT-based mesh refinement.
    Input:  per-vertex features (xyz + projected CNN features)
    Output: per-vertex displacements Δxyz
    """

    def __init__(self, in_dim: int, hidden_dim: int = 256,
                 num_layers: int = 4, heads: int = 4,
                 dropout: float = 0.1, residual: bool = True):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(inplace=True)
        )

        self.gat_layers = nn.ModuleList([
            GATBlock(hidden_dim, hidden_dim, heads=heads,
                     dropout=dropout, residual=residual)
            for _ in range(num_layers)
        ])

        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ELU(inplace=True),
            nn.Linear(64, 3)   # predict Δx, Δy, Δz
        )
        # Initialise output to near-zero so training starts from 3DMM prior
        nn.init.uniform_(self.output_head[-1].weight, -1e-4, 1e-4)
        nn.init.zeros_(self.output_head[-1].bias)

    def forward(self, node_feats: torch.Tensor, edge_index: torch.Tensor):
        """
        node_feats : (V, in_dim)
        edge_index : (2, E)
        returns    : (V, 3) vertex displacements
        """
        h = self.input_proj(node_feats)
        for layer in self.gat_layers:
            h = layer(h, edge_index)
        return self.output_head(h)


# Batched GCN wrapper (handles B images in one forward pass via PyG Batch)
class BatchedGCNRefiner(nn.Module):
    """
    Wraps GATRefinementNetwork (or FallbackGCN) to handle a batch of meshes.

    node_feat_dim : xyz (3) + projected image features from CNN
    max_verts     : max vertices passed to GCN per batch item.
                    When V > max_verts a deterministic stride-based subset is processed
                    and the predicted deltas are scattered back (zeros for
                    unsampled vertices). Keeps graph size manageable on
                    low-VRAM GPUs (6 GB) even for the 47 K-vertex BFM mesh.
    """

    def __init__(self, node_feat_dim: int, hidden_dim: int = 128,
                 num_layers: int = 3, heads: int = 4,
                 dropout: float = 0.1, residual: bool = True,
                 max_verts: int = 8192):
        super().__init__()
        self.use_pyg   = PYG_AVAILABLE
        self.max_verts = max_verts
        if PYG_AVAILABLE:
            self.gcn = GATRefinementNetwork(node_feat_dim, hidden_dim,
                                            num_layers, heads, dropout, residual)
        else:
            self.gcn = FallbackGCN(node_feat_dim, hidden_dim, out_dim=3,
                                   num_layers=num_layers, dropout=dropout)
        # cache edge_index per (n_faces, device_str) pair
        self._edge_cache:     dict = {}
        # cache fixed vertex subsets per (V, device_str)
        self._sub_idx_cache:  dict = {}
        # cache batched edge_indices per (B, Vs, device_str)
        self._batched_ei_cache: dict = {}

    def _get_edge_index(self, faces: torch.Tensor,
                        sub_idx: torch.Tensor | None = None) -> torch.Tensor:
        """
        Build / retrieve cached edge_index.
        If sub_idx is provided build a *local* edge_index for just those verts.
        """
        if sub_idx is None:
            key = (int(faces.shape[0]), str(faces.device))
            if key not in self._edge_cache:
                self._edge_cache[key] = faces_to_edge_index(faces)
            return self._edge_cache[key]

        # --- subsampled edge_index ---
        # key includes the subset's first/last element so different subsets
        # don't collide; since sub_idx is now fixed this always hits.
        sub_key = (int(faces.shape[0]), int(sub_idx[0]), int(sub_idx[-1]),
                   str(faces.device))
        if sub_key not in self._edge_cache:
            V_full = int(faces.max()) + 1
            in_sub = torch.zeros(V_full, dtype=torch.bool, device=faces.device)
            in_sub[sub_idx] = True

            full_key = (int(faces.shape[0]), str(faces.device))
            if full_key not in self._edge_cache:
                self._edge_cache[full_key] = faces_to_edge_index(faces)
            ei = self._edge_cache[full_key]          # (2, E_full)

            mask   = in_sub[ei[0]] & in_sub[ei[1]]
            ei_sub = ei[:, mask]                     # (2, E_sub) – global

            remap = torch.full((V_full,), -1, dtype=torch.long,
                               device=faces.device)
            remap[sub_idx] = torch.arange(sub_idx.shape[0], device=faces.device)
            self._edge_cache[sub_key] = remap[ei_sub].contiguous()

        return self._edge_cache[sub_key]

    def _get_sub_idx(self, V: int, device: torch.device) -> torch.Tensor:
        """
        Return a fixed, deterministic vertex subset of size min(V, max_verts).

        Uses evenly-spaced stride indices (NOT random) so that:
          • the same subset is used every batch → subgraph edge_index is
            computed once and cached, never rebuilt.
          • training still sees a good spatial spread of the mesh.
        """
        key = (V, str(device))
        if key not in self._sub_idx_cache:
            self._sub_idx_cache[key] = torch.linspace(
                0, V - 1, self.max_verts, dtype=torch.long, device=device
            )
        return self._sub_idx_cache[key]

    def forward(self, verts: torch.Tensor, node_feats: torch.Tensor,
                faces: torch.Tensor) -> torch.Tensor:
        """
        verts      : (B, V, 3)
        node_feats : (B, V, F)   per-vertex features from CNN
        faces      : (Faces, 3)  same topology for all batch items
        returns    : refined_verts (B, V, 3)
        """
        B, V, _ = verts.shape

        # ── vertex subsampling ──────────────────────────────────────────────
        if V > self.max_verts:
            # Fixed deterministic subset — same indices every batch so the
            # subgraph edge_index is computed once and cached.
            sub_idx   = self._get_sub_idx(V, verts.device)
            verts_sub = verts[:, sub_idx, :]      # (B, Vs, 3)
            feats_sub = node_feats[:, sub_idx, :] # (B, Vs, F)
            edge_index = self._get_edge_index(faces, sub_idx)
            subsample  = True
        else:
            verts_sub  = verts
            feats_sub  = node_feats
            edge_index = self._get_edge_index(faces)
            sub_idx    = None
            subsample  = False
        # ────────────────────────────────────────────────────────────────────

        Vs = verts_sub.shape[1]
        combined = torch.cat([verts_sub, feats_sub], dim=-1)  # (B, Vs, 3+F)

        if self.use_pyg:
            # Create or retrieve batched edge_index
            batched_key = (B, Vs, str(verts.device))
            if batched_key not in self._batched_ei_cache:
                batched_ei = []
                for b in range(B):
                    batched_ei.append(edge_index + b * Vs)
                self._batched_ei_cache[batched_key] = torch.cat(batched_ei, dim=1)
            
            b_edge_index = self._batched_ei_cache[batched_key]
            
            # Flatten batch for GAT processing
            flattened_x = combined.view(B * Vs, -1)
            delta_sub = self.gcn(flattened_x, b_edge_index)
            delta_sub = delta_sub.view(B, Vs, 3)
        else:
            adj = build_adj_matrix(edge_index, Vs)
            delta_sub = torch.stack(
                [self.gcn(combined[b], adj) for b in range(B)], dim=0
            )  # (B, Vs, 3)

        if subsample:
            # Create a fresh zeros tensor (no gradient history) and scatter
            # the GCN deltas into the sampled positions.
            # delta[:, sub_idx, :] = delta_sub is differentiable here because
            # delta has no prior autograd history.
            delta = torch.zeros(B, V, 3, device=verts.device, dtype=delta_sub.dtype)
            delta[:, sub_idx, :] = delta_sub
        else:
            delta = delta_sub

        return verts + delta


# CNN feature → per-vertex feature projection
class VertexFeatureSampler(nn.Module):
    """
    Projects 3D vertices onto the image plane and samples CNN feature map
    values at each vertex location (bilinear sampling).
    This injects image evidence into the graph as node features.
    """

    def __init__(self, feat_channels: int, out_channels: int = 64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(feat_channels, out_channels),
            nn.LayerNorm(out_channels),
            nn.ELU(inplace=True)
        )

    def forward(self, verts: torch.Tensor,
                feat_map: torch.Tensor,
                intrinsic: torch.Tensor = None) -> torch.Tensor:
        """
        verts    : (B, V, 3)   3D vertices (camera space)
        feat_map : (B, C, H, W) intermediate CNN feature map
        intrinsic: (B, 3, 3)   optional camera matrix; if None uses orthographic proj
        returns  : (B, V, out_channels)
        """
        B, C, H, W = feat_map.shape
        V = verts.shape[1]

        # Orthographic projection: normalise x,y to [-1,1]
        xy = verts[:, :, :2].clone()                    # (B, V, 2)
        # Assume verts are already in roughly image-scale; normalise
        xy = xy / (xy.abs().max(dim=1, keepdim=True).values + 1e-8)
        grid = xy.unsqueeze(2)                          # (B, V, 1, 2)

        # Sample feat_map at each vertex location
        sampled = F.grid_sample(feat_map, grid, align_corners=True,
                                mode='bilinear', padding_mode='border')
        # sampled: (B, C, V, 1) → (B, V, C)
        sampled = sampled.squeeze(-1).permute(0, 2, 1).contiguous()

        return self.proj(sampled)                       # (B, V, out_channels)
