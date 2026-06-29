

"""
models/gcn.py
GCN Mesh Refinement Network.
Supports three graph architectures switchable from config:
  - 'gcn'       : GCNConv (vanilla spectral GCN)
  - 'graphsage' : SAGEConv (inductive, aggregation-based)
  - 'gat'       : GATv2Conv (attention-based, original default)

Usage in config.yaml:
    model:
      gcn:
        arch: "gat"        # gcn | graphsage | gat
        num_layers: 3
        hidden_dim: 128
        dropout: 0.1
        heads: 2           # only used by gat
        residual: true
        max_verts: 10000
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# PyTorch Geometric import
try:
    from torch_geometric.nn import GATv2Conv, GCNConv, SAGEConv
    PYG_AVAILABLE = True
except ImportError:
    print("[GCN] torch_geometric not found. Using built-in fallback GCN.")
    PYG_AVAILABLE = False

# Graph construction utilities

def faces_to_edge_index(faces: torch.Tensor) -> torch.Tensor:
    """Convert face indices (F,3) → undirected edge_index (2, E)."""
    e01 = faces[:, [0, 1]]
    e12 = faces[:, [1, 2]]
    e20 = faces[:, [2, 0]]
    edges = torch.cat([e01, e12, e20], dim=0)
    edges = torch.cat([edges, edges.flip(1)], dim=0)   # make undirected
    edges = torch.unique(edges, dim=0)
    return edges.T.contiguous()                         # (2, E)


def build_adj_matrix(edge_index: torch.Tensor, n_verts: int) -> torch.Tensor:
    """
    Build sparse normalised adjacency for fallback GCN.
    D^{-1/2} A D^{-1/2}
    """
    i, j = edge_index
    idx  = torch.arange(n_verts, device=edge_index.device)
    i    = torch.cat([i, idx])
    j    = torch.cat([j, idx])
    deg  = torch.zeros(n_verts, device=edge_index.device)
    deg.scatter_add_(0, i, torch.ones(i.shape[0], device=edge_index.device))
    deg_inv_sqrt = deg.pow(-0.5).clamp(max=1e5)
    val  = deg_inv_sqrt[i] * deg_inv_sqrt[j]
    return torch.sparse_coo_tensor(
        torch.stack([i, j]), val, (n_verts, n_verts)
    ).coalesce()

# Fallback pure-PyTorch GCN (no torch_geometric)

class SimpleGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=False)
        self.bn  = nn.BatchNorm1d(out_dim)

    def forward(self, x, adj):
        x = torch.sparse.mm(adj, x)
        x = self.lin(x)
        x = self.bn(x)
        return F.elu(x)


class FallbackGCN(nn.Module):
    """Pure-PyTorch fallback when torch_geometric is unavailable."""

    def __init__(self, in_dim, hidden_dim, out_dim=3, num_layers=4, dropout=0.1):
        super().__init__()
        layers = [SimpleGCNLayer(in_dim, hidden_dim)]
        for _ in range(num_layers - 2):
            layers.append(SimpleGCNLayer(hidden_dim, hidden_dim))
        self.layers   = nn.ModuleList(layers)
        self.out_lin  = nn.Linear(hidden_dim, out_dim)
        self.dropout  = dropout

    def forward(self, x, adj):
        for layer in self.layers:
            x = layer(x, adj)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.out_lin(x)

# Architecture blocks (PyG-based)

class GCNBlock(nn.Module):
    """
    Vanilla GCNConv block.
    Simple, fast, good baseline. Best for small/regular meshes.
    No attention — treats all neighbours equally.
    """
    def __init__(self, in_dim, out_dim, dropout=0.1, residual=True):
        super().__init__()
        self.conv     = GCNConv(in_dim, out_dim)
        self.norm     = nn.LayerNorm(out_dim)
        self.act      = nn.ELU(inplace=True)
        self.residual = residual
        self.proj     = nn.Linear(in_dim, out_dim) if (in_dim != out_dim and residual) else None
        self.dropout  = dropout

    def forward(self, x, edge_index):
        h = self.conv(x, edge_index)
        h = F.dropout(h, p=self.dropout, training=self.training)
        if self.residual:
            res = self.proj(x) if self.proj is not None else x
            h   = h + res
        return self.act(self.norm(h))


class GraphSAGEBlock(nn.Module):
    """
    SAGEConv block.
    Inductive — aggregates mean of neighbours then concatenates with self.
    Good for large meshes and generalising to unseen faces.
    Usually faster than GAT with competitive quality.
    """
    def __init__(self, in_dim, out_dim, dropout=0.1, residual=True):
        super().__init__()
        self.conv     = SAGEConv(in_dim, out_dim)
        self.norm     = nn.LayerNorm(out_dim)
        self.act      = nn.ELU(inplace=True)
        self.residual = residual
        self.proj     = nn.Linear(in_dim, out_dim) if (in_dim != out_dim and residual) else None
        self.dropout  = dropout

    def forward(self, x, edge_index):
        h = self.conv(x, edge_index)
        h = F.dropout(h, p=self.dropout, training=self.training)
        if self.residual:
            res = self.proj(x) if self.proj is not None else x
            h   = h + res
        return self.act(self.norm(h))


class GATBlock(nn.Module):
    """
    GATv2Conv block.
    Attention-based — learns which neighbours matter more.
    Best quality but slowest. Good for complex/irregular meshes.
    heads must divide out_dim evenly.
    """
    def __init__(self, in_dim, out_dim, heads=4, dropout=0.1, residual=True):
        super().__init__()
        assert out_dim % heads == 0, \
            f"out_dim ({out_dim}) must be divisible by heads ({heads})"
        self.conv     = GATv2Conv(in_dim, out_dim // heads, heads=heads,
                                  dropout=dropout, concat=True)
        self.norm     = nn.LayerNorm(out_dim)
        self.act      = nn.ELU(inplace=True)
        self.residual = residual
        self.proj     = nn.Linear(in_dim, out_dim) if (in_dim != out_dim and residual) else None
        self.dropout  = dropout

    def forward(self, x, edge_index):
        h = self.conv(x, edge_index)
        h = F.dropout(h, p=self.dropout, training=self.training)
        if self.residual:
            res = self.proj(x) if self.proj is not None else x
            h   = h + res
        return self.act(self.norm(h))

# Unified Refinement Network (picks architecture from config)

ARCH_REGISTRY = {
    'gcn':       GCNBlock,
    'graphsage': GraphSAGEBlock,
    'gat':       GATBlock,
}


class MeshRefinementNetwork(nn.Module):
    """
    Unified mesh refinement network.
    Architecture selected by `arch` argument: 'gcn' | 'graphsage' | 'gat'

    Input:  per-vertex features (xyz + projected CNN features)
    Output: per-vertex displacements Δxyz
    """

    def __init__(self, in_dim: int, hidden_dim: int = 128,
                 num_layers: int = 3, arch: str = 'gat',
                 heads: int = 4, dropout: float = 0.1,
                 residual: bool = True):
        super().__init__()

        arch = arch.lower()
        if arch not in ARCH_REGISTRY:
            raise ValueError(
                f"Unknown GCN arch '{arch}'. Choose from: {list(ARCH_REGISTRY)}"
            )
        self.arch = arch
        print(f"[GCN] Using architecture: {arch.upper()}")

        BlockCls = ARCH_REGISTRY[arch]

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(inplace=True)
        )

        # Build layers
        layers = []
        for _ in range(num_layers):
            if arch == 'gat':
                layers.append(BlockCls(hidden_dim, hidden_dim,
                                       heads=heads, dropout=dropout,
                                       residual=residual))
            else:
                layers.append(BlockCls(hidden_dim, hidden_dim,
                                       dropout=dropout, residual=residual))
        self.layers = nn.ModuleList(layers)

        # Output head — initialised near-zero so training starts from 3DMM prior
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ELU(inplace=True),
            nn.Linear(64, 3)
        )
        nn.init.uniform_(self.output_head[-1].weight, -1e-4, 1e-4)
        nn.init.zeros_(self.output_head[-1].bias)

    def forward(self, node_feats: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        """
        node_feats : (V, in_dim)
        edge_index : (2, E)
        returns    : (V, 3) vertex displacements
        """
        h = self.input_proj(node_feats)
        for layer in self.layers:
            h = layer(h, edge_index)
        return self.output_head(h)

# Batched GCN Wrapper

class BatchedGCNRefiner(nn.Module):
    """
    Wraps MeshRefinementNetwork (or FallbackGCN) to handle a batch of meshes.

    max_verts : hard cap on vertices passed to GCN per batch item.
                When V > max_verts a deterministic stride-based subset is used
                and deltas are scattered back to full resolution.
                Set max_verts = target_verts in config to process full mesh.
    """

    def __init__(self, node_feat_dim: int, hidden_dim: int = 128,
                 num_layers: int = 3, arch: str = 'gat',
                 heads: int = 4, dropout: float = 0.1,
                 residual: bool = True, max_verts: int = 10000):
        super().__init__()
        self.use_pyg   = PYG_AVAILABLE
        self.max_verts = max_verts

        if PYG_AVAILABLE:
            self.gcn = MeshRefinementNetwork(
                node_feat_dim, hidden_dim, num_layers,
                arch=arch, heads=heads, dropout=dropout, residual=residual
            )
        else:
            self.gcn = FallbackGCN(
                node_feat_dim, hidden_dim, out_dim=3,
                num_layers=num_layers, dropout=dropout
            )

        # Caches — avoid recomputing topology every forward pass
        self._edge_cache:       dict = {}
        self._sub_idx_cache:    dict = {}
        self._batched_ei_cache: dict = {}

    # edge index helpers

    def _get_edge_index(self, faces: torch.Tensor,
                        sub_idx: torch.Tensor = None) -> torch.Tensor:
        if sub_idx is None:
            key = (int(faces.shape[0]), str(faces.device))
            if key not in self._edge_cache:
                self._edge_cache[key] = faces_to_edge_index(faces)
            return self._edge_cache[key]

        sub_key = (int(faces.shape[0]), int(sub_idx[0]), int(sub_idx[-1]),
                   str(faces.device))
        if sub_key not in self._edge_cache:
            V_full = int(faces.max()) + 1
            in_sub = torch.zeros(V_full, dtype=torch.bool, device=faces.device)
            in_sub[sub_idx] = True

            full_key = (int(faces.shape[0]), str(faces.device))
            if full_key not in self._edge_cache:
                self._edge_cache[full_key] = faces_to_edge_index(faces)
            ei = self._edge_cache[full_key]

            mask   = in_sub[ei[0]] & in_sub[ei[1]]
            ei_sub = ei[:, mask]

            remap = torch.full((V_full,), -1, dtype=torch.long,
                               device=faces.device)
            remap[sub_idx] = torch.arange(sub_idx.shape[0], device=faces.device)
            self._edge_cache[sub_key] = remap[ei_sub].contiguous()

        return self._edge_cache[sub_key]

    def _get_sub_idx(self, V: int, device: torch.device) -> torch.Tensor:
        key = (V, str(device))
        if key not in self._sub_idx_cache:
            self._sub_idx_cache[key] = torch.linspace(
                0, V - 1, self.max_verts, dtype=torch.long, device=device
            )
        return self._sub_idx_cache[key]

    # forward

    def forward(self, verts: torch.Tensor, node_feats: torch.Tensor,
                faces: torch.Tensor) -> torch.Tensor:
        """
        verts      : (B, V, 3)
        node_feats : (B, V, F)
        faces      : (F_faces, 3)
        returns    : refined_verts (B, V, 3)
        """
        B, V, _ = verts.shape

        # vertex subsampling
        if V > self.max_verts:
            sub_idx    = self._get_sub_idx(V, verts.device)
            verts_sub  = verts[:, sub_idx, :]
            feats_sub  = node_feats[:, sub_idx, :]
            edge_index = self._get_edge_index(faces, sub_idx)
            subsample  = True
        else:
            verts_sub  = verts
            feats_sub  = node_feats
            edge_index = self._get_edge_index(faces)
            sub_idx    = None
            subsample  = False

        Vs       = verts_sub.shape[1]
        combined = torch.cat([verts_sub, feats_sub], dim=-1)  # (B, Vs, 3+F)

        if self.use_pyg:
            # Batch all graphs into one large disconnected graph
            batched_key = (B, Vs, str(verts.device))
            if batched_key not in self._batched_ei_cache:
                self._batched_ei_cache[batched_key] = torch.cat(
                    [edge_index + b * Vs for b in range(B)], dim=1
                )
            b_edge_index = self._batched_ei_cache[batched_key]

            delta_sub = self.gcn(
                combined.view(B * Vs, -1), b_edge_index
            ).view(B, Vs, 3)
        else:
            adj = build_adj_matrix(edge_index, Vs)
            delta_sub = torch.stack(
                [self.gcn(combined[b], adj) for b in range(B)], dim=0
            )

        # Scatter deltas back to full resolution
        if subsample:
            delta = torch.zeros(B, V, 3,
                                device=verts.device, dtype=delta_sub.dtype)
            delta[:, sub_idx, :] = delta_sub
        else:
            delta = delta_sub

        return verts + delta

# CNN feature → per-vertex feature projection

class VertexFeatureSampler(nn.Module):
    """
    Projects 3D vertices onto the image plane and samples CNN feature map
    values at each vertex location via bilinear interpolation.
    Injects image evidence into the graph as node features.
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
        verts    : (B, V, 3)   3D vertices in camera space
        feat_map : (B, C, H, W)
        returns  : (B, V, out_channels)
        """
        B, C, H, W = feat_map.shape

        # Orthographic projection: normalise xy to [-1, 1]
        xy   = verts[:, :, :2].clone()
        xy   = xy / (xy.abs().max(dim=1, keepdim=True).values + 1e-8)
        grid = xy.unsqueeze(2)   # (B, V, 1, 2)

        sampled = F.grid_sample(feat_map, grid, align_corners=True,
                                mode='bilinear', padding_mode='border')
        # (B, C, V, 1) → (B, V, C)
        sampled = sampled.squeeze(-1).permute(0, 2, 1).contiguous()
        return self.proj(sampled)

# Architecture comparison summary (for reference)
"""
ARCHITECTURE COMPARISON
arch        │ Speed   │ Memory │ Quality │ Best for
gcn         │ Fast    │ Low    │ Good    │ Baseline, ablation
graphsage   │ Medium  │ Medium │ Better  │ Large meshes, inductive
gat         │ Slow    │ High   │ Best    │ Complex faces, fine detail

CONFIG EXAMPLE:
    model:
      gcn:
        arch: "gat"        # change to gcn | graphsage | gat
        num_layers: 3
        hidden_dim: 128
        dropout: 0.1
        heads: 2           # only applies to gat
        residual: true
        max_verts: 10000
"""