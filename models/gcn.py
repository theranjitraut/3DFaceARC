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

    node_feat_dim: xyz (3) + projected image features from CNN
    """

    def __init__(self, node_feat_dim: int, hidden_dim: int = 256,
                 num_layers: int = 4, heads: int = 4,
                 dropout: float = 0.1, residual: bool = True):
        super().__init__()
        self.use_pyg = PYG_AVAILABLE
        if PYG_AVAILABLE:
            self.gcn = GATRefinementNetwork(node_feat_dim, hidden_dim,
                                            num_layers, heads, dropout, residual)
        else:
            self.gcn = FallbackGCN(node_feat_dim, hidden_dim, out_dim=3,
                                   num_layers=num_layers, dropout=dropout)
        self._edge_cache = {}   # cache edge_index per face topology

    def _get_edge_index(self, faces: torch.Tensor) -> torch.Tensor:
        key = faces.shape[0]
        if key not in self._edge_cache:
            self._edge_cache[key] = faces_to_edge_index(faces)
        return self._edge_cache[key]

    def forward(self, verts: torch.Tensor, node_feats: torch.Tensor,
                faces: torch.Tensor) -> torch.Tensor:
        """
        verts      : (B, V, 3)
        node_feats : (B, V, F)   per-vertex features from CNN
        faces      : (Faces, 3)  same topology for all batch items
        returns    : refined_verts (B, V, 3)
        """
        B, V, _ = verts.shape
        edge_index = self._get_edge_index(faces).to(verts.device)

        # Concatenate xyz + features → (B, V, 3+F)
        combined = torch.cat([verts, node_feats], dim=-1)   # (B, V, 3+F)

        if self.use_pyg:
            # Build a PyG Batch for efficient processing
            graphs = []
            for b in range(B):
                graphs.append(GeoData(x=combined[b], edge_index=edge_index))
            batch = GeoBatch.from_data_list(graphs)
            delta = self.gcn(batch.x, batch.edge_index)    # (B*V, 3)
            delta = delta.view(B, V, 3)
        else:
            # Fallback: process each sample independently with shared adj matrix
            adj = build_adj_matrix(edge_index, V)
            deltas = []
            for b in range(B):
                d = self.gcn(combined[b], adj)              # (V, 3)
                deltas.append(d)
            delta = torch.stack(deltas, dim=0)              # (B, V, 3)

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
