# 3DFaceARC: Architecture & Technical Walkthrough

Welcome to the comprehensive technical documentation for **3DFaceARC** (Face Analysis and Reconstruction Components System). This document provides an in-depth breakdown of the system architecture, data flow, models, self-supervised losses, training procedures, and inference outputs, reflecting the **verified, working state** of the project.

---

## 1. System Overview

**3DFaceARC** is a self-supervised deep learning framework for **3D face reconstruction from unannotated single-view RGB images**. Unlike traditional supervised methods requiring expensive ground-truth 3D scans, 3DFaceARC combines:

- A statistical **3D Morphable Model (3DMM)** prior — the **Basel Face Model 2019** (47,439 vertices, 94,464 faces)
- A **CNN encoder** (ResNet50, pretrained ImageNet) that regresses 3DMM coefficients from raw images
- A **Graph Attention Network (GAT)** that refines coarse mesh geometry using local CNN features
- A **differentiable renderer** that projects the 3D face back to 2D for self-supervised loss computation
- A frozen **ArcFace identity model** (ONNX) that enforces identity consistency between input and reconstruction

### Full Pipeline Diagram
```
Input RGB Image (224x224)
        |
        v
+----------------------------------+
|       ResNet50 CNN Encoder       |  <- pretrained ImageNet weights
|   layer3 feature map (1024-ch)   |  -> global feat: (B, 2048)
|   global avg pool -> FC          |     feat map:    (B, 1024, H, W)
+--------+-------------------------+
         |
         v (global feat)
+----------------------------------+
|     CoeffRegressor (MLP)         |
|   Linear(2048->1024->512->total) |  -> shape (80), exp (64), tex (80),
|   BatchNorm + ELU + Dropout      |     angles (3), translation (3), scale (1)
+--------+-------------------------+
         |
         v
+----------------------------------+
|    BFMLayer — BFM2019 Decoder    |  <- model2019_bfm.h5 (290 MB HDF5)
|   47,439 vertices, 94,464 faces  |  -> coarse verts: (B, 47439, 3)
|   shape + expression + texture   |     colors:       (B, 47439, 3)
+--------+-------------------------+
         |  coarse mesh
         v
+----------------------------------+
|    VertexFeatureSampler          |  <- hooks into ResNet layer3 output
|    F.grid_sample bilinear        |  -> per-vertex feats: (B, 47439, 64)
+--------+-------------------------+
         |
         v
+----------------------------------+
|    BatchedGCNRefiner (GAT)       |  4 layers, hidden=256, heads=4
|    GATBlock x4 + residual        |  -> delta xyz: (B, 47439, 3)
+--------+-------------------------+
         |  refined_verts = coarse + delta
         v
+----------------------------------+
|    Differentiable Renderer       |  PyTorch3D or fallback z-buffer
|    orthographic projection       |  -> rendered_img:  (B, 3, 224, 224)
|    vertex colour shading         |     silhouette:    (B, 1, 224, 224)
+--------+-------------------------+
         |
         v
+----------------------------------+
|    Frozen ArcFace (ONNX)         |  model.onnx (174 MB, 112x112 input)
|    onnxruntime inference         |  -> id_embed: (B, 512) L2-normalised
+----------------------------------+
         |
         v
   Self-Supervised Losses (no 3D annotations needed)
```

---

## 2. Data Handling & Preprocessing Pipeline

**Location**: [data/dataset.py](file:///c:/Users/user 22/Desktop/3DFaceARC/data/dataset.py)

No pre-computed annotations are required. The dataset module discovers all images **recursively** from the configured root directory, supporting any folder structure (flat or per-identity subdirectories).

### Verified Dataset Stats (current run)
- **17,532 total images** across **105 identity subfolders**
- Format: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`
- No labels, no 3D ground truth, no special naming required

### Key Components

#### [MediaPipeLandmarkExtractor](file:///c:/Users/user 22/Desktop/3DFaceARC/data/dataset.py#L18)
Extracts 68-point pseudo-landmarks on-the-fly using Google MediaPipe FaceMesh (468-point model).
- `MEDIAPIPE_68_MAPPING`: Maps a fixed 68-index subset to the standard IBUG/Dlib scheme.
- Landmarks are cached in `_landmark_cache` to avoid recomputation across epochs.
- Used as weak geometric supervision in the landmark loss.

#### [FaceDataset](file:///c:/Users/user 22/Desktop/3DFaceARC/data/dataset.py#L89)
PyTorch `Dataset` that:
1. Recursively scans `root_dir` with `rglob('*')` — finds images in any subfolder depth.
2. Detects & crops the primary face using MediaPipe FaceDetection (10% padding), falls back to 80% centre crop. **Note:** A persistent MediaPipe session is maintained across the dataset to avoid the massive 3-5s TFLite initialization overhead per image.
3. Returns per-sample dict:
   - `image`: Augmented + ImageNet-normalised tensor `(3, 224, 224)`
   - `image_raw`: Unaugmented `[0,1]` tensor — target for photometric loss
   - `landmarks`: Normalised `(68, 2)` pseudo-landmarks in `[-1, 1]`
   - `path`: Source file string

#### [build_dataloaders](file:///c:/Users/user 22/Desktop/3DFaceARC/data/dataset.py#L198)
Splits dataset into Train / Val / Test (85% / 10% / 5% by default). Augmentation (colour jitter, horizontal flip, random grayscale) is applied only to the training split.

---

## 3. Model Components

**Integration point**: [models/3DFaceARC.py](file:///c:/Users/user 22/Desktop/3DFaceARC/models/3DFaceARC.py) — [3DFaceARC](file:///c:/Users/user 22/Desktop/3DFaceARC/models/3DFaceARC.py#L17)

**Verified trainable parameter count: 26,883,306**

### 3.1 CNN Encoder & Regressor — [models/encoder.py](file:///c:/Users/user 22/Desktop/3DFaceARC/models/encoder.py)

**[FaceEncoder](file:///c:/Users/user 22/Desktop/3DFaceARC/models/encoder.py#L59)**
- Backbone: `resnet50` (default), `resnet34`, or `efficientnet_b3`
- A forward hook on `backbone.layer3` captures the intermediate feature map `(B, 1024, H, W)` for use by the vertex sampler
- Global average pooling output `(B, 2048)` feeds the regressor

**[CoeffRegressor](file:///c:/Users/user 22/Desktop/3DFaceARC/models/encoder.py#L15)** — MLP head:
```
Linear(2048, 1024) -> BN -> ELU -> Dropout(0.3)
Linear(1024, 512)  -> BN -> ELU -> Dropout(0.2)
Linear(512, total_coeffs)
```
Coefficient outputs with constraints:
| Name | Dim | Constraint |
|---|---|---|
| `shape` | 80 | raw |
| `exp` | 64 | raw |
| `tex` | 80 | raw |
| `angles` | 3 | `* 0.3` radians |
| `translation` | 3 | `* 0.1` |
| `scale` | 1 | `sigmoid * 2.0 + 0.5` → [0.5, 2.5] |

### 3.2 Basel Face Model 2019 — [models/bfm.py](file:///c:/Users/user 22/Desktop/3DFaceARC/models/bfm.py)

**File support** (auto-detected by extension):
- `.h5` / `.hdf5` — **BFM2019** (current, recommended): 47,439 verts, 94,464 faces
- `.mat` — BFM09 legacy format
- Falls back to a synthetic random BFM if no file is present

**BFM2019 HDF5 structure used:**
```
shape/model/mean          (142317,)   <- mean shape
shape/model/pcaBasis      (142317, 199)
shape/model/pcaVariance   (199,)
expression/model/pcaBasis (142317, 100)
color/model/pcaBasis      (142317, 199)
shape/representer/cells   (3, 94464)  <- transposed to (94464, 3)
```

**[BFMLayer](file:///c:/Users/user 22/Desktop/3DFaceARC/models/bfm.py#L89)** — differentiable 3DMM decoder:

$$\text{Shape}(\alpha, \beta) = \mu_{shape} + U_{shape}(\alpha \odot \sigma_{shape}) + U_{exp}(\beta \odot \sigma_{exp})$$
$$\text{Texture}(\gamma) = \mu_{tex} + U_{tex}(\gamma \odot \sigma_{tex})$$

All basis matrices are registered as non-trainable `nn.Module` buffers for fast GPU matmul.

### 3.3 GCN Mesh Refinement — [models/gcn.py](file:///c:/Users/user 22/Desktop/3DFaceARC/models/gcn.py)

**[VertexFeatureSampler](file:///c:/Users/user 22/Desktop/3DFaceARC/models/gcn.py#L221)**
- Projects 3D vertices (orthographic) → normalised 2D grid coordinates `[-1, 1]`
- Uses `F.grid_sample` (bilinear) to sample the CNN layer3 feature map at each vertex location
- Projects 1024 CNN channels → 64-d per-vertex features via `Linear + LayerNorm + ELU`
- Processes up to 5,000 vertices per forward pass (memory-efficient); remainder padded with zeros

**[BatchedGCNRefiner](file:///c:/Users/user 22/Desktop/3DFaceARC/models/gcn.py#L160)**
- Node features: `3 (xyz) + 64 (CNN) = 67-d` input
- **With PyTorch Geometric** (installed): [GATRefinementNetwork](file:///c:/Users/user 22/Desktop/3DFaceARC/models/gcn.py#L114) — multi-head Graph Attention layers (3 layers, 128-d hidden, 2 heads, residual + dropout). To stay within VRAM limits (e.g. 6GB), a deterministic stride-based subset of vertices (max 6,000) is passed to the GCN. The subset is fixed, which allows PyG's subgraph edge indexing to be cached and dramatically speeds up the forward pass.
- **Fallback** (no PyG): symmetric normalized adjacency GCN in native PyTorch
- Output: `(B, V, 3)` displacement vectors added to coarse vertices (unsampled vertices receive zero displacement).

### 3.4 Differentiable Renderer — [models/renderer.py](file:///c:/Users/user 22/Desktop/3DFaceARC/models/renderer.py)

- **[PyTorch3DRenderer](file:///c:/Users/user 22/Desktop/3DFaceARC/models/renderer.py#L68)**: Full SoftPhongShader + SoftSilhouetteShader rendering (requires `pytorch3d` package)
- **[FallbackRenderer](file:///c:/Users/user 22/Desktop/3DFaceARC/models/renderer.py#L29)**: Custom differentiable z-buffer painter's algorithm in native PyTorch — **active in the current configuration**

### 3.5 ArcFace Identity Encoder — [models/encoder.py](file:///c:/Users/user 22/Desktop/3DFaceARC/models/encoder.py#L114)

**Fully frozen** during training (no gradient updates).

**[ArcFaceEncoder](file:///c:/Users/user 22/Desktop/3DFaceARC/models/encoder.py#L114)** supports two formats:

| Format | File | Loader |
|---|---|---|
| `.onnx` | `data/pretrained/model.onnx` | `onnxruntime.InferenceSession` |
| `.pth` | any `.pth` state dict | `torchvision.models.resnet50` |

Current configuration uses **ONNX** (`model.onnx`, 174 MB, InsightFace ArcFace R50):
- Input: `(B, 3, 112, 112)` — images auto-resized internally
- Output: `(B, 512)` L2-normalised identity embedding
- Runs on CPU via onnxruntime (GPU version: `pip install onnxruntime-gpu`)

---

## 4. Self-Supervised Loss Formulation

**Location**: [utils/losses.py](file:///c:/Users/user 22/Desktop/3DFaceARC/utils/losses.py)

$$L_{total} = w_1 L_{photo} + w_2 L_{perceptual} + w_3 L_{landmark} + w_4 L_{reg} + w_5 L_{smooth} + w_6 L_{sym} + w_7 L_{id}$$

| Loss | Weight | Description |
|---|---|---|
| `photometric` | 1.0 | L1 pixel loss between rendered and input (masked by silhouette) |
| `perceptual` | 0.3 | VGG16 feature difference at relu1_2, relu2_2, relu3_3, relu4_3 |
| `landmark` | 0.5 | MSE between projected 3D landmarks and MediaPipe pseudo-labels |
| `shape_reg` | 0.1 | L2 penalty on shape/exp/tex coefficients (keeps faces near prior) |
| `smooth` | 0.05 | Laplacian mesh smoothness — penalises sharp vertex spikes |
| `symmetry` | 0.2 | L1 difference between rendered image and its horizontal flip |
| `id_preserve` | 0.5 | Cosine distance between ArcFace embeddings of input and rendered face |

### Key Loss Details

**Photometric**: $L_{photo} = \frac{1}{N}\sum |(I_{render} - I_{raw}) \odot M|$ where $M$ is the silhouette mask.

**Perceptual Loss (VGG)**: Computes features at intermediate layers. Note that this always runs in float32, ignoring AMP contexts, to prevent `NaN` overflow in VGG's `BatchNorm` layers.

**Laplacian Smoothness**: $L_{smooth} = \frac{1}{V}\sum_{i} \|v_i - \frac{1}{|N(i)|}\sum_{j \in N(i)} v_j\|^2_2$. The connectivity arrays (`idx_i`, `idx_j`, `deg`) are cached in memory keyed by face topology to save huge per-batch computation costs.

**Identity Preservation**: $L_{id} = 1 - \cos(\text{emb}_{input}, \text{emb}_{render})$

---

## 5. Training Pipeline — [train.py](file:///c:/Users/user 22/Desktop/3DFaceARC/train.py)

- **Optimizer**: AdamW (lr=1e-4, weight_decay=1e-5)
- **Scheduler**: Cosine decay with 5-epoch linear warmup
- **Mixed Precision**: `torch.cuda.amp.GradScaler` + `autocast` (CUDA only)
- **Gradient clipping**: max norm = 1.0
- **Checkpointing**: every 5 epochs + best validation loss
- **Logging**: TensorBoard — run `tensorboard --logdir logs/`

### Data Split (default)
| Split | Fraction | Images (17,532 total) |
|---|---|---|
| Train | 85% | ~14,902 |
| Val | 10% | ~1,753 |
| Test | 5% | ~877 |

---

## 6. Inference Pipeline — [inference.py](file:///c:/Users/user 22/Desktop/3DFaceARC/inference.py)

1. **Preprocessing**: MediaPipe face crop → resize to 224×224 → ImageNet normalise
2. **Forward pass**: `3DFaceARC` under `torch.no_grad()`
3. **Outputs per image**:
   - `{name}_mesh.obj` — 3D mesh with vertex colours (Blender / MeshLab compatible)
   - `{name}_comparison.png` — Input | Rendered | Depth map
   - `{name}_turntable/` — 24-frame 360° rotation sequence

---

## 7. Verified Runtime Configuration

Tested and passing as of project setup:

| Item | Value |
|---|---|
| Python | 3.10.11 |
| PyTorch | 2.12.0 |
| Device | CUDA (GPU) |
| BFM | BFM2019, model2019_bfm.h5, 47,439 verts |
| ArcFace | model.onnx (InsightFace R50, 512-d), onnxruntime CPU |
| torch-geometric | Installed (GAT active) |
| pytorch3d | Not installed (fallback z-buffer renderer active) |
| Dataset | 17,532 images, 105 identity subfolders |
| Trainable params | 26,883,306 |

---

## 8. References

- **BFM2019**: Basel Face Model 2019 — https://faces.dmi.unibas.ch/bfm/bfm2019.html
- **3DMM**: Blanz & Vetter, SIGGRAPH 1999
- **GAT**: Graph Attention Networks — Velickovic et al., ICLR 2018
- **ArcFace**: Deng et al., CVPR 2019 — https://github.com/deepinsight/insightface
- **PyTorch Geometric**: Fey & Lenssen, ICLR-W 2019
- **MediaPipe FaceMesh**: Kartynnik et al., 2019
- **PyTorch3D**: Johnson et al., 2020
