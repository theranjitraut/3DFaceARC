# 3DFaceARC
> **Self-supervised 3D face reconstruction from unannotated single-view images**  
> CNN Encoder → BFM2019 3DMM → GCN Mesh Refinement → Differentiable Rendering

---

## Project Structure

```
3DFaceARC/
├── configs/
│   └── config.yaml           # All hyperparameters and paths
├── data/
│   ├── dataset.py            # Unannotated dataset loader + MediaPipe pseudo-labels
│   ├── images/               # Face images (subdirectories per identity supported)
│   ├── BFM/
│   │   └── model2019_bfm.h5  # Basel Face Model 2019 (HDF5, not tracked in git)
│   └── pretrained/
│       └── model.onnx        # ArcFace identity model (ONNX, not tracked in git)
├── models/
│   ├── encoder.py            # ResNet50 CNN + 3DMM regressor + ArcFace (ONNX/PTH)
│   ├── bfm.py                # BFM loader (.h5 / .mat) + differentiable 3DMM layer
│   ├── gcn.py                # GCN/GAT mesh refinement network
│   ├── renderer.py           # Differentiable renderer (PyTorch3D / z-buffer fallback)
│   └── 3DFaceARC.py           # Full end-to-end model
├── utils/
│   ├── losses.py             # All self-supervised losses
│   ├── metrics.py            # PSNR, SSIM, NME, mesh regularity
│   └── visualise.py          # Grid plots, OBJ export, turntable
├── checkpoints/              # Saved model checkpoints (not tracked in git)
├── outputs/                  # Reconstruction visualisations (not tracked in git)
├── logs/                     # TensorBoard logs (not tracked in git)
├── train.py                  # Training entry point
├── inference.py              # Inference entry point
├── setup_and_demo.py         # Quick-start pipeline verification (no real data needed)
├── requirements.txt
└── .gitignore
```

---

## Quick Start

### 1. Create environment and install dependencies
```bash
python -m venv env
env\Scripts\activate          # Windows
pip install -r requirements.txt
```

> **GPU note**: Install PyTorch with CUDA first from https://pytorch.org/get-started/locally/  
> For GPU-accelerated ArcFace inference: `pip install onnxruntime-gpu` (instead of `onnxruntime`)

### 2. Verify the full setup
```bash
python setup_and_demo.py
```
Runs a complete end-to-end pipeline check on synthetic data. No real images or model files needed.

### 3. Place your data

**Face images** — flat folder or any subdirectory structure, both supported:
```
data/images/
    person_A/
        img001.jpg
        img002.png
    person_B/
        img001.jpg
    lone_face.jpg
```

**BFM 2019 model** (HDF5, recommended):
```
data/BFM/model2019_bfm.h5
```
Download from: https://faces.dmi.unibas.ch/bfm/bfm2019.html  
*(Falls back to a synthetic BFM automatically if not present.)*

**ArcFace identity model** (ONNX):
```
data/pretrained/model.onnx
```
Download `w600k_r50.onnx` from InsightFace model zoo: https://github.com/deepinsight/insightface/tree/master/model_zoo  
*(Falls back to random-init ResNet50 if not present — identity loss is noisier early in training.)*

### 4. Train
```bash
python train.py --config configs/config.yaml
```

Monitor training:
```bash
tensorboard --logdir logs/
```

### 5. Inference
```bash
# Single image
python inference.py \
    --input path/to/face.jpg \
    --checkpoint checkpoints/epoch_050_best.pt

# Entire folder
python inference.py \
    --input data/images/ \
    --checkpoint checkpoints/epoch_050_best.pt \
    --output_dir results/
```

---

## Pipeline Overview

```
Input RGB Image (224x224)
        |
        v
+-------------------+
|  ResNet50 Encoder |  <- pretrained ImageNet weights
|  CNN Backbone     |  -> global feat (2048-d) + layer3 feat map (1024-ch)
+--------+----------+
         |
         v
+-------------------+
|  MLP Regressor    |  -> shape (80), exp (64), tex (80),
|  3DMM Coeffs      |     angles (3), translation (3), scale (1)
+--------+----------+
         |
         v
+-------------------+
|  BFM2019 Decoder  |  -> coarse 3D mesh (47439 x 3), vertex colours
|  (3DMM Prior)     |     94464 triangular faces
+--------+----------+
         |  coarse mesh
         v
+-------------------+
|  Vertex Feature   |  <- projects verts onto CNN layer3 feature map
|  Sampler          |  -> per-vertex CNN features (V x 64)
+--------+----------+
         |
         v
+-------------------+
|  GAT-GCN Refiner  |  -> vertex displacements (V x 3)
|  4 layers, 256-d  |     4-head attention, residual connections
+--------+----------+
         |  refined mesh
         v
+-------------------+
|  Differentiable   |  -> rendered image (3 x 224 x 224)
|  Renderer         |     silhouette mask (1 x 224 x 224)
+--------+----------+
         |
         v
  Self-Supervised Losses (no 3D GT needed):
  [OK] Photometric     (rendered vs input pixels, L1)
  [OK] Perceptual      (VGG16 feature space, relu1-4)
  [OK] Pseudo-landmark (MediaPipe 68-pt -> MSE)
  [OK] Shape reg       (L2 penalty on 3DMM coefficients)
  [OK] Laplacian smooth(mesh smoothness / edge regularity)
  [OK] Face symmetry   (horizontal flip consistency)
  [OK] Identity        (ArcFace cosine similarity, ONNX)
```

---

## Performance Optimizations

3DFaceARC is heavily optimized to run on consumer GPUs (e.g., 6 GB VRAM) while processing high-resolution 47K-vertex meshes:
- **Persistent MediaPipe Sessions**: Eliminates the 3–5s TFLite initialization cost per image, massively accelerating the data loader.
- **Deterministic GCN Subsampling**: Processes a fixed, evenly-spaced stride subset of vertices (e.g., 6K-8K out of 47K) per batch. This allows PyTorch Geometric subgraph edge indices to be computed once and cached, eliminating 564K+ edge filtering operations per batch.
- **Topology Caching**: The Laplacian smoothness loss computes its face connectivity arrays (`idx_i`, `idx_j`, `deg`) once per mesh topology and caches them on the GPU.
- **AMP NaN Guards**: The VGG perceptual loss is explicitly cast to float32 to prevent half-precision overflow in BatchNorm layers, ensuring stable `GradScaler` updates without `NaN` poisoning.

---

## Configuration

Key settings in `configs/config.yaml`:

| Setting | Default | Description |
|---|---|---|
| `dataset.root_dir` | `./data/images` | Image folder (subdirs auto-discovered) |
| `dataset.image_size` | 224 | Input resolution |
| `bfm.model_path` | `./data/BFM/model2019_bfm.h5` | BFM2019 HDF5 path |
| `training.epochs` | 100 | Training epochs |
| `training.batch_size` | 6 | Batch size (tune based on VRAM) |
| `training.learning_rate` | 1e-4 | AdamW LR |
| `training.device` | `cuda` | `cuda` / `cpu` |
| `model.backbone` | `resnet50` | `resnet34` / `efficientnet_b3` |
| `model.gcn.num_layers` | 3 | GCN depth |
| `model.gcn.hidden_dim` | 128 | GCN hidden width |
| `model.gcn.max_verts`  | 6000 | Max vertices passed to GCN (controls VRAM) |
| `paths.pretrained_arcface` | `./data/pretrained/model.onnx` | ArcFace ONNX path |

---

## Hardware Requirements

| | Minimum | Recommended |
|---|---|---|
| GPU | GTX 1050 Ti (4 GB) | RTX 3060+ (8 GB+) |
| RAM | 8 GB | 16 GB |
| Storage | 10 GB free | 50 GB (for large datasets) |

---

## Outputs

After inference, you get per-image:
- `{name}_mesh.obj` — 3D mesh with vertex colours (open in Blender / MeshLab)
- `{name}_comparison.png` — Input | Rendered | Depth side-by-side
- `{name}_turntable/` — 24 rotation frames (360° view)

---

## References

- **BFM2019**: Basel Face Model 2019 — https://faces.dmi.unibas.ch/bfm/bfm2019.html
- **3DMM**: Blanz & Vetter, SIGGRAPH 1999
- **GAT**: Graph Attention Networks — Velickovic et al., ICLR 2018
- **ArcFace**: Deng et al., CVPR 2019 — https://github.com/deepinsight/insightface
- **PyTorch Geometric**: Fey & Lenssen, ICLR-W 2019
- **MediaPipe FaceMesh**: Kartynnik et al., 2019
- **PyTorch3D**: Johnson et al., 2020
