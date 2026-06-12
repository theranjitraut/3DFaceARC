# 3DFaceARC
> **Self-supervised 3D face reconstruction from unannotated single-view images**  
> CNN Encoder → 3DMM → GCN Mesh Refinement → Differentiable Rendering

---
## Structure

```
FaceARCs/
├── configs/
│   └── config.yaml           # All hyperparameters and paths
├── data/
│   ├── dataset.py            # Unannotated dataset loader + MediaPipe pseudo-labels
│   ├── images/               # ← Put your face images here (any structure)
│   └── BFM/                  # ← Put BFM09_model_info.mat here (optional)
├── models/
│   ├── encoder.py            # ResNet50 CNN + 3DMM coefficient regressor
│   ├── bfm.py                # Basel Face Model differentiable decoder
│   ├── gcn.py                # GCN/GAT mesh refinement network
│   ├── renderer.py           # Differentiable renderer (PyTorch3D / fallback)
│   └── facearcs.py           # Full end-to-end model
├── utils/
│   ├── losses.py             # All self-supervised losses
│   ├── metrics.py            # PSNR, SSIM, NME, mesh regularity
│   └── visualise.py          # Grid plots, OBJ export, turntable
├── checkpoints/              # Saved model checkpoints
├── outputs/                  # Reconstruction visualisations
├── logs/                     # TensorBoard logs
├── train.py                  # Training entry point
├── inference.py              # Inference entry point
├── setup_and_demo.py         # Quick-start demo (no real data needed)
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies

```bash
# Core dependencies
pip install -r requirements.txt

# PyTorch Geometric (recommended — improves GCN)
pip install torch-scatter torch-sparse torch-geometric \
    -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

# PyTorch3D (recommended — improves renderer)
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```

### 2. Verify setup

```bash
python setup_and_demo.py
```

This runs a full pipeline check on synthetic data with no real images needed.

### 3. Prepare your data

Place your unannotated face images (any common format) into `data/images/`:
```
data/images/
    face_001.jpg
    face_002.png
    subdir/
        face_003.jpg
    ...
```
**No annotations, no 3D labels, no special naming required.**

### 4. (Optional but recommended) Download BFM09

The Basel Face Model provides a statistical face shape prior.
Download from: https://faces.dmi.unibas.ch/bfm/
Place `BFM09_model_info.mat` at `data/BFM/BFM09_model_info.mat`

*Without this file, a synthetic BFM is used automatically — training still works but quality will be lower.*

### 5. Train

```bash
python train.py --config configs/config.yaml
```

Monitor training:
```bash
tensorboard --logdir logs/
```

### 6. Inference

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
Input RGB Image (224×224)
        │
        ▼
┌─────────────────┐
│  ResNet50       │  ← pretrained ImageNet backbone
│  CNN Encoder    │
└────────┬────────┘
         │  global feature (2048-d)
         ▼
┌─────────────────┐
│  MLP Regressor  │  → shape (80), exp (64), tex (80),
│  3DMM Coeffs    │     angles (3), translation (3), scale (1)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  BFM Decoder    │  → coarse 3D mesh (V×3), vertex colours (V×3)
│  (3DMM Prior)   │
└────────┬────────┘
         │  coarse mesh
         ▼
┌─────────────────┐
│  Vertex Feature │  ← samples CNN feature map at projected vertex locations
│  Sampler        │
└────────┬────────┘
         │  per-vertex features (V × 67-d)
         ▼
┌─────────────────┐
│  GAT-GCN        │  → vertex displacements (V×3)
│  Refiner        │     4 layers, 256-d hidden, 4 attention heads
└────────┬────────┘
         │  refined mesh
         ▼
┌─────────────────┐
│  Differentiable │  → rendered image (3×224×224)
│  Renderer       │     silhouette mask (1×224×224)
└────────┬────────┘
         │
         ▼
   Self-supervised Losses (no 3D GT needed):
   ✓ Photometric loss    (rendered vs input pixels)
   ✓ Perceptual loss     (VGG feature space)
   ✓ Pseudo-landmark     (MediaPipe 68-pt → NME)
   ✓ Shape regularisation(3DMM prior)
   ✓ Laplacian smooth    (mesh quality)
   ✓ Face symmetry       (horizontal flip)
   ✓ Identity preserve   (ArcFace embeddings)
```

---

## Configuration

Key settings in `configs/config.yaml`:

| Setting | Default | Description |
|---|---|---|
| `dataset.root_dir` | `./data/images` | Your image folder |
| `dataset.image_size` | 224 | Input resolution |
| `training.epochs` | 100 | Training epochs |
| `training.batch_size` | 16 | Batch size (reduce if OOM) |
| `training.learning_rate` | 1e-4 | AdamW LR |
| `model.backbone` | `resnet50` | `resnet34` / `efficientnet_b3` |
| `model.gcn.num_layers` | 4 | GCN depth |
| `model.gcn.hidden_dim` | 256 | GCN width |
| `training.device` | `cuda` | `cuda` / `cpu` |

---

## Hardware Requirements

| | Minimum | Recommended |
|---|---|---|
| GPU | GTX 1050 Ti (4GB) | RTX 3050 Ti (8GB)+ |
| RAM | 8 GB | 16 GB |
| Storage | 256 GB SSD | 512 GB SSD |

---

## Outputs

After inference, you get:
- `{name}_mesh.obj` — 3D mesh with vertex colours (open in Blender/MeshLab)
- `{name}_comparison.png` — Input | Rendered | Depth side-by-side
- `{name}_turntable/` — 24 rotation frames (360° view)

---

## References

- BFM: Basel Face Model — https://faces.dmi.unibas.ch/bfm/
- 3DMM: Blanz & Vetter, SIGGRAPH 1999
- GAT: Graph Attention Networks — Veličković et al., ICLR 2018
- PyTorch3D — Johnson et al., 2020
- MediaPipe FaceMesh — Kartynnik et al., 2019
