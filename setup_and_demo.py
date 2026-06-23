"""
setup_and_demo.py
Quick-start script for 3DFaceARC.
Run this first to:
  1. Verify all imports
  2. Test the pipeline end-to-end on a synthetic image (no real data needed)
  3. Print a system summary

Usage:
    python setup_and_demo.py
"""

import sys
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import platform
import numpy as np

# Dependency check
def check_deps():
    deps = {
        'torch':          'torch',
        'torchvision':    'torchvision',
        'numpy':          'numpy',
        'cv2':            'opencv-python',
        'PIL':            'Pillow',
        'mediapipe':      'mediapipe',
        'scipy':          'scipy',
        'sklearn':        'scikit-learn',
        'trimesh':        'trimesh',
        'open3d':         'open3d',
        'matplotlib':     'matplotlib',
        'tqdm':           'tqdm',
        'yaml':           'PyYAML',
        'timm':           'timm',
        'einops':         'einops',
        'tensorboard':    'tensorboard',
    }
    optional = {
        'torch_geometric': 'torch-geometric (optional, improves GCN)',
        'pytorch3d':       'pytorch3d    (optional, improves renderer)',
    }

    print("3DFaceARC - Dependency Check")
    all_ok = True
    for mod, pkg in deps.items():
        try:
            __import__(mod)
            print(f"  [OK] {pkg}")
        except ImportError:
            print(f"  [FAIL] {pkg}  <- MISSING (pip install {pkg})")
            all_ok = False

    print("\nOptional (recommended):")
    for mod, pkg in optional.items():
        try:
            __import__(mod)
            print(f"  [OK] {pkg}")
        except ImportError:
            print(f"  [OPT] {pkg}  <- not installed")

    print()
    return all_ok


# System info
def print_system_info():
    import torch
    print("System Information")
    print(f"Python   : {sys.version.split()[0]}")
    print(f"Platform : {platform.system()} {platform.machine()}")
    print(f"PyTorch  : {torch.__version__}")
    print(f"CUDA     : {'Available (' + torch.version.cuda + ')' if torch.cuda.is_available() else 'Not available'}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}    : {props.name} ({props.total_memory//1024**2} MB)")
    print()


# End-to-end demo on synthetic data
def run_demo():
    import torch
    import yaml
    print("End-to-End Pipeline Demo (synthetic data)")

    # Load config
    with open('configs/config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg['training']['device'] = device
    print(f"  Running on: {device.upper()}")

    # Override BFM path for demo (will use synthetic BFM)
    cfg['bfm']['model_path'] = 'data/BFM/NONEXISTENT.mat'

    # Build model
    print("\n[1/4] Building 3DFaceARC model...")
    from models.3DFaceARC import 3DFaceARC
    model = 3DFaceARC(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"        Trainable parameters: {n_params:,}")

    # Build loss
    print("[2/4] Building loss function...")
    from utils.losses import 3DFaceARCLoss
    criterion = 3DFaceARCLoss(cfg).to(device)

    # Synthetic batch
    print("[3/4] Creating synthetic batch (B=2, 3×224×224)...")
    B = 2
    synthetic_images = torch.rand(B, 3, 224, 224).to(device)
    batch = {
        'image':     synthetic_images,
        'image_raw': torch.rand(B, 3, 224, 224),
        'landmarks': torch.zeros(B, 68, 2),
        'path':      ['synthetic_0', 'synthetic_1'],
    }

    # Forward pass
    print("[4/4] Running forward pass...")
    model.train()
    with torch.no_grad():
        out = model(synthetic_images)

    print("\nOutput shapes")
    print(f"coarse_verts  : {out['coarse_verts'].shape}")
    print(f"refined_verts : {out['refined_verts'].shape}")
    print(f"colors        : {out['colors'].shape}")
    print(f"faces         : {out['faces'].shape}")
    print(f"rendered_img  : {out['rendered_img'].shape}")
    print(f"id_embed      : {out['id_embed_input'].shape}")

    print("\nCoefficient shapes")
    for k, v in out['coeffs'].items():
        print(f"{k:15s}: {tuple(v.shape)}")

    # Loss forward (skip grad for demo)
    with torch.no_grad():
        losses = criterion(out, batch)
    print("\nLosses")
    for k, v in losses.items():
        print(f"    {k:20s}: {v.item():.4f}")

    print("Demo completed successfully!")
    print()
    print("Next steps:")
    print("1. Place your face images in:  data/images/")
    print("2. (Optional) Download BFM09:  https://faces.dmi.unibas.ch/bfm/")
    print("and place at: data/BFM/BFM09_model_info.mat")
    print("3. Start training:")
    print("python train.py --config configs/config.yaml")
    print("4. Run inference:")
    print("python inference.py --input path/to/face.jpg \\")
    print("--checkpoint checkpoints/epoch_050_best.pt")
    print()

# Entry
if __name__ == '__main__':
    ok = check_deps()
    print_system_info()
    if ok:
        run_demo()
    else:
        print("Please install missing dependencies first:")
        print("  pip install -r requirements.txt")
        sys.exit(1)
