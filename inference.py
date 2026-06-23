"""
inference.py
3DFaceARC Inference — run 3D face reconstruction on a single image
or a folder of images.

Usage:
    python inference.py --input path/to/image.jpg --checkpoint checkpoints/epoch_050_best.pt
    python inference.py --input path/to/folder/  --checkpoint checkpoints/epoch_050_best.pt --output_dir results/
"""

import argparse
import yaml
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import mediapipe as mp

from models.3DFaceARC import 3DFaceARC
from utils.visualise  import export_obj, save_reconstruction_grid, visualise_turntable

# Image preprocessing
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

def preprocess_image(img_path: str, image_size: int = 224, device: str = 'cpu'):
    """
    Load, detect face, crop, and preprocess a single image for inference.
    Returns: normalised tensor (1,3,H,W), raw tensor (1,3,H,W)
    """
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {img_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Face detection + crop
    mp_detect = mp.solutions.face_detection
    with mp_detect.FaceDetection(model_selection=1, min_detection_confidence=0.4) as det:
        results = det.process(img_rgb)
    h, w = img_rgb.shape[:2]
    if results.detections:
        bb = results.detections[0].location_data.relative_bounding_box
        x1 = max(0, int((bb.xmin - 0.1) * w))
        y1 = max(0, int((bb.ymin - 0.1) * h))
        x2 = min(w, int((bb.xmin + bb.width  + 0.1) * w))
        y2 = min(h, int((bb.ymin + bb.height + 0.1) * h))
        face = img_rgb[y1:y2, x1:x2]
        if face.size == 0:
            face = img_rgb
    else:
        print("[Inference] No face detected — using full image.")
        face = img_rgb

    face = cv2.resize(face, (image_size, image_size))
    pil  = Image.fromarray(face)

    norm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])
    raw = transforms.ToTensor()(pil)

    return norm(pil).unsqueeze(0).to(device), raw.unsqueeze(0), face


# Single image inference
@torch.no_grad()
def reconstruct_single(img_path: str, model: torch.nn.Module,
                        cfg: dict, output_dir: str):
    """
    Run 3D reconstruction on a single image and save results.
    """
    device = cfg['training']['device']
    img_size = cfg['dataset']['image_size']

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(img_path).stem

    # Preprocess
    img_tensor, img_raw, face_crop = preprocess_image(img_path, img_size, device)

    # Forward pass
    t0 = time.time()
    out = model(img_tensor)
    elapsed = time.time() - t0
    print(f"[Inference] Reconstruction time: {elapsed*1000:.1f} ms")

    # Extract results
    verts  = out['refined_verts'][0].cpu().numpy()      # (V, 3)
    colors = out['colors'][0].cpu().numpy()             # (V, 3) [0,1]
    faces  = out['faces'].cpu().numpy()                 # (F, 3)
    rendered = out['rendered_img'][0].cpu().permute(1,2,0).numpy().clip(0,1)

    # Save OBJ mesh
    obj_path = out_dir / f'{stem}_mesh.obj'
    export_obj(verts, faces, colors, str(obj_path))

    # Save side-by-side comparison
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(face_crop)
    axes[0].set_title("Input Face")
    axes[0].axis('off')

    axes[1].imshow(rendered)
    axes[1].set_title("Rendered Reconstruction")
    axes[1].axis('off')

    # Depth map
    z = verts[:, 2]
    axes[2].scatter(verts[::10, 0], verts[::10, 1],
                    c=(z[::10]-z.min())/(z.max()-z.min()+1e-8),
                    cmap='plasma', s=1)
    axes[2].set_title("Depth Map")
    axes[2].axis('off')
    axes[2].invert_yaxis()

    plt.tight_layout()
    comp_path = out_dir / f'{stem}_comparison.png'
    plt.savefig(comp_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Inference] Comparison saved → {comp_path}")

    # Turntable
    tt_dir = out_dir / f'{stem}_turntable'
    visualise_turntable(verts, faces, colors, str(tt_dir), n_frames=24)

    # Print coefficients
    print("\n── 3DMM Coefficients (first 5 values each) ──")
    for k, v in out['coeffs'].items():
        vals = v[0].cpu().numpy()
        print(f"  {k:15s}: {vals[:5]}")

    return out

# Batch inference on a folder
@torch.no_grad()
def reconstruct_folder(folder: str, model: torch.nn.Module,
                        cfg: dict, output_dir: str):
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    paths = sorted([p for p in Path(folder).rglob('*') if p.suffix.lower() in exts])
    print(f"[Inference] Found {len(paths)} images in {folder}")

    for i, p in enumerate(paths):
        print(f"\n[{i+1}/{len(paths)}] Processing: {p.name}")
        try:
            reconstruct_single(str(p), model, cfg, output_dir)
        except Exception as e:
            print(f"  ERROR: {e}")


# Load model helper
def load_model(checkpoint_path: str, cfg: dict) -> torch.nn.Module:
    device = cfg['training']['device']
    model = 3DFaceARC(cfg).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f"[Model] Loaded from {checkpoint_path} (epoch {ckpt.get('epoch','?')})")
    return model

# Entry point
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='3DFaceARC Inference')
    parser.add_argument('--input',      type=str, required=True,
                        help='Path to image file or folder')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained checkpoint .pt')
    parser.add_argument('--config',     type=str, default='configs/config.yaml')
    parser.add_argument('--output_dir', type=str, default='outputs/inference')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    # Force eval device
    if not torch.cuda.is_available():
        cfg['training']['device'] = 'cpu'

    model = load_model(args.checkpoint, cfg)

    inp = Path(args.input)
    if inp.is_dir():
        reconstruct_folder(str(inp), model, cfg, args.output_dir)
    else:
        reconstruct_single(str(inp), model, cfg, args.output_dir)
