"""
data/dataset.py
FaceARCs Dataset — handles unannotated single-view face images.
Pseudo-labels (landmarks) are generated on-the-fly via MediaPipe.
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import mediapipe as mp
from pathlib import Path

# Landmark Extractor  (pseudo-annotation, no manual labels needed)
class MediaPipeLandmarkExtractor:
    """
    Extracts 68 facial landmarks from a face image using MediaPipe FaceMesh.
    These serve as weak/pseudo supervision in place of manual annotations.
    """
    # MediaPipe FaceMesh indices that roughly correspond to the 68-point scheme
    MEDIAPIPE_68_MAPPING = [
        162, 234, 93, 58, 172, 136, 149, 148, 152, 377,
        378, 365, 397, 288, 323, 454, 389, 71, 63, 105,
        66, 107, 336, 296, 334, 293, 301, 168, 197, 5,
        4, 75, 97, 2, 326, 305, 33, 160, 158, 133,
        153, 144, 362, 385, 387, 263, 373, 380, 61, 39,
        37, 0, 267, 269, 291, 405, 314, 17, 84, 181,
        78, 82, 13, 312, 308, 317, 14, 87
    ]

    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )

    def extract(self, image_rgb: np.ndarray):
        """
        Args:
            image_rgb: HxWx3 uint8 RGB image
        Returns:
            landmarks: (68, 2) float32 array in pixel coords, or None if not detected
        """
        h, w = image_rgb.shape[:2]
        results = self.face_mesh.process(image_rgb)
        if not results.multi_face_landmarks:
            return None

        face_lmks = results.multi_face_landmarks[0].landmark
        pts = np.array([[lmk.x * w, lmk.y * h] for lmk in face_lmks], dtype=np.float32)
        # Select the 68-point subset
        lmks_68 = pts[self.MEDIAPIPE_68_MAPPING]
        return lmks_68

    def __del__(self):
        self.face_mesh.close()

# Augmentation transforms
def get_transforms(image_size: int, augment: bool = True):
    if augment:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])


# Main Dataset
SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


class FaceDataset(Dataset):
    """
    Unannotated single-view face image dataset.
    Pseudo-landmarks are extracted at load time via MediaPipe.

    Directory structure expected:
        root_dir/
            img_001.jpg
            img_002.png
            ...   (flat folder OR nested subfolders, both supported)
    """

    def __init__(self, root_dir: str, image_size: int = 224,
                 augment: bool = False, cache_landmarks: bool = True):
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.augment = augment
        self.transform = get_transforms(image_size, augment)

        # Collect all image paths (recursive)
        self.image_paths = sorted([
            p for p in self.root_dir.rglob('*')
            if p.suffix.lower() in SUPPORTED_EXTENSIONS
        ])
        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"No images found in {root_dir}")
        print(f"[Dataset] Found {len(self.image_paths)} images in {root_dir}")

        # Landmark extractor (pseudo-annotations)
        self.landmark_extractor = MediaPipeLandmarkExtractor()

        # Optional: cache landmarks to avoid re-computing each epoch
        self.cache_landmarks = cache_landmarks
        self._landmark_cache = {}

    def __len__(self):
        return len(self.image_paths)

    def _load_image(self, path: Path):
        img = cv2.imread(str(path))
        if img is None:
            raise IOError(f"Cannot load image: {path}")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img_rgb

    def _detect_and_crop_face(self, img_rgb: np.ndarray):
        """
        Use MediaPipe Face Detection to crop the largest face region.
        Falls back to centre crop if no face is detected.
        """
        mp_detect = mp.solutions.face_detection
        with mp_detect.FaceDetection(model_selection=1, min_detection_confidence=0.4) as det:
            results = det.process(img_rgb)
        h, w = img_rgb.shape[:2]
        if results.detections:
            det0 = results.detections[0].location_data.relative_bounding_box
            x1 = max(0, int((det0.xmin - 0.1) * w))
            y1 = max(0, int((det0.ymin - 0.1) * h))
            x2 = min(w, int((det0.xmin + det0.width + 0.1) * w))
            y2 = min(h, int((det0.ymin + det0.height + 0.1) * h))
            crop = img_rgb[y1:y2, x1:x2]
            if crop.size == 0:
                crop = img_rgb
        else:
            # Centre crop (80% of image)
            m = 0.10
            crop = img_rgb[int(h*m):int(h*(1-m)), int(w*m):int(w*(1-m))]
        return cv2.resize(crop, (self.image_size, self.image_size))

    def __getitem__(self, idx: int):
        path = self.image_paths[idx]
        img_rgb = self._load_image(path)

        # Face crop
        face_crop = self._detect_and_crop_face(img_rgb)

        # Pseudo-landmarks (cached)
        key = str(path)
        if self.cache_landmarks and key in self._landmark_cache:
            landmarks = self._landmark_cache[key]
        else:
            landmarks = self.landmark_extractor.extract(face_crop)
            if landmarks is None:
                # Use a uniform grid as fallback (no face detected)
                landmarks = np.zeros((68, 2), dtype=np.float32)
            else:
                # Normalise to [-1, 1]
                landmarks = landmarks / (self.image_size / 2.0) - 1.0
            if self.cache_landmarks:
                self._landmark_cache[key] = landmarks

        # PIL image → tensor
        pil_img = Image.fromarray(face_crop)
        img_tensor = self.transform(pil_img)

        # Raw image tensor (for photometric loss, no normalisation)
        raw_transform = transforms.Compose([
            transforms.ToTensor()  # [0,1]
        ])
        img_raw = raw_transform(pil_img)

        return {
            'image': img_tensor,           # normalised, augmented
            'image_raw': img_raw,          # [0,1], no augmentation
            'landmarks': torch.from_numpy(landmarks).float(),  # (68,2)
            'path': str(path)
        }

# DataLoader factory
def build_dataloaders(cfg: dict):
    """
    Build train / val / test DataLoaders from config dict.
    No annotation files required.
    """
    image_size = cfg['dataset']['image_size']
    root_dir   = cfg['dataset']['root_dir']
    n_workers  = cfg['dataset']['num_workers']
    batch_size = cfg['training']['batch_size']

    full_dataset = FaceDataset(root_dir, image_size=image_size,
                               augment=False, cache_landmarks=True)
    n = len(full_dataset)

    # Split sizes
    n_train = int(n * cfg['dataset']['train_split'])
    n_val   = int(n * cfg['dataset']['val_split'])
    n_test  = n - n_train - n_val

    train_set, val_set, test_set = random_split(
        full_dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )

    # Enable augmentation only on train
    train_set.dataset.augment = cfg['dataset']['augmentation']

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True,  num_workers=n_workers, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size,
                              shuffle=False, num_workers=n_workers, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=1,
                              shuffle=False, num_workers=0,         pin_memory=False)

    print(f"[DataLoader] Train={n_train} | Val={n_val} | Test={n_test}")
    return train_loader, val_loader, test_loader
