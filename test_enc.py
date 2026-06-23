import torch
from torch.amp import autocast
from models.encoder import FaceEncoder

enc = FaceEncoder().cuda()
x = torch.randn(6, 3, 224, 224, device='cuda')

with autocast(device_type='cuda'):
    c, f = enc(x)
    print('c has nan:', torch.isnan(c['shape']).any().item())
    print('f has nan:', torch.isnan(f).any().item())
