import torch
from torch.amp import autocast
from torchvision.models import resnet50

net = resnet50().cuda()
x = torch.randn(6, 3, 224, 224, device='cuda')

with autocast(device_type='cuda'):
    out = net(x)
    print('resnet has nan:', torch.isnan(out).any().item())
