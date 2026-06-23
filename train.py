"""
train.py
3DFaceARC Training Script — self-supervised, no annotation required.

Usage:
    python train.py --config configs/config.yaml
    python train.py --config configs/config.yaml --resume checkpoints/epoch_10.pt
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
import argparse
import yaml
import time
import logging
from pathlib import Path

import torch
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

from data.dataset    import build_dataloaders
from models.3DFaceARC import 3DFaceARC
from utils.losses    import 3DFaceARCLoss
from utils.metrics   import compute_metrics
from utils.visualise import save_reconstruction_grid

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger('3DFaceARC')

# Scheduler factory
def build_scheduler(optimizer, cfg, n_batches):
    sched = cfg['training']['scheduler']
    epochs = cfg['training']['epochs']
    warmup = cfg['training']['warmup_epochs']

    if sched == 'cosine':
        # Linear warmup then cosine decay
        def lr_lambda(step):
            epoch = step / max(n_batches, 1)
            if epoch < warmup:
                return epoch / warmup
            progress = (epoch - warmup) / max(epochs - warmup, 1)
            return 0.5 * (1.0 + torch.cos(torch.tensor(3.14159 * progress)).item())
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    elif sched == 'step':
        return optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    elif sched == 'plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5,
                                                     factor=0.5, verbose=True)
    return None

# Checkpoint helpers
def save_checkpoint(model, optimizer, scaler, epoch, val_loss, cfg, tag='latest'):
    ckpt_dir = Path(cfg['paths']['checkpoints'])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f'epoch_{epoch:03d}_{tag}.pt'
    torch.save({
        'epoch':     epoch,
        'model':     model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scaler':    scaler.state_dict(),
        'val_loss':  val_loss,
        'cfg':       cfg,
    }, path)
    log.info(f"Checkpoint saved → {path}")
    return path


def load_checkpoint(path, model, optimizer=None, scaler=None):
    ckpt = torch.load(path, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    if optimizer and 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
    if scaler and 'scaler' in ckpt:
        scaler.load_state_dict(ckpt['scaler'])
    log.info(f"Resumed from {path} (epoch {ckpt['epoch']})")
    return ckpt['epoch']

# Train one epoch
def train_epoch(model, loader, criterion, optimizer, scaler,
                scheduler, cfg, writer, global_step, epoch):
    model.train()
    device    = cfg['training']['device']
    grad_clip = cfg['training']['grad_clip']
    accumulate_grad_batches = cfg['training'].get('accumulate_grad_batches', 1)
    log_every = max(1, len(loader) // 10)

    epoch_losses  = {}
    skipped_steps = 0        # counts batches where GradScaler skipped optimizer
    t0 = time.time()

    optimizer.zero_grad(set_to_none=True)

    for i, batch in enumerate(loader):
        try:
            if i == 0: log.info(f"  [DEBUG] Batch 0 starting...")
            images = batch['image'].to(device, non_blocking=True)
            batch['image_raw'] = batch['image_raw'].to(device, non_blocking=True)
            batch['landmarks'] = batch['landmarks'].to(device, non_blocking=True)

            with autocast(device_type='cuda', enabled=False):
                if i == 0: log.info(f"  [DEBUG] Running model forward...")
                out    = model(images)
                if i == 0: log.info(f"  [DEBUG] Computing criterion...")
                losses = criterion(out, batch)
                loss   = losses['total'] / accumulate_grad_batches

            if i == 0: log.info(f"  [DEBUG] Running backward...")
            scaler.scale(loss).backward()
            if i == 0: log.info(f"  [DEBUG] Backward complete.")

            if (i + 1) % accumulate_grad_batches == 0 or (i + 1) == len(loader):
                scale_before = scaler.get_scale()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                # Detect when GradScaler silently skips optimizer.step
                # (happens when gradients contain NaN/Inf)
                if scaler.get_scale() < scale_before:
                    skipped_steps += 1
                    log.warning(
                        f"  E{epoch:03d} [{i+1}/{len(loader)}] "
                        f"GradScaler skipped optimizer step (NaN/Inf grads). "
                        f"Scale {scale_before:.0f} → {scaler.get_scale():.0f}. "
                        f"Skipped so far this epoch: {skipped_steps}"
                    )

                if scheduler and not isinstance(
                        scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step()

            # Accumulate
            for k, v in losses.items():
                epoch_losses[k] = epoch_losses.get(k, 0.0) + v.item()

            if (i + 1) % log_every == 0:
                lr      = optimizer.param_groups[0]['lr']
                elapsed = time.time() - t0
                total   = losses['total'].item()
                # Per-component breakdown
                comp = "  ".join(
                    f"{k}={v.item():.3f}"
                    for k, v in losses.items()
                    if k != 'total'
                )
                log.info(
                    f"  E{epoch:03d} [{i+1}/{len(loader)}] "
                    f"loss={total:.4f}  lr={lr:.2e}  t={elapsed:.1f}s  "
                    f"scale={scaler.get_scale():.0f}\n"
                    f"    [{comp}]"
                )

            # TensorBoard
            writer.add_scalar('train/loss_total', losses['total'].item(), global_step)
            for k, v in losses.items():
                if k != 'total':
                    writer.add_scalar(f'train/{k}', v.item(), global_step)
            writer.add_scalar('train/scaler_scale', scaler.get_scale(), global_step)
            global_step += 1

        except Exception as e:
            log.error(f"  E{epoch:03d} [{i+1}/{len(loader)}] Batch error (skipping): {e}")
            optimizer.zero_grad(set_to_none=True)  # clear any partial grads
            torch.cuda.empty_cache()
            continue

    if skipped_steps > 0:
        log.warning(
            f"  Epoch {epoch}: GradScaler skipped {skipped_steps}/{len(loader)} "
            f"optimizer steps due to NaN/Inf gradients."
        )

    n = max(len(epoch_losses.get('total', [0])), len(loader), 1) if not epoch_losses else len(loader)
    n = max(n, 1)
    return {k: v / n for k, v in epoch_losses.items()}, global_step

# Validation epoch
@torch.no_grad()
def val_epoch(model, loader, criterion, cfg, writer, epoch):
    model.eval()
    device = cfg['training']['device']
    epoch_losses = {}
    all_metrics  = []
    vis_done     = False

    for i, batch in enumerate(loader):
        images = batch['image'].to(device, non_blocking=True)
        batch['image_raw'] = batch['image_raw'].to(device, non_blocking=True)
        batch['landmarks'] = batch['landmarks'].to(device, non_blocking=True)
        
        out    = model(images)
        losses = criterion(out, batch)

        for k, v in losses.items():
            epoch_losses[k] = epoch_losses.get(k, 0.0) + v.item()

        # Compute metrics
        metrics = compute_metrics(out, batch)
        all_metrics.append(metrics)

        # Save a visualisation grid once per epoch
        if not vis_done:
            save_reconstruction_grid(
                batch['image_raw'], out['rendered_img'],
                out['refined_verts'], out['faces'],
                save_dir=cfg['paths']['outputs'],
                epoch=epoch
            )
            vis_done = True

    n = len(loader)
    avg_losses  = {k: v / n for k, v in epoch_losses.items()}
    avg_metrics = {k: sum(m[k] for m in all_metrics) / len(all_metrics)
                   for k in all_metrics[0]}

    # TensorBoard
    for k, v in avg_losses.items():
        writer.add_scalar(f'val/{k}', v, epoch)
    for k, v in avg_metrics.items():
        writer.add_scalar(f'val/{k}', v, epoch)

    return avg_losses, avg_metrics

# Main training loop
def train(cfg: dict, resume: str = None):
    device = cfg['training']['device']
    if device == 'cuda' and not torch.cuda.is_available():
        log.warning("CUDA not available — falling back to CPU.")
        device = cfg['training']['device'] = 'cpu'
    log.info(f"Device: {device}")

    # Data
    train_loader, val_loader, _ = build_dataloaders(cfg)
    log.info(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # Model
    model = 3DFaceARC(cfg).to(device)
    if hasattr(torch, 'compile'):
        if sys.platform != 'win32':
            log.info("Applying torch.compile() for JIT speedup...")
            model = torch.compile(model)
        else:
            log.info("Skipping torch.compile() on Windows due to lack of Triton support.")
        
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Trainable parameters: {n_params:,}")

    # Loss
    criterion = 3DFaceARCLoss(cfg).to(device)

    # Optimiser
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg['training']['learning_rate'],
        weight_decay=cfg['training']['weight_decay']
    )

    # Scaler (AMP)
    scaler = GradScaler('cuda', enabled=False)

    # Scheduler
    accumulate_grad_batches = cfg['training'].get('accumulate_grad_batches', 1)
    n_optimizer_steps = max(1, len(train_loader) // accumulate_grad_batches)
    scheduler = build_scheduler(optimizer, cfg, n_optimizer_steps)

    # TensorBoard
    writer = SummaryWriter(log_dir=cfg['paths']['logs'])

    start_epoch = 1
    if resume:
        start_epoch = load_checkpoint(resume, model, optimizer, scaler) + 1

    best_val_loss = float('inf')
    global_step   = (start_epoch - 1) * len(train_loader)

    log.info("Starting 3DFaceARC self-supervised training")

    for epoch in range(start_epoch, cfg['training']['epochs'] + 1):
        log.info(f"\nEpoch {epoch}/{cfg['training']['epochs']}")

        # Train
        train_losses, global_step = train_epoch(
            model, train_loader, criterion, optimizer, scaler,
            scheduler, cfg, writer, global_step, epoch
        )
        log.info(f"Train loss: {train_losses['total']:.4f}")

        # Validate
        val_losses, val_metrics = val_epoch(
            model, val_loader, criterion, cfg, writer, epoch
        )
        log.info(f"Val   loss: {val_losses['total']:.4f}")
        log.info(f"Val metrics: " +" | ".join(f"{k}={v:.4f}" for k, v in val_metrics.items()))

        # Plateau scheduler step
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_losses['total'])

        # Save checkpoint
        if epoch % cfg['training']['save_every'] == 0:
            save_checkpoint(model, optimizer, scaler, epoch,
                            val_losses['total'], cfg)

        # Save best
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            save_checkpoint(model, optimizer, scaler, epoch,
                            val_losses['total'], cfg, tag='best')
            log.info(f"  ★ New best val loss: {best_val_loss:.4f}")

    writer.close()
    log.info("\nTraining complete!")

# Entry point
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='3DFaceARC Training')
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--resume', type=str, default=None,help='Path to checkpoint to resume from')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    train(cfg, resume=args.resume)
