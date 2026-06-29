"""
train.py
FaceARCs Training Script — self-supervised, no annotation required.

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
import torch.nn.functional as F          # ← added: needed for F.mse_loss in debug block
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from data.dataset       import build_dataloaders
from models.facearcs    import FaceARCs
from utils.losses       import FaceARCsLoss
from utils.metrics      import compute_metrics
from utils.visualise    import save_reconstruction_grid, export_obj, visualise_turntable
from utils.plot_metrics import MetricsTracker


# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger('FaceARCs')


# Scheduler factory
def build_scheduler(optimizer, cfg, n_batches):
    sched  = cfg['training']['scheduler']
    epochs = cfg['training']['epochs']
    warmup = cfg['training']['warmup_epochs']

    if sched == 'cosine':
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
def save_checkpoint(model, optimizer, epoch, val_loss, cfg, tag='latest'):
    ckpt_dir = Path(cfg['paths']['checkpoints'])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f'epoch_{epoch:03d}_{tag}.pt'
    torch.save({
        'epoch':     epoch,
        'model':     model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'val_loss':  val_loss,
        'cfg':       cfg,
    }, path)
    log.info(f"Checkpoint saved → {path}")
    return path


def load_checkpoint(path, model, optimizer=None):
    ckpt = torch.load(path, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    if optimizer and 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
    log.info(f"Resumed from {path} (epoch {ckpt['epoch']})")
    return ckpt['epoch']


# Train one epoch
def train_epoch(model, loader, criterion, optimizer,
                scheduler, cfg, writer, global_step, epoch):
    model.train()
    device    = cfg['training']['device']
    grad_clip = cfg['training']['grad_clip']
    accumulate_grad_batches = cfg['training'].get('accumulate_grad_batches', 1)
    log_every = max(1, len(loader) // 10)

    epoch_losses  = {}
    t0 = time.time()

    optimizer.zero_grad(set_to_none=True)

    for i, batch in enumerate(loader):
        try:
            if i == 0: log.info(f"  [DEBUG] Batch 0 starting...")
            images = batch['image'].to(device, non_blocking=True)
            batch['image_raw'] = batch['image_raw'].to(device, non_blocking=True)
            batch['landmarks'] = batch['landmarks'].to(device, non_blocking=True)

            if i == 0: log.info(f"  [DEBUG] Running model forward...")
            out = model(images)

            if i == 0: log.info(f"  [DEBUG] Computing criterion...")
            losses = criterion(out, batch)
            loss   = losses['total'] / accumulate_grad_batches

            if i == 0: log.info(f"  [DEBUG] Running backward...")
            loss.backward()
            if i == 0: log.info(f"  [DEBUG] Backward complete.")

            if (i + 1) % accumulate_grad_batches == 0 or (i + 1) == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                if scheduler and not isinstance(
                        scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step()

            for k, v in losses.items():
                epoch_losses[k] = epoch_losses.get(k, 0.0) + v.item()

            if (i + 1) % log_every == 0:
                lr      = optimizer.param_groups[0]['lr']
                elapsed = time.time() - t0
                total   = losses['total'].item()
                comp = "  ".join(
                    f"{k}={v.item():.3f}"
                    for k, v in losses.items()
                    if k != 'total'
                )
                log.info(
                    f"  E{epoch:03d} [{i+1}/{len(loader)}] "
                    f"loss={total:.4f}  lr={lr:.2e}  t={elapsed:.1f}s\n"
                    f"    [{comp}]"
                )

            writer.add_scalar('train/loss_total', losses['total'].item(), global_step)
            for k, v in losses.items():
                if k != 'total':
                    writer.add_scalar(f'train/{k}', v.item(), global_step)
            global_step += 1

        except Exception as e:
            log.error(f"  E{epoch:03d} [{i+1}/{len(loader)}] Batch error (skipping): {e}")
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            continue

    n = max(len(loader), 1)
    return {k: v / n for k, v in epoch_losses.items()}, global_step


# Validation epoch
@torch.no_grad()
def val_epoch(model, loader, criterion, cfg, writer, epoch):
    model.eval()
    device = cfg['training']['device']
    epoch_losses = {}
    all_metrics  = []
    vis_done     = False
    sample_mesh  = None

    for i, batch in enumerate(loader):
        images = batch['image'].to(device, non_blocking=True)
        batch['image_raw'] = batch['image_raw'].to(device, non_blocking=True)
        batch['landmarks'] = batch['landmarks'].to(device, non_blocking=True)

        out    = model(images)

        rendered = out['rendered_img']
        log.debug(f"  Rendered stats — min:{rendered.min():.3f} "
                  f"max:{rendered.max():.3f} mean:{rendered.mean():.3f}")

        losses = criterion(out, batch)

        for k, v in losses.items():
            val = v.item()
            if torch.isfinite(torch.tensor(val)):
                epoch_losses[k] = epoch_losses.get(k, 0.0) + val
            else:
                log.warning(f"  Val batch {i}: NaN/Inf in loss '{k}', skipping")

        metrics = compute_metrics(out, batch)
        all_metrics.append(metrics)

        if not vis_done:
            save_reconstruction_grid(
                batch['image_raw'], out['rendered_img'],
                out['refined_verts'], out['faces'],
                save_dir=cfg['paths']['outputs'],
                epoch=epoch
            )

            import numpy as np
            verts  = out['refined_verts'][0].cpu().numpy()             # (V, 3)
            faces  = out['faces'].cpu().numpy()                         # (F, 3)
            rendered_np = (
                out['rendered_img'][0].detach().cpu()
                .permute(1, 2, 0).numpy().clip(0, 1)
                .reshape(-1, 3)
            )
            n_verts = verts.shape[0]
            if rendered_np.shape[0] >= n_verts:
                colors = rendered_np[:n_verts]
            else:
                repeats = (n_verts // rendered_np.shape[0]) + 1
                colors  = np.array((rendered_np.tolist() * repeats)[:n_verts])

            sample_mesh = {'verts': verts, 'faces': faces, 'colors': colors}
            vis_done = True

    n = len(loader)
    avg_losses  = {k: v / n for k, v in epoch_losses.items()}
    avg_metrics = {k: sum(m[k] for m in all_metrics) / len(all_metrics)
                   for k in all_metrics[0]}

    for k, v in avg_losses.items():
        writer.add_scalar(f'val/{k}', v, epoch)
    for k, v in avg_metrics.items():
        writer.add_scalar(f'val/{k}', v, epoch)

    return avg_losses, avg_metrics, sample_mesh


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
    model = FaceARCs(cfg).to(device)
    if hasattr(torch, 'compile'):
        if sys.platform != 'win32':
            log.info("Applying torch.compile() for JIT speedup...")
            model = torch.compile(model)
        else:
            log.info("Skipping torch.compile() on Windows due to lack of Triton support.")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Trainable parameters: {n_params:,}")

    # Loss
    criterion = FaceARCsLoss(cfg).to(device)

    # Optimiser
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg['training']['learning_rate'],
        weight_decay=cfg['training']['weight_decay']
    )

    # Scheduler
    accumulate_grad_batches = cfg['training'].get('accumulate_grad_batches', 1)
    n_optimizer_steps = max(1, len(train_loader) // accumulate_grad_batches)
    scheduler = build_scheduler(optimizer, cfg, n_optimizer_steps)

    # TensorBoard
    writer = SummaryWriter(log_dir=cfg['paths']['logs'])

    # Metrics tracker
    tracker = MetricsTracker(
        save_dir=cfg['paths']['outputs'],
        experiment_name='facearcs'
    )

    start_epoch = 1
    if resume:
        start_epoch = load_checkpoint(resume, model, optimizer) + 1

    best_val_loss = float('inf')
    global_step   = (start_epoch - 1) * len(train_loader)

    log.info("=" * 60)
    log.info("Starting FaceARCs self-supervised training")
    log.info("=" * 60)

    for epoch in range(start_epoch, cfg['training']['epochs'] + 1):
        log.info(f"\n── Epoch {epoch}/{cfg['training']['epochs']} ──")

        # Train
        train_losses, global_step = train_epoch(
            model, train_loader, criterion, optimizer,
            scheduler, cfg, writer, global_step, epoch
        )
        log.info(f"  Train loss: {train_losses['total']:.4f}")

        # Validate
        val_losses, val_metrics, sample_mesh = val_epoch(
            model, val_loader, criterion, cfg, writer, epoch
        )
        log.info(f"  Val   loss: {val_losses['total']:.4f}")
        log.info(f"  Val metrics: " +
                 " | ".join(f"{k}={v:.4f}" for k, v in val_metrics.items()))

        # Plateau scheduler step
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_losses['total'])

        # MetricsTracker update & plot save
        tracker.update_train(
            epoch=epoch,
            losses=train_losses,
            lr=optimizer.param_groups[0]['lr']
        )
        tracker.update_val(epoch=epoch, losses=val_losses, metrics=val_metrics)
        tracker.save_all_plots()
        tracker.print_summary()

        # Save checkpoint
        if epoch % cfg['training']['save_every'] == 0:
            save_checkpoint(model, optimizer, epoch,
                            val_losses['total'], cfg)

        # Save best
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            save_checkpoint(model, optimizer, epoch,
                            val_losses['total'], cfg, tag='best')
            log.info(f"  ★ New best val loss: {best_val_loss:.4f}")

            # Export mesh and turntable frames for the new best epoch
            if sample_mesh is not None:
                out_dir = Path(cfg['paths']['outputs'])

                # OBJ export
                obj_path = out_dir / f'mesh_best_epoch_{epoch:03d}.obj'
                try:
                    export_obj(
                        sample_mesh['verts'],
                        sample_mesh['faces'],
                        sample_mesh['colors'],
                        path=str(obj_path)
                    )
                    log.info(f"  Mesh exported → {obj_path}")
                except Exception as e:
                    log.warning(f"  OBJ export failed: {e}")

                # Turntable frames export
                turntable_dir = out_dir / f'turntable_best_epoch_{epoch:03d}'
                try:
                    visualise_turntable(
                        sample_mesh['verts'],
                        sample_mesh['faces'],
                        sample_mesh['colors'],
                        save_path=str(turntable_dir),
                        n_frames=36
                    )
                    log.info(f"  Turntable frames → {turntable_dir}")
                except Exception as e:
                    log.warning(f"  Turntable export failed: {e}")

    writer.close()
    log.info("\nTraining complete!")


# Entry point
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FaceARCs Training')
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    train(cfg, resume=args.resume)