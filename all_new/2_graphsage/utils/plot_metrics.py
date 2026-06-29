"""
utils/plot_metrics.py
Saves the following plots automatically during and after training:
    1.  Train vs Validation Loss (total)          — per epoch
    2.  Individual Loss Terms (7 sub-losses)      — per epoch
    3.  PSNR curve                                — per epoch
    4.  SSIM curve                                — per epoch
    5.  Mesh Regularity curve                     — per epoch
    6.  Learning Rate schedule                    — per epoch
    7.  GradScaler scale factor                   — per epoch
    8.  Loss heatmap (all 7 terms across epochs)  — end of training
    9.  Reconstruction comparison grid            — per epoch (images)
    10. Per-loss contribution pie chart           — end of training
    11. Train vs Val gap (overfitting monitor)    — per epoch
    12. All metrics combined dashboard            — end of training
    13. Cosine Identity + NME 2D curve            — per epoch
    14. MAE / Normal Consistency / Edge Length    — per epoch
"""

import os
import json
import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')   # non-interactive backend for server/training
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from datetime import datetime
import torch

log = logging.getLogger('FaceARCs')

# Force white background globally so no environment/style-sheet can override it
matplotlib.rcParams['figure.facecolor']  = 'white'
matplotlib.rcParams['axes.facecolor']    = 'white'
matplotlib.rcParams['savefig.facecolor'] = 'white'

# Colour palette (consistent across all plots)
COLOURS = {
    'train':             '#2196F3',   # blue
    'val':               '#F44336',   # red
    'photometric':       '#FF9800',   # orange
    'perceptual':        '#9C27B0',   # purple
    'landmark':          '#4CAF50',   # green
    'shape_reg':         '#00BCD4',   # cyan
    'smooth':            '#795548',   # brown
    'symmetry':          '#FF5722',   # deep orange
    'id_preserve':       '#E91E63',   # pink
    'psnr':              '#3F51B5',   # indigo
    'ssim':              '#009688',   # teal
    'mesh_reg':          '#8BC34A',   # light green
    'lr':                '#607D8B',   # blue grey
    'scale':             '#FFC107',   # amber
    'gap':               '#FF5722',   # deep orange
    'cosine_identity':   '#673AB7',   # deep purple
    'nme_2d':            '#F06292',   # pink 300
    'mae':               '#26C6DA',   # cyan 400
    'normal_consistency':'#66BB6A',   # green 400
    'mean_edge_length':  '#FFA726',   # orange 400
}

LOSS_KEYS = [
    'photometric', 'perceptual', 'landmark',
    'shape_reg', 'smooth', 'symmetry', 'id_preserve'
]

LOSS_LABELS = {
    'photometric': 'Photometric (×1.0)',
    'perceptual':  'Perceptual (×0.3)',
    'landmark':    'Landmark (×0.5)',
    'shape_reg':   'Shape Reg. (×0.1)',
    'smooth':      'Smoothness (×0.05)',
    'symmetry':    'Symmetry (×0.2)',
    'id_preserve': 'Identity (×0.5)',
}

# Shared savefig kwargs — guarantees white background on every plot
_SAVE_KW = dict(dpi=150, bbox_inches='tight', facecolor='white')

# MetricsTracker
class MetricsTracker:
    """
    Tracks and saves all training metrics epoch by epoch.

    Call update_train() after each training epoch.
    Call update_val()   after each validation epoch.
    Call save_all_plots() to write all plots to disk.
    """

    def __init__(self, save_dir: str, experiment_name: str = 'facearcs'):
        self.save_dir  = Path(save_dir)
        self.exp_name  = experiment_name
        self.plots_dir = self.save_dir / 'plots'
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        self.json_path = self.save_dir / f'{experiment_name}_metrics.json'

        self.history = {
            # bookkeeping
            'epochs':             [],
            # total losses
            'train_total':        [],
            'val_total':          [],
            # per-term losses
            'train_losses':       {k: [] for k in LOSS_KEYS},
            'val_losses':         {k: [] for k in LOSS_KEYS},
            # image-quality metrics
            'psnr':               [],
            'ssim':               [],
            'mae':                [],
            # geometry metrics
            'mesh_reg':           [],
            'normal_consistency': [],
            'mean_edge_length':   [],
            # identity / landmark metrics
            'cosine_identity':    [],
            'nme_2d':             [],
            # training internals
            'lr':                 [],
            'scaler_scale':       [],
        }

        if self.json_path.exists():
            self._load_json()
            print(f"[Metrics] Resumed history from {self.json_path} "
                  f"({len(self.history['epochs'])} epochs)")

    # update methods

    def update_train(self, epoch: int, losses: dict,
                     lr: float = 0.0, scaler_scale: float = 1.0):
        """
        Call after each training epoch.
        losses: dict with keys 'total', 'photometric', 'perceptual', etc.
        """
        if epoch not in self.history['epochs']:
            self.history['epochs'].append(epoch)
            self.history['train_total'].append(losses.get('total', 0.0))
            self.history['lr'].append(lr)
            self.history['scaler_scale'].append(scaler_scale)
            for k in LOSS_KEYS:
                self.history['train_losses'][k].append(losses.get(k, 0.0))

    def update_val(self, epoch: int, losses: dict, metrics: dict):
        """
        Call after each validation epoch.
        losses  : dict with 'total' + individual loss keys
        metrics : dict from compute_metrics() — all keys stored
        """
        if epoch not in self.history['epochs']:
            return   # train update hasn't happened yet — skip

        # Guard against duplicate calls for the same epoch
        n_val   = len(self.history['val_total'])
        n_train = self.history['epochs'].index(epoch) + 1
        if n_val >= n_train:
            log.warning(
                f"[Metrics] update_val called twice for epoch {epoch} — ignoring duplicate.")
            return

        # Total + per-term losses
        self.history['val_total'].append(losses.get('total', 0.0))
        for k in LOSS_KEYS:
            self.history['val_losses'][k].append(losses.get(k, 0.0))

        # Image-quality metrics
        self.history['psnr'].append(metrics.get('psnr', 0.0))
        self.history['ssim'].append(metrics.get('ssim', 0.0))
        self.history['mae'].append(metrics.get('mae', 0.0))

        # Geometry metrics
        self.history['mesh_reg'].append(metrics.get('mesh_regularity', 0.0))
        self.history['normal_consistency'].append(
            metrics.get('normal_consistency', 0.0))
        self.history['mean_edge_length'].append(
            metrics.get('mean_edge_length', 0.0))

        # Identity / landmark
        self.history['cosine_identity'].append(
            metrics.get('cosine_identity', 0.0))
        self.history['nme_2d'].append(metrics.get('nme_2d', 0.0))

        self._save_json()

    # JSON persistence

    def _save_json(self):
        with open(self.json_path, 'w') as f:
            json.dump(self.history, f, indent=2)

    def _load_json(self):
        with open(self.json_path, 'r') as f:
            saved = json.load(f)
        # Merge loaded keys so new keys added to history aren't lost
        for k, v in saved.items():
            self.history[k] = v

    # helpers 

    def _epochs(self):
        return self.history['epochs']

    def _val_epochs(self):
        """Returns the subset of train epochs for which val data exists."""
        n = len(self.history['val_total'])
        return self.history['epochs'][:n]

    def _fig_path(self, name: str) -> Path:
        return self.plots_dir / f'{self.exp_name}_{name}.png'

    def _safe_loss_row(self, k: str, epochs: list) -> list:
        """Return a loss row padded with zeros if length doesn't match epochs."""
        row = self.history['train_losses'][k]
        if not row:
            return []
        if len(row) != len(epochs):
            log.warning(
                f"[Metrics] '{k}' has {len(row)} entries vs "
                f"{len(epochs)} epochs — padding with zeros.")
            row = list(row) + [0.0] * (len(epochs) - len(row))
        return row

    def _simple_line(self, ax, data, val_epochs, colour, label, marker='D'):
        """Draw a single validation-metric line with best-point marker."""
        if not data:
            return
        ax.plot(val_epochs[:len(data)], data,
                color=colour, linewidth=2, marker=marker,
                markersize=4, label=label)

    # PLOT 1 — Train vs Validation Total Loss
    def plot_total_loss(self):
        fig, ax = plt.subplots(figsize=(10, 5), facecolor='white')
        ax.set_facecolor('white')
        epochs     = self._epochs()
        val_epochs = self._val_epochs()

        ax.plot(epochs, self.history['train_total'],
                color=COLOURS['train'], marker='o', markersize=4,
                linewidth=2, label='Train Loss')

        if self.history['val_total']:
            ax.plot(val_epochs, self.history['val_total'],
                    color=COLOURS['val'], marker='s', markersize=4,
                    linewidth=2, label='Val Loss')
            best_idx = int(np.argmin(self.history['val_total']))
            best_ep  = val_epochs[best_idx]
            best_val = self.history['val_total'][best_idx]
            ax.axvline(best_ep, color='gold', linestyle='--', linewidth=1.5,
                       label=f'Best Val (Ep {best_ep}: {best_val:.4f})')
            ax.scatter([best_ep], [best_val], color='gold', zorder=5, s=100)

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Train vs Validation Loss (Total)',
                     fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=1)
        ax.set_ylim(bottom=0)
        plt.tight_layout()
        fig.savefig(self._fig_path('01_total_loss'), **_SAVE_KW)
        plt.close(fig)
        print(f"[Metrics] Saved plot: 01_total_loss")

    # PLOT 2 — Individual Loss Terms (Training)
    def plot_individual_losses_train(self):
        fig, axes = plt.subplots(3, 3, figsize=(15, 12), facecolor='white')
        axes   = axes.flatten()
        epochs = self._epochs()

        axes[0].set_facecolor('white')
        axes[0].plot(epochs, self.history['train_total'],
                     color=COLOURS['train'], linewidth=2,
                     marker='o', markersize=3)
        axes[0].set_title('Total Loss (Train)', fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(bottom=0)

        for i, k in enumerate(LOSS_KEYS):
            ax   = axes[i + 1]
            ax.set_facecolor('white')
            data = self.history['train_losses'][k]
            if data:
                ax.plot(epochs[:len(data)], data,
                        color=COLOURS.get(k, '#555'),
                        linewidth=2, marker='o', markersize=3)
                if len(data) > 3:
                    z = np.polyfit(range(len(data)), data, 1)
                    p = np.poly1d(z)
                    ax.plot(epochs[:len(data)], p(range(len(data))),
                            '--', color='grey', linewidth=1, alpha=0.7)
            ax.set_title(LOSS_LABELS.get(k, k), fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(bottom=0)

        axes[8].set_visible(False)
        for ax in axes[:8]:
            ax.set_xlabel('Epoch', fontsize=9)
            ax.set_ylabel('Loss', fontsize=9)

        fig.suptitle('Individual Loss Terms — Training',
                     fontsize=15, fontweight='bold', y=1.01)
        plt.tight_layout()
        fig.savefig(self._fig_path('02_individual_losses_train'), **_SAVE_KW)
        plt.close(fig)
        print(f"[Metrics] Saved plot: 02_individual_losses_train")

    # PLOT 3 — Individual Loss Terms (Train vs Val)
    def plot_individual_losses_comparison(self):
        val_epochs = self._val_epochs()
        if not val_epochs:
            return

        fig, axes = plt.subplots(2, 4, figsize=(18, 9), facecolor='white')
        axes   = axes.flatten()
        epochs = self._epochs()

        for i, k in enumerate(LOSS_KEYS):
            ax     = axes[i]
            ax.set_facecolor('white')
            t_data = self.history['train_losses'][k]
            v_data = self.history['val_losses'][k]
            if t_data:
                ax.plot(epochs[:len(t_data)], t_data,
                        color=COLOURS['train'], linewidth=2,
                        marker='o', markersize=3, label='Train')
            if v_data:
                ax.plot(val_epochs[:len(v_data)], v_data,
                        color=COLOURS['val'], linewidth=2,
                        marker='s', markersize=3, label='Val')
            ax.set_title(LOSS_LABELS.get(k, k), fontweight='bold', fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(bottom=0)
            ax.set_xlabel('Epoch', fontsize=8)

        axes[7].set_visible(False)
        fig.suptitle('Train vs Val — Individual Loss Terms',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        fig.savefig(self._fig_path('03_individual_losses_comparison'), **_SAVE_KW)
        plt.close(fig)
        print(f"[Metrics] Saved plot: 03_individual_losses_comparison")

    # PLOT 4 — PSNR Curve
    def plot_psnr(self):
        if not self.history['psnr']:
            return
        val_epochs = self._val_epochs()
        fig, ax = plt.subplots(figsize=(9, 4), facecolor='white')
        ax.set_facecolor('white')
        ax.plot(val_epochs, self.history['psnr'],
                color=COLOURS['psnr'], linewidth=2,
                marker='D', markersize=5, label='PSNR (dB)')

        best_idx = int(np.argmax(self.history['psnr']))
        ax.scatter([val_epochs[best_idx]], [self.history['psnr'][best_idx]],
                   color='gold', zorder=5, s=120,
                   label=f"Best: {self.history['psnr'][best_idx]:.2f} dB "
                         f"(Ep {val_epochs[best_idx]})")

        for ref, label in [(25, 'Acceptable'), (30, 'Good'), (35, 'Excellent')]:
            ax.axhline(ref, linestyle=':', color='grey', linewidth=1, alpha=0.6)
            ax.text(val_epochs[0], ref + 0.3, label,
                    fontsize=8, color='grey', alpha=0.8)

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('PSNR (dB)', fontsize=12)
        ax.set_title('Peak Signal-to-Noise Ratio (PSNR) — Validation',
                     fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(self._fig_path('04_psnr'), **_SAVE_KW)
        plt.close(fig)
        print(f"[Metrics] Saved plot: 04_psnr")

    # PLOT 5 — SSIM Curve
    def plot_ssim(self):
        if not self.history['ssim']:
            return
        val_epochs = self._val_epochs()
        fig, ax = plt.subplots(figsize=(9, 4), facecolor='white')
        ax.set_facecolor('white')
        ax.plot(val_epochs, self.history['ssim'],
                color=COLOURS['ssim'], linewidth=2,
                marker='D', markersize=5, label='SSIM')

        best_idx = int(np.argmax(self.history['ssim']))
        ax.scatter([val_epochs[best_idx]], [self.history['ssim'][best_idx]],
                   color='gold', zorder=5, s=120,
                   label=f"Best: {self.history['ssim'][best_idx]:.4f} "
                         f"(Ep {val_epochs[best_idx]})")

        ax.axhline(0.9, linestyle=':', color='grey', linewidth=1, alpha=0.6)
        ax.text(val_epochs[0], 0.905, 'Target ≥ 0.90',
                fontsize=8, color='grey', alpha=0.8)

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('SSIM', fontsize=12)
        ax.set_ylim(0, 1.05)
        ax.set_title('Structural Similarity Index (SSIM) — Validation',
                     fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(self._fig_path('05_ssim'), **_SAVE_KW)
        plt.close(fig)
        print(f"[Metrics] Saved plot: 05_ssim")

    # PLOT 6 — Mesh Regularity
    def plot_mesh_regularity(self):
        if not self.history['mesh_reg']:
            return
        val_epochs = self._val_epochs()
        fig, ax = plt.subplots(figsize=(9, 4), facecolor='white')
        ax.set_facecolor('white')
        ax.plot(val_epochs, self.history['mesh_reg'],
                color=COLOURS['mesh_reg'], linewidth=2,
                marker='^', markersize=5, label='Mesh Regularity')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Mesh Regularity (↑ better)', fontsize=12)
        ax.set_title('Mesh Regularity (Edge Length Variance) — Validation',
                     fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()   # lower raw value = more regular = shown higher
        plt.tight_layout()
        fig.savefig(self._fig_path('06_mesh_regularity'), **_SAVE_KW)
        plt.close(fig)
        print(f"[Metrics] Saved plot: 06_mesh_regularity")

    # PLOT 7 — Learning Rate Schedule
    def plot_learning_rate(self):
        if not self.history['lr']:
            return
        epochs = self._epochs()
        fig, ax = plt.subplots(figsize=(9, 4), facecolor='white')
        ax.set_facecolor('white')
        ax.semilogy(epochs, self.history['lr'],
                    color=COLOURS['lr'], linewidth=2, marker='o', markersize=3)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Learning Rate (log scale)', fontsize=12)
        ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, which='both')
        plt.tight_layout()
        fig.savefig(self._fig_path('07_learning_rate'), **_SAVE_KW)
        plt.close(fig)
        print(f"[Metrics] Saved plot: 07_learning_rate")

    # PLOT 8 — GradScaler Scale Factor
    def plot_scaler_scale(self):
        if not self.history['scaler_scale']:
            return
        epochs = self._epochs()
        fig, ax = plt.subplots(figsize=(9, 4), facecolor='white')
        ax.set_facecolor('white')
        ax.semilogy(epochs, self.history['scaler_scale'],
                    color=COLOURS['scale'], linewidth=2,
                    marker='o', markersize=3)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('GradScaler Scale (log scale)', fontsize=12)
        ax.set_title('GradScaler Scale Factor\n'
                     '(drops indicate NaN/Inf gradients were detected)',
                     fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, which='both')
        scales = np.array(self.history['scaler_scale'])
        for i in range(1, len(scales)):
            if scales[i] < scales[i - 1] * 0.9:
                ax.axvspan(epochs[i] - 0.5, epochs[i] + 0.5,
                           color='red', alpha=0.15)
        plt.tight_layout()
        fig.savefig(self._fig_path('08_scaler_scale'), **_SAVE_KW)
        plt.close(fig)
        print(f"[Metrics] Saved plot: 08_scaler_scale")

    # PLOT 9 — Train/Val Gap (Overfitting Monitor)
    def plot_overfitting_gap(self):
        if not self.history['val_total']:
            return
        n   = min(len(self.history['train_total']),
                  len(self.history['val_total']))
        ep  = self._epochs()[:n]
        gap = [v - t for t, v in zip(
            self.history['train_total'][:n],
            self.history['val_total'][:n])]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor='white')
        for ax in axes:
            ax.set_facecolor('white')

        axes[0].plot(ep, self.history['train_total'][:n],
                     color=COLOURS['train'], linewidth=2,
                     marker='o', markersize=3, label='Train')
        axes[0].plot(ep, self.history['val_total'][:n],
                     color=COLOURS['val'], linewidth=2,
                     marker='s', markersize=3, label='Val')
        axes[0].fill_between(ep,
                             self.history['train_total'][:n],
                             self.history['val_total'][:n],
                             alpha=0.15, color=COLOURS['gap'], label='Gap')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Train vs Val Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(bottom=0)

        bar_colours = [COLOURS['gap'] if g > 0 else COLOURS['train']
                       for g in gap]
        axes[1].bar(ep, gap, color=bar_colours, alpha=0.7)
        axes[1].axhline(0, color='black', linewidth=0.8)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Val Loss − Train Loss')
        axes[1].set_title('Overfitting Gap\n'
                          '(positive = val > train = overfitting)')
        axes[1].grid(True, alpha=0.3)

        fig.suptitle('Overfitting Monitor', fontsize=14, fontweight='bold')
        plt.tight_layout()
        fig.savefig(self._fig_path('09_overfitting_gap'), **_SAVE_KW)
        plt.close(fig)
        print(f"[Metrics] Saved plot: 09_overfitting_gap")

    # PLOT 10 — Loss Heatmap (all 7 losses across all epochs)
    def plot_loss_heatmap(self):
        epochs = self._epochs()
        if len(epochs) < 2:
            return

        data   = []
        labels = []
        for k in LOSS_KEYS:
            row = self._safe_loss_row(k, epochs)
            if not row:
                continue
            arr = np.array(row, dtype=np.float32)
            arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
            data.append(arr)
            labels.append(LOSS_LABELS.get(k, k))

        if not data:
            return

        data = np.array(data)
        cmap = LinearSegmentedColormap.from_list(
            'facearcs', ['#1565C0', '#FFFFFF', '#C62828'])

        fig, ax = plt.subplots(
            figsize=(max(12, len(epochs) * 0.4), 5), facecolor='white')
        ax.set_facecolor('white')
        im = ax.imshow(data, aspect='auto', cmap=cmap,
                       vmin=0, vmax=1, interpolation='nearest')
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_xticks(range(len(epochs)))
        ax.set_xticklabels(epochs, fontsize=8, rotation=45)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_title('Loss Heatmap (row-normalised) — Training\n'
                     'Blue = low (good), Red = high (needs attention)',
                     fontsize=13, fontweight='bold')
        plt.colorbar(im, ax=ax, label='Normalised loss', shrink=0.8)
        plt.tight_layout()
        fig.savefig(self._fig_path('10_loss_heatmap'), **_SAVE_KW)
        plt.close(fig)
        print(f"[Metrics] Saved plot: 10_loss_heatmap")

    # PLOT 11 — Pie chart: loss contribution at last epoch
    def plot_loss_contribution_pie(self):
        contributions = {}
        for k in LOSS_KEYS:
            d = self.history['train_losses'][k]
            if d:
                contributions[k] = abs(d[-1])

        if not contributions:
            return

        labels = [LOSS_LABELS.get(k, k) for k in contributions]
        sizes  = list(contributions.values())
        colors = [COLOURS.get(k, '#999') for k in contributions]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor='white')
        for ax in axes:
            ax.set_facecolor('white')

        wedges, texts, autotexts = axes[0].pie(
            sizes, labels=labels, colors=colors,
            autopct='%1.1f%%', startangle=140,
            pctdistance=0.8, textprops={'fontsize': 9}
        )
        for at in autotexts:
            at.set_fontsize(8)
        axes[0].set_title(f'Loss Contribution\n(Epoch {self._epochs()[-1]})',
                          fontweight='bold')

        bars = axes[1].barh(labels, sizes, color=colors,
                            alpha=0.8, edgecolor='white')
        axes[1].set_xlabel('Loss Value', fontsize=11)
        axes[1].set_title('Loss Magnitude Comparison\n(last epoch)',
                          fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='x')
        for bar, val in zip(bars, sizes):
            axes[1].text(bar.get_width() + bar.get_width() * 0.02,
                         bar.get_y() + bar.get_height() / 2,
                         f'{val:.4f}', va='center', fontsize=9)

        fig.suptitle('Loss Term Contribution Analysis',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        fig.savefig(self._fig_path('11_loss_contribution'), **_SAVE_KW)
        plt.close(fig)
        print(f"[Metrics] Saved plot: 11_loss_contribution")

    # PLOT 12 — All Metrics Combined Dashboard
    def plot_dashboard(self):
        epochs     = self._epochs()
        val_epochs = self._val_epochs()
        if not epochs:
            return

        fig = plt.figure(figsize=(20, 14), facecolor='white')
        gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)

        # Row 0: Total loss / PSNR / SSIM / LR 
        ax_total = fig.add_subplot(gs[0, 0])
        ax_total.set_facecolor('white')
        ax_total.plot(epochs, self.history['train_total'],
                      color=COLOURS['train'], linewidth=2,
                      marker='o', markersize=3, label='Train')
        if self.history['val_total']:
            ax_total.plot(val_epochs, self.history['val_total'],
                          color=COLOURS['val'], linewidth=2,
                          marker='s', markersize=3, label='Val')
        ax_total.set_title('Total Loss', fontweight='bold')
        ax_total.legend(fontsize=8)
        ax_total.grid(True, alpha=0.3)
        ax_total.set_ylim(bottom=0)

        ax_psnr = fig.add_subplot(gs[0, 1])
        ax_psnr.set_facecolor('white')
        if self.history['psnr']:
            ax_psnr.plot(val_epochs, self.history['psnr'],
                         color=COLOURS['psnr'], linewidth=2,
                         marker='D', markersize=4)
        ax_psnr.set_title('PSNR (dB)', fontweight='bold')
        ax_psnr.grid(True, alpha=0.3)

        ax_ssim = fig.add_subplot(gs[0, 2])
        ax_ssim.set_facecolor('white')
        if self.history['ssim']:
            ax_ssim.plot(val_epochs, self.history['ssim'],
                         color=COLOURS['ssim'], linewidth=2,
                         marker='D', markersize=4)
        ax_ssim.set_title('SSIM', fontweight='bold')
        ax_ssim.set_ylim(0, 1.05)
        ax_ssim.grid(True, alpha=0.3)

        ax_lr = fig.add_subplot(gs[0, 3])
        ax_lr.set_facecolor('white')
        if self.history['lr']:
            ax_lr.semilogy(epochs, self.history['lr'],
                           color=COLOURS['lr'], linewidth=2,
                           marker='o', markersize=3)
        ax_lr.set_title('Learning Rate', fontweight='bold')
        ax_lr.grid(True, alpha=0.3, which='both')

        # Row 1: First 4 individual train losses 
        for i, k in enumerate(LOSS_KEYS[:4]):
            ax = fig.add_subplot(gs[1, i])
            ax.set_facecolor('white')
            d  = self.history['train_losses'][k]
            if d:
                ax.plot(epochs[:len(d)], d,
                        color=COLOURS.get(k, '#555'),
                        linewidth=2, marker='o', markersize=3)
            ax.set_title(LOSS_LABELS.get(k, k), fontweight='bold', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(bottom=0)

        # Row 2: Remaining 3 losses / cosine identity / NME / gap 
        for i, k in enumerate(LOSS_KEYS[4:]):
            ax = fig.add_subplot(gs[2, i])
            ax.set_facecolor('white')
            d  = self.history['train_losses'][k]
            if d:
                ax.plot(epochs[:len(d)], d,
                        color=COLOURS.get(k, '#555'),
                        linewidth=2, marker='o', markersize=3)
            ax.set_title(LOSS_LABELS.get(k, k), fontweight='bold', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(bottom=0)

        ax_gap = fig.add_subplot(gs[2, 3])
        ax_gap.set_facecolor('white')
        if self.history['val_total']:
            n   = min(len(self.history['train_total']),
                      len(self.history['val_total']))
            gap = [v - t for t, v in zip(
                self.history['train_total'][:n],
                self.history['val_total'][:n])]
            bar_colours = [COLOURS['val'] if g > 0 else COLOURS['train']
                           for g in gap]
            ax_gap.bar(epochs[:n], gap, color=bar_colours, alpha=0.7)
            ax_gap.axhline(0, color='black', linewidth=0.8)
        ax_gap.set_title('Val−Train Gap', fontweight='bold', fontsize=9)
        ax_gap.grid(True, alpha=0.3)

        for ax in fig.get_axes():
            ax.set_xlabel('Epoch', fontsize=8)

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
        if self.history['val_total']:
            title = (
                f'FaceARCs Training Dashboard — {timestamp}\n'
                f'Epochs completed: {len(epochs)}  |  '
                f'Best val loss: {min(self.history["val_total"]):.4f}'
            )
        else:
            title = (
                f'FaceARCs Training Dashboard — {timestamp}\n'
                f'Epochs completed: {len(epochs)}'
            )
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.01)
        fig.savefig(self._fig_path('12_dashboard'), **_SAVE_KW)
        plt.close(fig)
        print(f"[Metrics] Saved plot: 12_dashboard")

    # PLOT 13 — Cosine Identity & NME 2D
    def plot_identity_and_landmarks(self):
        val_epochs = self._val_epochs()
        if not val_epochs:
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor='white')
        for ax in axes:
            ax.set_facecolor('white')

        # Cosine Identity (higher is better)
        if self.history['cosine_identity']:
            data = self.history['cosine_identity']
            axes[0].plot(val_epochs[:len(data)], data,
                         color=COLOURS['cosine_identity'], linewidth=2,
                         marker='D', markersize=5, label='Cosine Identity')
            best_idx = int(np.argmax(data))
            axes[0].scatter(
                [val_epochs[best_idx]], [data[best_idx]],
                color='gold', zorder=5, s=120,
                label=f"Best: {data[best_idx]:.4f} (Ep {val_epochs[best_idx]})")
            axes[0].axhline(0.7, linestyle=':', color='grey',
                            linewidth=1, alpha=0.6)
            axes[0].text(val_epochs[0], 0.715, 'Target ≥ 0.70',
                         fontsize=8, color='grey', alpha=0.8)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Cosine Similarity', fontsize=12)
        axes[0].set_ylim(-1.05, 1.05)
        axes[0].set_title('ArcFace Cosine Identity — Validation\n'
                          '(higher is better, target > 0.70)',
                          fontsize=13, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)

        # NME 2D (lower is better)
        if self.history['nme_2d']:
            data = self.history['nme_2d']
            axes[1].plot(val_epochs[:len(data)], data,
                         color=COLOURS['nme_2d'], linewidth=2,
                         marker='^', markersize=5, label='NME 2D')
            best_idx = int(np.argmin(data))
            axes[1].scatter(
                [val_epochs[best_idx]], [data[best_idx]],
                color='gold', zorder=5, s=120,
                label=f"Best: {data[best_idx]:.4f} (Ep {val_epochs[best_idx]})")
            axes[1].axhline(0.05, linestyle=':', color='grey',
                            linewidth=1, alpha=0.6)
            axes[1].text(val_epochs[0], 0.052, 'Target < 0.05',
                         fontsize=8, color='grey', alpha=0.8)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('NME', fontsize=12)
        axes[1].set_title('Normalised Mean Error 2D Landmarks — Validation\n'
                          '(lower is better, target < 0.05)',
                          fontsize=13, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)

        fig.suptitle('Identity & Landmark Metrics', fontsize=14, fontweight='bold')
        plt.tight_layout()
        fig.savefig(self._fig_path('13_identity_landmarks'), **_SAVE_KW)
        plt.close(fig)
        print(f"[Metrics] Saved plot: 13_identity_landmarks")

    # PLOT 14 — MAE / Normal Consistency / Mean Edge Length
    def plot_geometry_and_mae(self):
        val_epochs = self._val_epochs()
        if not val_epochs:
            return

        fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor='white')
        for ax in axes:
            ax.set_facecolor('white')

        # MAE (lower is better)
        if self.history['mae']:
            data = self.history['mae']
            axes[0].plot(val_epochs[:len(data)], data,
                         color=COLOURS['mae'], linewidth=2,
                         marker='o', markersize=4, label='MAE')
        axes[0].set_xlabel('Epoch', fontsize=11)
        axes[0].set_ylabel('MAE', fontsize=11)
        axes[0].set_title('Mean Absolute Error — Validation\n(lower is better)',
                          fontsize=12, fontweight='bold')
        axes[0].legend(fontsize=9)
        axes[0].grid(True, alpha=0.3)

        # Normal Consistency (higher is better, target > 0.9)
        if self.history['normal_consistency']:
            data = self.history['normal_consistency']
            axes[1].plot(val_epochs[:len(data)], data,
                         color=COLOURS['normal_consistency'], linewidth=2,
                         marker='s', markersize=4, label='Normal Consistency')
            axes[1].axhline(0.9, linestyle=':', color='grey',
                            linewidth=1, alpha=0.6)
            axes[1].text(val_epochs[0], 0.905, 'Target ≥ 0.90',
                         fontsize=8, color='grey', alpha=0.8)
        axes[1].set_xlabel('Epoch', fontsize=11)
        axes[1].set_ylabel('Cosine Similarity', fontsize=11)
        axes[1].set_ylim(-1.05, 1.05)
        axes[1].set_title('Normal Consistency — Validation\n(higher is better)',
                          fontsize=12, fontweight='bold')
        axes[1].legend(fontsize=9)
        axes[1].grid(True, alpha=0.3)

        # Mean Edge Length (should be stable — big changes = mesh collapse)
        if self.history['mean_edge_length']:
            data = self.history['mean_edge_length']
            axes[2].plot(val_epochs[:len(data)], data,
                         color=COLOURS['mean_edge_length'], linewidth=2,
                         marker='^', markersize=4, label='Mean Edge Length')
            if len(data) > 1:
                baseline = data[0]
                axes[2].axhline(baseline, linestyle='--', color='grey',
                                linewidth=1, alpha=0.6,
                                label=f'Baseline (Ep {val_epochs[0]}): {baseline:.4f}')
        axes[2].set_xlabel('Epoch', fontsize=11)
        axes[2].set_ylabel('Edge Length', fontsize=11)
        axes[2].set_title('Mean Edge Length — Validation\n'
                          '(large change = mesh collapse warning)',
                          fontsize=12, fontweight='bold')
        axes[2].legend(fontsize=9)
        axes[2].grid(True, alpha=0.3)

        fig.suptitle('Geometry & Pixel-level Metrics', fontsize=14, fontweight='bold')
        plt.tight_layout()
        fig.savefig(self._fig_path('14_geometry_mae'), **_SAVE_KW)
        plt.close(fig)
        print(f"[Metrics] Saved plot: 14_geometry_mae")

    # Reconstruction Grid (called from val_epoch in train.py via visualise.py)
    def save_reconstruction_grid(self,
                                  inputs:    torch.Tensor,
                                  rendered:  torch.Tensor,
                                  verts:     torch.Tensor,
                                  epoch:     int,
                                  n_samples: int = 4):
        """Save a grid of: input | rendered | depth map."""
        recon_dir = self.plots_dir / 'reconstructions'
        recon_dir.mkdir(exist_ok=True)

        n = min(n_samples, inputs.shape[0])
        fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n), facecolor='white')
        if n == 1:
            axes = axes[np.newaxis, :]

        for i in range(n):
            for ax in axes[i]:
                ax.set_facecolor('white')

            inp = inputs[i].detach().cpu()
            if inp.shape[0] == 3:
                inp = inp.permute(1, 2, 0)
            axes[i, 0].imshow(inp.numpy().clip(0, 1))
            axes[i, 0].set_title('Input', fontweight='bold')
            axes[i, 0].axis('off')

            rend = rendered[i].detach().cpu()
            if rend.shape[0] == 3:
                rend = rend.permute(1, 2, 0)
            axes[i, 1].imshow(rend.numpy().clip(0, 1))
            axes[i, 1].set_title('Rendered', fontweight='bold')
            axes[i, 1].axis('off')

            v      = verts[i].detach().cpu().numpy()
            z      = v[:, 2]
            z_norm = (z - z.min()) / (z.max() - z.min() + 1e-8)
            sc = axes[i, 2].scatter(
                v[::15, 0], v[::15, 1],
                c=z_norm[::15], cmap='plasma', s=0.8, alpha=0.7
            )
            axes[i, 2].set_title('Depth Map (Z)', fontweight='bold')
            axes[i, 2].axis('off')
            axes[i, 2].invert_yaxis()
            plt.colorbar(sc, ax=axes[i, 2], shrink=0.8, label='depth')

        plt.suptitle(f'Reconstruction — Epoch {epoch}',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        path = recon_dir / f'reconstruction_epoch_{epoch:03d}.png'
        fig.savefig(path, dpi=130, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"[Metrics] Saved reconstruction: epoch {epoch:03d}")

    # save_all_plots — call at end of each epoch
    def save_all_plots(self):
        """Save every plot. Safe to call after every epoch."""
        self.plot_total_loss()
        self.plot_individual_losses_train()
        self.plot_individual_losses_comparison()
        self.plot_psnr()
        self.plot_ssim()
        self.plot_mesh_regularity()
        self.plot_learning_rate()
        self.plot_scaler_scale()
        self.plot_overfitting_gap()
        self.plot_loss_heatmap()
        self.plot_loss_contribution_pie()
        self.plot_dashboard()
        self.plot_identity_and_landmarks()   # new
        self.plot_geometry_and_mae()         # new
        print(f"[Metrics] All plots saved → {self.plots_dir}")

    # Print summary table to console
    def print_summary(self):
        epochs = self._epochs()
        if not epochs:
            return

        print(f"  FaceARCs Training Summary — {len(epochs)} epochs completed")
        print(f"  {'Metric':<30} {'Current':>10} {'Best':>10} {'Epoch':>8}")

        def _row(name, data, ep_list=None, higher_is_better=False):
            if not data:
                return
            ep  = (ep_list or epochs)[:len(data)]
            cur = data[-1]
            best_idx = int(np.argmax(data) if higher_is_better
                           else np.argmin(data))
            print(f"  {name:<30} {cur:>10.4f} {data[best_idx]:>10.4f}"
                  f" {ep[best_idx]:>8}")

        val_ep = self._val_epochs()

        # Losses
        _row("Train Loss (total)",     self.history['train_total'])
        _row("Val Loss (total)",       self.history['val_total'],   val_ep)

        # Image quality
        _row("PSNR (dB)",              self.history['psnr'],        val_ep, True)
        _row("SSIM",                   self.history['ssim'],        val_ep, True)
        _row("MAE",                    self.history['mae'],         val_ep)

        # Geometry
        _row("Mesh Regularity",        self.history['mesh_reg'],    val_ep)
        _row("Normal Consistency",     self.history['normal_consistency'], val_ep, True)
        _row("Mean Edge Length",       self.history['mean_edge_length'],   val_ep)

        # Identity / landmarks
        _row("Cosine Identity",        self.history['cosine_identity'], val_ep, True)
        _row("NME 2D",                 self.history['nme_2d'],          val_ep)

        # Per-term train losses
        for k in LOSS_KEYS:
            _row(f"  └ {LOSS_LABELS.get(k, k)[:26]}",
                 self.history['train_losses'][k])