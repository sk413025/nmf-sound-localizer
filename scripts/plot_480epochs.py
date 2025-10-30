#!/usr/bin/env python3
import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

log_path = Path('results/dt_min_480epochs_20251030_223037/training.log')
out_dir = log_path.parent

train_losses = []
test_losses = []
train_e = []
train_m = []
test_e = []
test_m = []
epochs = []

lines = log_path.read_text().splitlines()
i = 0
while i < len(lines):
    m = re.match(r"Epoch\s+(\d+)/(\d+):", lines[i])
    if m:
        epoch = int(m.group(1))
        tl = lines[i+1] if i+1 < len(lines) else ''
        ta = lines[i+2] if i+2 < len(lines) else ''
        tb = lines[i+3] if i+3 < len(lines) else ''
        lm = re.search(r"Train loss:\s*([0-9.]+) \| Test loss:\s*([0-9.]+)", tl)
        if lm:
            t_loss = float(lm.group(1)); v_loss = float(lm.group(2))
        else:
            t_loss = np.nan; v_loss = np.nan
        am = re.search(r"Train acc:\s*expert=([0-9.]+), atom=([0-9.]+)", ta)
        if am:
            te = float(am.group(1)); tm = float(am.group(2))
        else:
            te = np.nan; tm = np.nan
        bm = re.search(r"Test acc:\s*expert=([0-9.]+), atom=([0-9.]+)", tb)
        if bm:
            ve = float(bm.group(1)); vm = float(bm.group(2))
        else:
            ve = np.nan; vm = np.nan
        epochs.append(epoch)
        train_losses.append(t_loss)
        test_losses.append(v_loss)
        train_e.append(te)
        train_m.append(tm)
        test_e.append(ve)
        test_m.append(vm)
        i += 4
    else:
        i += 1

if not epochs:
    raise SystemExit('No epoch data parsed from log')

epochs = np.array(epochs)
train_losses = np.array(train_losses)
test_losses = np.array(test_losses)
train_e = np.array(train_e)
train_m = np.array(train_m)
test_e = np.array(test_e)
test_m = np.array(test_m)

# Loss plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(epochs, train_losses, label='Train loss', alpha=0.7)
ax1.plot(epochs, test_losses, label='Test loss', alpha=0.7, linewidth=2)
ax1.axvline(x=381, color='red', linestyle='--', alpha=0.5, label='Best epoch (381)')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.set_title('Train vs Test Loss (480 epochs)', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(alpha=0.3)

# Accuracy plot
ax2.plot(epochs, train_m, label='Train atom acc', alpha=0.6)
ax2.plot(epochs, test_m, label='Test atom acc', alpha=0.8, linewidth=2)
ax2.plot(epochs, train_e, label='Train expert acc', alpha=0.6)
ax2.plot(epochs, test_e, label='Test expert acc', alpha=0.8, linewidth=2)
ax2.axvline(x=381, color='red', linestyle='--', alpha=0.5, label='Best epoch (381)')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Accuracy', fontsize=12)
ax2.set_title('Per-step Accuracy (expert & atom)', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(out_dir / 'training_curves_480epochs.png', dpi=200, bbox_inches='tight')
plt.close()

# Also create zoomed view around best epoch
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
zoom_start = max(0, 381-100)
zoom_end = min(len(epochs), 381+100)
zoom_mask = (epochs >= zoom_start) & (epochs <= zoom_end)

ax1.plot(epochs[zoom_mask], train_losses[zoom_mask], label='Train loss', alpha=0.7, marker='o', markersize=2)
ax1.plot(epochs[zoom_mask], test_losses[zoom_mask], label='Test loss', alpha=0.7, linewidth=2, marker='s', markersize=2)
ax1.axvline(x=381, color='red', linestyle='--', alpha=0.5, label='Best epoch (381)')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.set_title('Loss (zoomed: epochs 280-480)', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(alpha=0.3)

ax2.plot(epochs[zoom_mask], test_m[zoom_mask], label='Test atom acc', alpha=0.8, linewidth=2, marker='s', markersize=2)
ax2.plot(epochs[zoom_mask], test_e[zoom_mask], label='Test expert acc', alpha=0.8, linewidth=2, marker='^', markersize=2)
ax2.axvline(x=381, color='red', linestyle='--', alpha=0.5, label='Best epoch (381)')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Test Accuracy', fontsize=12)
ax2.set_title('Test Accuracy (zoomed)', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(out_dir / 'training_curves_480epochs_zoomed.png', dpi=200, bbox_inches='tight')
plt.close()

print(f'Saved: {out_dir / "training_curves_480epochs.png"}')
print(f'Saved: {out_dir / "training_curves_480epochs_zoomed.png"}')
