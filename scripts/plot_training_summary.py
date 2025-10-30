#!/usr/bin/env python3
import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

log_path = Path('results/dt_min_120epochs_20251029_202025/training.log')
out_dir = log_path.parent

# regex to capture epoch lines
epoch_re = re.compile(r"Epoch\s+(\d+)/(\d+):\n\s+Train loss:\s+([0-9.]+) \| Test loss:\s+([0-9.]+).*\n\s+Train acc:\s+expert=([0-9.]+), atom=([0-9.]+)\n\s+Test acc:\s+expert=([0-9.]+), atom=([0-9.]+)", re.M)

text = log_path.read_text()
# We'll parse iteratively because some logs have lines separated differently; safer to find lines per epoch
train_losses = []
test_losses = []
train_e = []
train_m = []
test_e = []
test_m = []
epochs = []

# simpler line-by-line parse
lines = text.splitlines()
i = 0
while i < len(lines):
    m = re.match(r"Epoch\s+(\d+)/(\d+):", lines[i])
    if m:
        epoch = int(m.group(1))
        # next lines expected pattern
        tl = lines[i+1] if i+1 < len(lines) else ''
        ta = lines[i+2] if i+2 < len(lines) else ''
        tb = lines[i+3] if i+3 < len(lines) else ''
        # parse losses
        lm = re.search(r"Train loss:\s*([0-9.]+) \| Test loss:\s*([0-9.]+)", tl)
        if lm:
            t_loss = float(lm.group(1)); v_loss = float(lm.group(2))
        else:
            t_loss = np.nan; v_loss = np.nan
        # parse train acc
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

# convert to numpy
epochs = np.array(epochs)
train_losses = np.array(train_losses)
test_losses = np.array(test_losses)
train_e = np.array(train_e)
train_m = np.array(train_m)
test_e = np.array(test_e)
test_m = np.array(test_m)

# Loss plot
plt.figure(figsize=(8,4))
plt.plot(epochs, train_losses, label='Train loss', marker='o')
plt.plot(epochs, test_losses, label='Test loss', marker='s')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train vs Test Loss (120 epochs)')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(out_dir / 'loss_curve_120epochs.png', dpi=200)
plt.close()

# Accuracy plot (atom acc and expert acc separately on same axes)
plt.figure(figsize=(8,4))
plt.plot(epochs, train_m, label='Train atom acc', marker='o')
plt.plot(epochs, test_m, label='Test atom acc', marker='s')
plt.plot(epochs, train_e, label='Train expert acc', marker='^')
plt.plot(epochs, test_e, label='Test expert acc', marker='v')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Per-step Accuracy (expert & atom)')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(out_dir / 'acc_curve_120epochs.png', dpi=200)
plt.close()

print('Saved plots:', out_dir / 'loss_curve_120epochs.png', out_dir / 'acc_curve_120epochs.png')
