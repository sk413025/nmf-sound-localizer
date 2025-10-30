# 120 Epochs 訓練指令 - tmux 執行指南

## 🚀 快速啟動指令

### 方法 1: 一鍵啟動（推薦）

```bash
cd /Users/sbplab/jnrle/LDVReorientation/worktrees/mdp-decision-transformer && \
tmux new-session -d -s dt_train_120 './run_120epochs.sh' && \
echo "✓ Training started in tmux session 'dt_train_120'" && \
echo "To attach: tmux attach -t dt_train_120" && \
echo "To detach: Press Ctrl+B then D"
```

### 方法 2: 分步執行

```bash
# 1. 進入工作目錄
cd /Users/sbplab/jnrle/LDVReorientation/worktrees/mdp-decision-transformer

# 2. 創建新的 tmux session
tmux new-session -s dt_train_120

# 3. 在 tmux 中執行訓練
./run_120epochs.sh

# 4. 訓練開始後，按 Ctrl+B 然後按 D 來 detach
```

---

## 📋 tmux 常用操作

### 管理 session

```bash
# 查看所有 tmux sessions
tmux ls

# 連接到訓練 session
tmux attach -t dt_train_120

# 刪除 session（訓練完成後）
tmux kill-session -t dt_train_120
```

### 在 session 內操作

- **Detach（離開但保持運行）**: `Ctrl+B` 然後按 `D`
- **滾動查看歷史輸出**: `Ctrl+B` 然後按 `[`，用方向鍵滾動，按 `q` 退出
- **複製模式**: `Ctrl+B` 然後按 `[`

---

## 📊 訓練配置

| 參數 | 值 |
|------|-----|
| Epochs | 120 |
| Batch Size | 4 |
| Learning Rate | 3e-3 |
| Model Dimension | 128 |
| Test Split | 20% (seed: 42) |
| Device | CPU |
| Distillation Weight | 0.5 |

---

## 📁 輸出文件

訓練會在以下位置創建時間戳記目錄：
```
results/dt_min_120epochs_YYYYMMDD_HHMMSS/
├── training.log          # 完整訓練日誌
├── ckpt_latest.pth       # 最新檢查點
├── ckpt_best.pth         # 最佳模型（測試損失最低）
└── controllability.jsonl # 可控性診斷數據
```

---

## 🔍 監控訓練進度

### 即時查看訓練日誌

```bash
# 方法 1: 在 tmux 內查看（推薦）
tmux attach -t dt_train_120

# 方法 2: tail 最新的日誌文件
tail -f results/dt_min_120epochs_*/training.log

# 方法 3: 查看最近 50 行
tail -n 50 results/dt_min_120epochs_*/training.log
```

### 搜尋特定 epoch 結果

```bash
# 查看所有 epoch 摘要
grep "Epoch.*Test loss" results/dt_min_120epochs_*/training.log

# 查看最佳測試損失
grep "Best test loss" results/dt_min_120epochs_*/training.log
```

---

## ⏱️ 預估訓練時間

基於 5 epochs 的訓練時間估算：
- 如果 5 epochs ≈ 2 分鐘
- 120 epochs ≈ **48 分鐘**（約 0.8 小時）

**建議**: 
- 訓練開始後可以 detach tmux session
- 每 10-20 分鐘檢查一次進度
- 或者設置完成後的通知

---

## ⚠️ 重要提醒

1. **不要關閉終端**: 
   - Detach tmux session 後可以安全關閉終端視窗
   - 訓練會在背景持續運行

2. **檢查空間**:
   ```bash
   df -h .  # 確認有足夠磁碟空間
   ```

3. **緊急停止**:
   ```bash
   # 連接到 session
   tmux attach -t dt_train_120
   
   # 按 Ctrl+C 停止訓練
   ```

---

## 📈 訓練完成後的分析

訓練完成後，執行以下分析：

```bash
# 載入檢查點查看訓練歷史
python3 << 'EOF'
import torch
import matplotlib.pyplot as plt

# 找到最新的輸出目錄
import glob
dirs = sorted(glob.glob('results/dt_min_120epochs_*'))
if dirs:
    latest_dir = dirs[-1]
    print(f"分析目錄: {latest_dir}")
    
    ckpt = torch.load(f'{latest_dir}/ckpt_latest.pth', 
                      map_location='cpu', weights_only=False)
    
    history = ckpt['history']
    epochs = [h['epoch'] for h in history]
    train_losses = [h['train_loss'] for h in history]
    test_losses = [h['test_loss'] for h in history]
    
    # 繪製曲線
    plt.figure(figsize=(12, 5))
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('120 Epochs Training Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{latest_dir}/training_curves_120epochs.pdf', dpi=300)
    print(f"圖表已保存: {latest_dir}/training_curves_120epochs.pdf")
    
    # 顯示統計
    print(f"\n訓練摘要:")
    print(f"  總 epochs: {len(history)}")
    print(f"  最終訓練損失: {train_losses[-1]:.4f}")
    print(f"  最終測試損失: {test_losses[-1]:.4f}")
    print(f"  最佳測試損失: {min(test_losses):.4f} (Epoch {epochs[test_losses.index(min(test_losses))]})")
EOF
```

---

**準備好了嗎？執行上方的一鍵啟動指令即可開始訓練！** 🚀
