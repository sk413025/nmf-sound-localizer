# Train/Test Split for Decision Transformer

## 更新說明 (2025-10-29)

為了監控和防止 overfitting，我已經更新了 `scripts/dt_pointer_ldv.py` 加入以下功能：

### 主要改進

1. **訓練/測試集切分**
   - 使用 `--test_split` 參數控制測試集比例（預設 0.15，約 17/111 樣本）
   - 採用分層切分（stratified split），確保每個角度在訓練和測試集都有代表性
   - 使用 `--split_seed` 確保可重現性

2. **每個 Epoch 的測試集評估**
   - 訓練和測試損失同時監控
   - 訓練和測試準確率（expert 和 atom）同時評估
   - 自動追蹤最佳測試損失和對應的 epoch

3. **Early Stopping 指標**
   - 保存最佳模型到 `ckpt_best.pth`
   - 在訓練日誌中標記最佳 epoch
   - 可視化訓練/測試損失差距

### 使用方法

#### 快速測試（10 epochs）

```bash
./test_train_test_split.sh
```

#### 自訂訓練

```bash
python -u scripts/dt_pointer_ldv.py \
  --traj_dir results/dt_traj_qk_kmeans \
  --out_dir results/your_experiment_name \
  --epochs 40 \
  --batch_size 4 \
  --test_split 0.15 \
  --split_seed 42 \
  --device cpu \
  2>&1 | tee results/your_experiment_name/run.log
```

#### 分析訓練曲線

訓練完成後，使用分析工具視覺化結果：

```bash
python scripts/analyze_training_curves.py \
  --ckpt results/your_experiment_name/ckpt_latest.pth \
  --out_dir results/your_experiment_name
```

這會生成：
- `loss_curves.pdf` - 訓練/測試損失曲線
- `accuracy_curves.pdf` - 訓練/測試準確率曲線
- 終端輸出包含 overfitting 警告和統計摘要

### 新增參數

```
--test_split FLOAT      測試集比例 (預設: 0.15)
--split_seed INT        隨機種子 (預設: 42)
```

### 輸出格式示例

```
================================================================================
Train/Test Split
================================================================================
Total samples: 111
Train samples: 94 (84.7%)
Test samples:  17 (15.3%)
Train steps: 564
Test steps:  102
Split seed: 42

================================================================================
Training Start
================================================================================
Epoch 1/10:
  Train loss: 6.4532 | Test loss: 6.5123 | Δ: +0.0591
  Train acc:  expert=0.381, atom=0.536
  Test acc:   expert=0.372, atom=0.529
  ✓ Best test loss so far!
```

### 如何判斷 Overfitting

觀察以下指標：

1. **測試損失不再下降**
   - 如果訓練損失持續降低，但測試損失停滯或上升 → overfitting

2. **訓練/測試損失差距擴大**
   - `Δ` 值持續增加表示模型記憶訓練數據

3. **分析工具警告**
   - `⚠️ Test loss has not improved for N epochs`
   - `⚠️ Large gap suggests overfitting`

### 建議流程

1. **先跑 10-20 epochs** 觀察趨勢
   ```bash
   ./test_train_test_split.sh  # 快速測試
   ```

2. **分析曲線**
   ```bash
   python scripts/analyze_training_curves.py \
     --ckpt results/dt_min_train_test_split_test/ckpt_latest.pth
   ```

3. **根據結果決定**
   - 如果測試損失仍在下降 → 可以增加 epochs
   - 如果測試損失已停滯 → 不要增加 epochs，考慮正則化或數據增強
   - 如果訓練/測試差距大 → 需要改進泛化能力

### 數據統計

- 總樣本：111
- 訓練集（85%）：94 樣本 × 6 步 = 564 訓練範例
- 測試集（15%）：17 樣本 × 6 步 = 102 測試範例
- 模型參數：320,640
- 訓練參數/範例比：568:1（仍然很高，需要密切監控）

### 下一步優化建議

如果發現 overfitting：

1. **增加正則化**
   - 增加 dropout（目前 0.1 → 0.2-0.3）
   - 增加 weight decay

2. **數據增強**
   - RTG 擾動
   - Residual 噪音注入

3. **減小模型**
   - d_model: 128 → 64
   - 減少參數到 ~80k

4. **收集更多數據**
   - 目標：1000+ 樣本
