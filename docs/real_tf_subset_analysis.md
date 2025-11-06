# 實測 H 子集分析：背景、方法、預期與結果、TDD 對齊與重現步驟

本文件記錄以實測資料（白噪音 X 與 LDV 量測震動 Y）在 80°–150°、每 5° 的角度子集上，估計與分析 Transfer Function H 的一次完整測試。內容包含：背景動機與方法設計、為何採用這些方法、預期與實際結果、如何解讀，以及如何重現此測試。最後說明此測試如何對齊 TDD 開發精神、是否能推動符合物理數學原理的實作。

---

## 背景與動機

- 物理定義：H 以 STFT 領域定義為 `H(f,θ) = |STFT(Y)/(STFT(X)+ε)|`，隨時間平均得到每角度的頻率響應。Y 為經過系統後的量測，X 為原始信號。
- 實測資料特性：
  - X 為純白噪音（頻譜平坦，利於 H 估計）。
  - Y 為 X 驅動盒子系統後由 LDV 量測到的震動訊號（聲-機構路徑的響應）。
  - 角度為 0°–180°、每 5°；目前 0°–75° 不足，先用 80°–150° 之子集。
  - X/Y 皆為 16 kHz，每角度 3 筆、成對錄製的檔案，`angle_XX` 為角度對應資料夾且一一對應同角度。
- 驗證目標：
  - 用實測資料驗證 H 估計流程（頻段限制、正規化、數值穩定）是否符合文件原理。
  - 在不引入偏置的情況下檢查角度可分辨性（角向差異是否顯著）。
  - 為後續 NMF 與定位步驟提供實測校準的基準與門檻。

---

## 方法與設計理由（為什麼這樣做）

- 角度子集選擇：只分析 80°–150° 的角度集合，因 0°–75° 資料尚不足。透過 symlink 建立子集，避免變更核心程式，且不複製大檔。
- STFT 與頻段設定：
  - `fs=16k, n_fft=2048, hop_length=512, window=hann`，與專案文件一致，平衡頻率解析與時間穩定性。
  - 僅保留 500–1500 Hz，對應語音關鍵頻段，亦可避開低頻噪聲與高頻干擾，符合文件建議。
- 正規化策略：
  - 採「每頻率在所有角度上的平均」作為參考（mean-normalization），避免指定角度作為參考帶來的偏置。
  - 之後做溫和全域縮放（約 [0.1, 0.9]），方便數值穩定與可視化；本次未開啟對比增強，避免改變角向排序的風險。
- 可分辨性度量：
  - 使用 `TransferFunctionProcessor.analyze_separability` 計算列向量 L2 正規化後的方向間相關係數（off-diagonal correlation）。此度量反映「頻率形狀相似度」，比僅看幅度更貼近「方向是否可區分」。
  - 提供每頻率的 range 平均值與角向平均響應的標準差，分別衡量頻率層級與角度層級的變異。
  - Condition number（數值條件）僅作參考，不作主要判斷，因其對尺度極敏感且未正規化可能造成解讀偏差。

---

## 預期結果（先於執行時的假設）

- 基本性質：H 非負、無 NaN/Inf；正規化後值域落在近似 [0.1, 0.9]。
- 維度與範圍：H 之頻率 bin 對應 500–1500 Hz 的長度；角度數為子集長度（80–150° 每 5°）。
- 角度依存性：角向平均響應（跨頻平均）在不同角度之間的標準差應大於小閾值（例如 >0.02），表示非退化為角向無差。
- 可分辨性：列向量 L2 正規化後的方向間平均相關係數明顯小於 1（例如 <0.98，視資料而調整）；每頻率 across angles 的 range 平均值應大於小閾值，顯示角向頻率響應具可分辨差異。

---

## 實際結果（本次執行）

- 角度列表：`[80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150]`（共 15 個）。
- H 形狀：`(129, 15)`（對應 500–1500 Hz 的 129 個頻率 bin）。
- 數值統計：`min=0.1000, max=0.9000, mean=0.3956`（符合 mean-normalization + 全域縮放）。
- 可分辨性指標：
  - mean off-diagonal correlation：`0.9716`
  - max/min off-diagonal correlation：`0.9944 / 0.9276`
  - mean_freq_range（每頻率之角向 range 的平均）：`0.4497`
  - angle response std（角向平均響應的標準差）：`0.1084`
  - condition_number（參考）：`≈ 77.96`
- 保存檔案：`out/real_tf_subset.pth`（包含 H、angles、config、meta、subset_report、separability）。

解讀：
- H 的值域、無 NaN/Inf 與頻段長度皆符合預期，mean-normalization 的效果正確。
- 角度依存性顯著（angle response std ≈ 0.1084），每頻率的角向差異也明顯（mean_freq_range ≈ 0.45）。
- 方向間相關性（0.97）表示方向頻譜形狀相似度高但可分，對於 5° 的細角度網格與中頻帶，屬合理現象。若需更強可分性，可考慮加入頻率權重或溫和對比增強（需先驗證不破壞排序）。
- condition number 僅作參考，避免過度解讀。

---

## 與 TDD 的對齊與推動實作合規

- 將原理規格化爲可度量的驗收條件：
  - 非負性/數值穩定（無 NaN/Inf）、頻段對齊（500–1500 Hz）、正規化（mean-normalization 後的值域與均值約束）、可分辨性（off-diagonal correlation 与 angle response std 的閾值）。
- 可轉換為 pytest 的 integration 測試：
  - 設定可調門檻（例如 mean correlation ≤ 0.98、angle response std ≥ 0.05 等），納入 CI 做回歸監控，防止未來修改破壞物理一致性。
- 實測與合成並用：
  - 已有合成數據測試驗證公式與流程，實測測試補足真實環境下的合理性範圍，兩者結合更符合 TDD “先規格→可驗證”的精神。
- 是否能推動符合原理的實作：
  - 可以。透過明確度量與固定門檻（可依探索性分析精煉），當實作偏離物理假設或數值穩定性時，測試會紅燈提示，促使在最小更動下恢復正確行爲。

---

## 重現步驟（完整指令）

前置：假設你有 conda 環境 `wavtokenizer`、並在專案根目錄。

1) 僅跑本次的實測子集分析（不修改任何核心碼）：

```bash
conda run -n wavtokenizer bash -lc \
  'export PYTHONPATH=$(pwd):$PYTHONPATH; \
   python scripts/analyze_real_tf_subset.py \
     --original "/Users/sbplab/jnrle/datasets/test_nmf_output_no_edge_with_original/white_noise_original_data_no_edge" \
     --box "/Users/sbplab/jnrle/datasets/test_nmf_output_no_edge_with_original/white_noise_box_data_no_edge" \
     --out out/real_tf_subset.pth \
     --angle-start 80 --angle-end 150 --angle-step 5 \
     --n-files 3'
```

執行後會輸出角度列表、H 統計與可分辨性指標，並保存結果至 `out/real_tf_subset.pth`。

2) 檢視保存結果（選擇性）：

```python
import torch
r = torch.load('out/real_tf_subset.pth', weights_only=False)
print(r['H'].shape, r['angles'])
print(r['separability'])
```

3) 相關的合成數據測試（可快速驗證流程正確性）：

```bash
# 僅跑新增的 H 流程測試檔（避免 coverage 外掛干擾）
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -c tests/pytest_no_cov.ini -q tests/test_transfer_function_pipeline.py
```

---

## 後續建議

- 加入 90° 參考正規化的對照測試，比對與 mean-normalization 的角向排序一致性（排序相關係數）。
- 依此實測分佈設定更精準的門檻，將此分析納入 pytest（加上 `@pytest.mark.integration`），成為回歸測試的一部分。
- 探索在 500–1500 Hz 內的頻率權重設計與溫和對比增強對角向可分性的提升，並新增“排序不變性”檢測，確保不破壞物理一致性。

---

本次測試將物理定義（Y/X 比值、頻段限制）、數值策略（ε 保護、正規化）、與可分辨性度量（形狀相似度）轉化為具體可驗證的指標與步驟，符合 TDD 將“原理→可度量規格→測試”的精神，有助於長期維持符合數學物理原理的實作品質。

