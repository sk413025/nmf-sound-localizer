# Nature Communications 論文 Worktree

本 worktree 包含 Nature Communications 論文的所有稿件檔案。

## 專案資訊

**論文標題**: Physics-Informed Neural Routing for Acoustic Localization

**核心結果**: 在 37 類聲學定位任務上達到 93.5% 準確率

**關鍵發現**: 物理與學習的協同是必要的—單獨任一者都無法有效工作

## Worktree 結構

- `manuscript/`: 稿件主要檔案 (Markdown 格式)
- `evidence_tracking/`: 連結稿件主張與實驗證據
- `scripts/`: 編譯與驗證腳本
- `writing_notes/`: 寫作工作區 (中文)

## 實驗證據譜系

所有稿件主張都連結到 development-workspace 中的特定 git commits。
詳見 `evidence_tracking/` 目錄。

**關鍵實驗 commits**:
- Master figure: 872aa65
- 消融實驗: b9dcafa
- SNR 穩健性: bd88710, cfdc4d9, e37f512

## 工作流程

### 撰寫稿件
1. 在 `manuscript/main_text/` 中撰寫 Markdown
2. 更新 `evidence_tracking/` 中的證據追蹤
3. 定期檢查字數: `python scripts/check_word_counts.py`
4. 檢查引用數量: `python scripts/validate_references.py`
5. 提交變更: `git commit -m "稿件: [章節] - [變更]"`

## 當前狀態

詳見 `提交狀態.md`

## 圖表連結

所有圖表都是從 development-workspace 連結過來的 (symlinks)，
避免複製並保持單一真實來源。

主要圖表:
- Figure 1: Master figure (已連結)
- Figure 4: Ablation study (已連結)

補充圖表:
- Supplementary Figure S1: Complete ablation (已連結)
- Supplementary Table S1: Ablation details (已連結)

## 提交流程

1. 完成所有章節初稿
2. 內部審閱
3. PI 審閱
4. 編譯最終版本
5. 完成提交材料
6. 透過 Nature Communications 入口網站提交

詳見 `manuscript/submission/submission_checklist.md`
