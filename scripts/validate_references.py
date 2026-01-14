#!/usr/bin/env python3
"""驗證引用數量 (Nature Communications 限制 ≤70)"""

import re
from pathlib import Path

def count_references(bib_file):
    """計算 .bib 檔案中的引用數量"""
    if not bib_file.exists():
        return 0

    with open(bib_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 計算 @article, @book, @inproceedings 等
    references = re.findall(r'^@\w+{', content, flags=re.MULTILINE)
    return len(references)

def main():
    project_root = Path(__file__).parent.parent
    bib_file = project_root / 'manuscript' / 'main_text' / 'references.bib'

    print("引用數量檢查")
    print("=" * 60)

    if not bib_file.exists():
        print("⚠ 找不到 references.bib")
        print("請建立 manuscript/main_text/references.bib")
        return 1

    ref_count = count_references(bib_file)
    limit = 70

    print(f"引用數量: {ref_count}/{limit}")

    if ref_count > limit:
        print(f"\n✗ 超過 Nature Communications 限制 ({ref_count - limit} 篇過多)")
        print("建議:")
        print("  1. 移除較不重要的引用")
        print("  2. 合併相關引用到同一個 citation")
        print("  3. 將部分引用移到補充材料")
        return 1
    elif ref_count == 0:
        print("\n⚠ 尚未加入引用")
        return 0
    else:
        print(f"\n✓ 在限制內 (剩餘 {limit - ref_count} 篇)")
        return 0

if __name__ == '__main__':
    exit(main())
