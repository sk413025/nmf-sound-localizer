#!/usr/bin/env python3
"""檢查各章節字數是否符合 Nature Communications 要求"""

import re
from pathlib import Path

# Nature Communications 字數限制
LIMITS = {
    'abstract.md': 150,
    'introduction.md': 1500,
    'results.md': 2500,
    'discussion.md': 1200,
    'methods.md': 3000
}

def count_words(text):
    """計算字數 (只計算英文單字)"""
    # 移除 markdown 語法
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)  # 移除連結
    text = re.sub(r'!\[([^\]]+)\]\([^\)]+\)', '', text)    # 移除圖片
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL) # 移除程式碼區塊
    text = re.sub(r'`[^`]+`', '', text)                    # 移除行內程式碼
    text = re.sub(r'#+\s', '', text)                       # 移除標題符號
    text = re.sub(r'\*\*([^\*]+)\*\*', r'\1', text)        # 移除粗體
    text = re.sub(r'\*([^\*]+)\*', r'\1', text)            # 移除斜體

    # 計算英文單字
    words = re.findall(r'\b[a-zA-Z]+\b', text)
    return len(words)

def main():
    project_root = Path(__file__).parent.parent
    main_text_dir = project_root / 'manuscript' / 'main_text'

    print("字數檢查 (Nature Communications 限制)")
    print("=" * 60)

    total_words = 0
    all_within_limit = True

    for section, limit in LIMITS.items():
        section_path = main_text_dir / section

        if not section_path.exists():
            print(f"⚠  {section:20s} 檔案不存在")
            continue

        with open(section_path, 'r', encoding='utf-8') as f:
            text = f.read()

        word_count = count_words(text)
        total_words += word_count

        percentage = (word_count / limit) * 100
        status = "✓" if word_count <= limit else "✗"

        if word_count > limit:
            all_within_limit = False
            print(f"{status} {section:20s} {word_count:5d}/{limit:5d} ({percentage:5.1f}%) 超過限制!")
        else:
            print(f"{status} {section:20s} {word_count:5d}/{limit:5d} ({percentage:5.1f}%)")

    print("=" * 60)
    print(f"總字數: {total_words} 字 (目標 ~9,350 字)")

    if all_within_limit:
        print("\n✓ 所有章節都在字數限制內")
        return 0
    else:
        print("\n✗ 有章節超過字數限制，請修改")
        return 1

if __name__ == '__main__':
    exit(main())
