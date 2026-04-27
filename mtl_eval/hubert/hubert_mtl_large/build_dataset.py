import os
from pypinyin import pinyin, Style

ROOT_DIR = r''#数据目录

for root, dirs, files in os.walk(ROOT_DIR):
    for f in files:
        if f.endswith('.wav'):
            char = os.path.basename(os.path.dirname(root))  # 汉字文件夹名
            # 转为无声调拼音，仅供 MFA 物理对齐使用
            py = pinyin(char, style=Style.NORMAL, strict=False)[0][0]

            with open(os.path.join(root, f.replace('.wav', '.lab')), 'w', encoding='utf-8') as tf:
                tf.write(py)
print("Step 1 完成：已生成无声调 .lab 文件供 MFA 物理对齐。")