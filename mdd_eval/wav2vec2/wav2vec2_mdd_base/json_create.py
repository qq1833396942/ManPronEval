import os
import json
import textgrid
import numpy as np
from pypinyin import pinyin, Style
from tqdm import tqdm

# ==========================================
# 1. 路径配置（填入你的 train/test/val 所在的根目录）
# ==========================================
ROOT_DIR = r''  # data数据集目录根目录下包含 train/test/val
SPLITS = ['train', 'test', 'val']


# ==========================================
# 2. 核心逻辑函数
# ==========================================

def get_pinyin_with_tone(char):
    """汉字转带调拼音，如 '丁' -> 'ding1'"""
    res = pinyin(char, style=Style.TONE3, strict=False)
    return res[0][0] if res else ""


def get_label_from_score(score_str):
    """
    根据第四行分数生成 0, 1, 2 的标签
    < 4: 0 | 4-7: 1 | >= 7: 2
    """
    try:
        score = float(score_str.strip())
        if score < 4:
            return 0
        elif 4 <= score < 7:
            return 1
        else:
            return 2
    except ValueError:
        return 1  # 默认值，防错处理


# ==========================================
# 3. 扫描与处理逻辑
# ==========================================

def process_split(split_name):
    split_dir = os.path.join(ROOT_DIR, split_name)
    if not os.path.exists(split_dir):
        print(f"跳过不存在的目录: {split_dir}")
        return

    split_dataset = []
    print(f"正在处理数据集划分: {split_name}...")

    # 递归遍历 (root 将会是 split/字/编号)
    for root, dirs, files in os.walk(split_dir):
        wav_files = [f for f in files if f.endswith('.wav')]

        for wav_name in wav_files:
            file_id = os.path.splitext(wav_name)[0]
            wav_path = os.path.join(root, wav_name)
            tg_path = os.path.join(root, f"{file_id}.TextGrid")

            if not os.path.exists(tg_path):
                continue

            try:
                # 解析 TextGrid
                tg = textgrid.TextGrid.fromFile(tg_path)

                # --- 核心提取 ---
                # 第一行 (Tier 1): 字
                ref_char = tg.tiers[0][0].mark.strip()
                target_pinyin = get_pinyin_with_tone(ref_char)

                # 第二行 (Tier 2): 实际拼音
                actual_pinyin = tg.tiers[1][0].mark.strip()

                # 第四行 (Tier 4): 分数
                raw_score = tg.tiers[3][0].mark.strip()  # Tier 索引从 0 开始，所以 4 是 3
                mdd_label = get_label_from_score(raw_score)

                # 记录每一个样本
                split_dataset.append({
                    "id": file_id,
                    "audio_path": wav_path,
                    "target_char": ref_char,
                    "target_pinyin": target_pinyin,
                    "actual_pinyin": actual_pinyin,
                    "score": float(raw_score),
                    "label": mdd_label
                })

            except Exception as e:
                print(f"处理文件 {file_id} 时出错: {e}")

    # 保存对应的 JSON
    out_file = f"metadata_{split_name}.json"
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(split_dataset, f, ensure_ascii=False, indent=4)
    print(f"完成！{split_name} 划分共收集 {len(split_dataset)} 条数据，保存至: {out_file}\n")


if __name__ == "__main__":
    for split in SPLITS:
        process_split(split)