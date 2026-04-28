import os
import json
import tgt
from tqdm import tqdm
from pypinyin import pinyin, Style

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = r""
SPLITS = ['train', 'test', 'val']


def get_pinyin_with_tone(char):
    res = pinyin(char, style=Style.TONE3, strict=False)
    return res[0][0] if res else ""


def get_mdd_label(score):
    if score < 4.0:
        return 0
    elif 4.0 <= score < 7.0:
        return 1
    else:
        return 2


def load_tg_safe(path):
    for enc in ['utf-8', 'utf-16', 'gbk']:
        try:
            return tgt.io.read_textgrid(path, encoding=enc)
        except Exception:
            continue
    return None


def process_split(split_name):
    split_dir = os.path.join(ROOT_DIR, split_name)
    if not os.path.exists(split_dir):
        print(f"⚠️ 找不到目录: {split_dir}")
        return []

    dataset = []
    print(f"\n🔍 正在扫描并解析 {split_name} 数据集 (WavLM MTL 版本)...")

    hanzi_folders = [d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))]

    for hanzi in tqdm(hanzi_folders, desc=f"Processing {split_name}"):
        hanzi_path = os.path.join(split_dir, hanzi)
        for score_folder in ['0', '1', '2']:
            score_path = os.path.join(hanzi_path, score_folder)
            if not os.path.exists(score_path):
                continue

            for file in os.listdir(score_path):
                if not file.endswith('.wav'):
                    continue

                file_id = file.replace('.wav', '')
                wav_path = os.path.join(score_path, file)
                tg_path = os.path.join(score_path, f"{file_id}.TextGrid")

                if not os.path.exists(tg_path):
                    continue

                tg = load_tg_safe(tg_path)
                if tg is None:
                    continue

                try:
                    target_char = tg.tiers[0].intervals[0].text.strip()
                    target_pinyin = get_pinyin_with_tone(target_char)
                    actual_pinyin = tg.tiers[1].intervals[0].text.strip()

                    start_time = tg.start_time
                    end_time = tg.end_time

                    score_total = float(tg.get_tier_by_name('SentenceScore').intervals[0].text)
                    score_initial = float(tg.get_tier_by_name('Subword1Score').intervals[0].text)
                    score_final = float(tg.get_tier_by_name('Subword2Score').intervals[0].text)

                    mdd_label = get_mdd_label(score_total)

                    dataset.append({
                        "id": file_id,
                        "audio_path": wav_path,
                        "target_char": target_char,
                        "target_pinyin": target_pinyin,
                        "actual_pinyin": actual_pinyin,
                        "score_total": score_total,
                        "score_initial": score_initial,
                        "score_final": score_final,
                        "mdd_label": mdd_label,
                        "start_time": start_time,
                        "end_time": end_time,
                    })
                except Exception:
                    continue

    return dataset


if __name__ == "__main__":
    for split in SPLITS:
        data = process_split(split)
        if len(data) > 0:
            out_file = os.path.join(CODE_DIR, f"metadata_{split}_mtl.json")
            with open(out_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            print(f"✅ 成功生成 {out_file}，共包含 {len(data)} 条多任务数据！")
