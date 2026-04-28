import os
import json
import tgt
from tqdm import tqdm
from pypinyin import pinyin, Style
# ==========================================
# 配置区 (已更新为你的实际路径)
# ==========================================
ROOT_DIR = r'C:\Users\14183\OneDrive\Desktop\belle\solo_data'  # 根目录下包含 train/test/val
SPLITS = ['train', 'test', 'val']

def load_tg_safe(path):
    """安全读取 TextGrid，兼容不同的编码格式"""
    for enc in ['utf-8', 'utf-16', 'gbk']:
        try:
            return tgt.io.read_textgrid(path, encoding=enc)
        except:
            continue
    return None

def process_split(split_name):
    split_dir = os.path.join(ROOT_DIR, split_name)
    if not os.path.exists(split_dir):
        print(f"⚠️ 找不到目录: {split_dir}")
        return []
        
    dataset = []
    print(f"\n🔍 正在扫描并解析 {split_name} 数据集...")
    
    # 提取所有汉字文件夹
    hanzi_folders = [d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))]
    
    # 遍历：split -> 汉字 -> 0/1/2 -> .wav & .TextGrid
    for hanzi in tqdm(hanzi_folders, desc=f"处理 {split_name} 进度"):
        hanzi_path = os.path.join(split_dir, hanzi)
        
        for score_folder in ['0', '1', '2']:
            folder_path = os.path.join(hanzi_path, score_folder)
            if not os.path.exists(folder_path): continue
            
            wav_files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
            
            for wav_file in wav_files:
                wav_path = os.path.join(folder_path, wav_file)
                tg_path = wav_path.replace('.wav', '.TextGrid')
                
                if not os.path.exists(tg_path):
                    continue
                    
                # 解析 TextGrid
                tg = load_tg_safe(tg_path)
                if tg is None:
                    continue
                
                try:
                    # === 提取汉字并转换为拼音 ===
                    words_tier = tg.get_tier_by_name('words')
                    
                    # 1. 提取真正的汉字文本（过滤掉首尾可能的空白静音段）
                    hanzi = ""
                    for interval in words_tier.intervals:
                        if interval.text.strip() != "":
                            hanzi = interval.text.strip()
                            break
                            
                    # 2. 将汉字转为带数字的拼音 (例如: "哀" -> "ai1", "衰" -> "shuai1")
                    if hanzi:
                        # Style.TONE3 会在拼音末尾加上 1-4 的声调数字
                        extracted_pinyin = pinyin(hanzi, style=Style.TONE3, heteronym=False)[0][0]
                    else:
                        extracted_pinyin = "unknown"
                    
                    # 3. 获取音节时间边界
                    syl_tier = tg.get_tier_by_name('syllables')
                    start_time = syl_tier.intervals[0].start_time
                    end_time = syl_tier.intervals[0].end_time
                    
                    # 4. 获取多维度专家打分 (转为 float)
                    score_total = float(tg.get_tier_by_name('SentenceScore').intervals[0].text)
                    score_initial = float(tg.get_tier_by_name('Subword1Score').intervals[0].text)
                    score_final = float(tg.get_tier_by_name('Subword2Score').intervals[0].text)
                    
                    # 构建 APA 专属字典
                    item_data = {
                        "audio_path": wav_path,
                        "target_pinyin": extracted_pinyin,
                        "start_time": start_time,
                        "end_time": end_time,
                        "score_total": score_total,
                        "score_initial": score_initial,
                        "score_final": score_final,
                        "original_level": int(score_folder)
                    }
                    dataset.append(item_data)
                    
                except Exception as e:
                    # 有些极端的坏数据可能没有这些层，直接跳过
                    continue
                    
    return dataset

if __name__ == "__main__":
    for split in SPLITS:
        data = process_split(split)
        if len(data) > 0:
            out_file = f"metadata_{split}_apa.json"
            with open(out_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            print(f"✅ {split} 阶段处理完成，共保存 {len(data)} 条 APA 样本至 {out_file}")