import json
import os

def clean_json_file(file_path, output_path):
    if not os.path.exists(file_path):
        print(f"跳过不存在的文件: {file_path}")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    cleaned_data = []
    removed_count = 0

    for item in data:
        # 1. 检查分数是否在合理的 0-10 范围内
        s_tot = item.get('score_total', 0)
        s_ini = item.get('score_initial', 0)
        s_fin = item.get('score_final', 0)
        
        # 👑 核心逻辑：剔除像 4185.0 这种离群值，以及负数
        if not (0 <= s_tot <= 10 and 0 <= s_ini <= 10 and 0 <= s_fin <= 10):
            removed_count += 1
            continue

        # 2. 检查音频时长是否过短 (至少需要 30ms 才能提取特征)
        duration = item.get('end_time', 0) - item.get('start_time', 0)
        if duration < 0.03:
            removed_count += 1
            continue

        # 3. 剔除非语音标注 (如 sil, sp)
        if item.get('target_pinyin', '').lower() in ['sil', 'sp', 'none', '']:
            removed_count += 1
            continue

        cleaned_data.append(item)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=4)

    print(f"✅ 文件 {file_path} 清洗完成:")
    print(f"   - 保留样本: {len(cleaned_data)}")
    print(f"   - 剔除异常: {removed_count}")

if __name__ == "__main__":
    files = [
        ('metadata_train_apa.json', 'metadata_train_apa.json'),
        ('metadata_val_apa.json', 'metadata_val_apa.json'),
        ('metadata_test_apa.json', 'metadata_test_apa.json')
    ]
    
    for in_f, out_f in files:
        clean_json_file(in_f, out_f)