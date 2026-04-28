import json
import os

CODE_DIR = os.path.dirname(os.path.abspath(__file__))


def clean_mtl_json_file(file_path, output_path):
    if not os.path.exists(file_path):
        print(f"⚠️ 跳过不存在的文件: {file_path}")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    cleaned_data = []
    removed_count = 0

    stats = {
        "score_outlier": 0,
        "duration_short": 0,
        "invalid_pinyin": 0,
        "invalid_mdd_label": 0,
        "file_missing": 0
    }

    for item in data:
        s_tot = item.get('score_total', 0)
        s_ini = item.get('score_initial', 0)
        s_fin = item.get('score_final', 0)
        if not (0 <= s_tot <= 10 and 0 <= s_ini <= 10 and 0 <= s_fin <= 10):
            removed_count += 1
            stats["score_outlier"] += 1
            continue

        duration = item.get('end_time', 0) - item.get('start_time', 0)
        if duration < 0.03:
            removed_count += 1
            stats["duration_short"] += 1
            continue

        target_py = item.get('target_pinyin', '').lower()
        actual_py = item.get('actual_pinyin', '').lower()
        invalid_labels = ['sil', 'sp', 'none', '']
        if target_py in invalid_labels or actual_py in invalid_labels:
            removed_count += 1
            stats["invalid_pinyin"] += 1
            continue

        mdd_label = item.get('mdd_label', -1)
        if mdd_label not in [0, 1, 2]:
            removed_count += 1
            stats["invalid_mdd_label"] += 1
            continue

        audio_path = item.get('audio_path', '')
        if not os.path.exists(audio_path):
            removed_count += 1
            stats["file_missing"] += 1
            continue

        cleaned_data.append(item)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=4)

    print(f"\n✅ 文件 {file_path} 清洗完成:")
    print(f"   - 保留样本: {len(cleaned_data)}")
    print(f"   - 总剔除异常数: {removed_count}")
    if removed_count > 0:
        print(f"     * 分数异常: {stats['score_outlier']}")
        print(f"     * 时长过短: {stats['duration_short']}")
        print(f"     * 拼音无效(sil/sp): {stats['invalid_pinyin']}")
        print(f"     * MDD标签越界: {stats['invalid_mdd_label']}")
        print(f"     * 实体音频丢失: {stats['file_missing']}")


if __name__ == "__main__":
    files = [
    (os.path.join(CODE_DIR, 'metadata_train_mtl.json'), os.path.join(CODE_DIR, 'metadata_train_mtl.json')),
    (os.path.join(CODE_DIR, 'metadata_val_mtl.json'), os.path.join(CODE_DIR, 'metadata_val_mtl.json')),
    (os.path.join(CODE_DIR, 'metadata_test_mtl.json'), os.path.join(CODE_DIR, 'metadata_test_mtl.json')),
]
    for in_f, out_f in files:
        clean_mtl_json_file(in_f, out_f)
