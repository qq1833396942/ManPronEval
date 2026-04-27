import json
import os
import torchaudio
from tqdm import tqdm


def clean_mdd_dataset(json_path, output_path):
    print(f"🔍 正在清洗: {json_path}")

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    cleaned_data = []
    bad_count = 0

    # 使用 tqdm 显示进度
    for item in tqdm(data, desc="检查音频中"):
        audio_path = item['audio_path']

        # 检查逻辑开始
        is_good = True
        try:
            # 1. 检查物理文件是否存在
            if not os.path.exists(audio_path):
                is_good = False
                reason = "文件不存在"
            else:
                # 2. 尝试用 torchaudio 解码
                # frame_offset 和 num_frames 设为小数值，只读个头，速度极快
                waveform, sr = torchaudio.load(audio_path, frame_offset=0, num_frames=160)

                # 3. 检查是否为空文件
                if waveform.shape[1] == 0:
                    is_good = False
                    reason = "空音频文件"
        except Exception as e:
            is_good = False
            reason = str(e)

        # 结果处理
        if is_good:
            cleaned_data.append(item)
        else:
            bad_count += 1
            print(f"\n🗑️ 剔除坏数据: {audio_path} | 原因: {reason}")

    # 保存清洗后的 JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=4)

    print(f"\n✅ 清洗完成！")
    print(f"📊 原始数据: {len(data)} 条")
    print(f"📊 剔除数据: {bad_count} 条")
    print(f"📊 剩余有效数据: {len(cleaned_data)} 条")
    print(f"💾 新 JSON 已保存至: {output_path}")


if __name__ == "__main__":
    # 针对你的三个文件分别清洗
    files_to_clean = [
        ('metadata_train.json', 'metadata_train_cleaned.json'),
        ('metadata_val.json', 'metadata_val_cleaned.json'),
        ('metadata_test.json', 'metadata_test_cleaned.json')
    ]

    for input_f, output_f in files_to_clean:
        if os.path.exists(input_f):
            clean_mdd_dataset(input_f, output_f)