import os
import json
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


# ==========================================
# 1. 拼音词表构建器
# ==========================================
def build_pinyin_vocab(json_paths):
    """
    遍历所有的 JSON 文件，找出所有独一无二的拼音，
    给每个拼音分配一个固定的数字 ID (比如 ding1 -> 15)
    """
    unique_pinyins = set()
    for path in json_paths:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    unique_pinyins.add(item['target_pinyin'])

    # 加上一个特殊的 [UNK] (未知) 占位符，以防万一
    pinyin2id = {"[UNK]": 0}
    for i, p in enumerate(sorted(list(unique_pinyins))):
        pinyin2id[p] = i + 1
    return pinyin2id


# ==========================================
# 2. PyTorch Dataset 核心类
# ==========================================
class MDDDataset(Dataset):
    def __init__(self, json_path, pinyin2id, target_sr=16000):
        super().__init__()
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        self.pinyin2id = pinyin2id
        self.target_sr = target_sr
        self.resamplers = {}  # 缓存重采样器，加速读取

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        audio_path = item['audio_path']
        pinyin = item['target_pinyin']
        label = item['label']  # 0, 1, 2

        # 1. 读取音频
        waveform, sr = torchaudio.load(audio_path)

        # 2. 统一转换为单声道 (Mono)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # 3. 重采样到 16kHz (Whisper/WavLM 的硬性要求)
        if sr != self.target_sr:
            if sr not in self.resamplers:
                self.resamplers[sr] = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr)
            waveform = self.resamplers[sr](waveform)

        # 去掉多余的 channel 维度，变成 1D 张量 [Time]
        waveform = waveform.squeeze(0)

        # 4. 获取拼音的数字 ID
        pinyin_id = self.pinyin2id.get(pinyin, self.pinyin2id["[UNK]"])

        return {
            "id": item["id"],
            "waveform": waveform,
            "pinyin_id": torch.tensor(pinyin_id, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long)
        }


# ==========================================
# 3. 数据打包器 (Collate Function)
# ==========================================
def collate_fn(batch):
    """
    因为每个录音的长度不一样，这个函数会在组装 Batch 时，
    自动把短的录音用 0 补齐 (Padding)，让它们变成一个规整的矩阵。
    """
    ids = [item['id'] for item in batch]
    pinyin_ids = torch.stack([item['pinyin_id'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])

    # 提取所有的 waveform
    waveforms = [item['waveform'] for item in batch]

    # 记录每个音频原本的真实长度（后续模型提特征时需要，防止把补的 0 当成声音算进去）
    lengths = torch.tensor([len(w) for w in waveforms], dtype=torch.long)

    # 核心：自动补齐 (Padding)
    # batch_first=True 表示输出形状为 [Batch_Size, Max_Time]
    padded_waveforms = pad_sequence(waveforms, batch_first=True, padding_value=0.0)

    return {
        "ids": ids,
        "waveforms": padded_waveforms,
        "lengths": lengths,
        "pinyin_ids": pinyin_ids,
        "labels": labels
    }