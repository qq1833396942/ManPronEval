import os
import json
import torch
import torchaudio
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

# ==========================================
# 1. 拼音词表构建器
# ==========================================
def build_pinyin_vocab(json_paths):
    unique_pinyins = set()
    for path in json_paths:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    unique_pinyins.add(item.get('target_pinyin', '[UNK]'))

    pinyin2id = {"[UNK]": 0}
    for i, p in enumerate(sorted(list(unique_pinyins))):
        pinyin2id[p] = i + 1
    return pinyin2id

# ==========================================
# 2. HuBERT 专属 Dataset
# ==========================================
class MDD_HuBERT_Dataset(Dataset):
    def __init__(self, json_path, pinyin2id, target_sr=16000):
        super().__init__()
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        self.pinyin2id = pinyin2id
        self.target_sr = target_sr
        self.resamplers = {}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        audio_path = item['audio_path']
        pinyin = item.get('target_pinyin', '[UNK]')
        label = item['label']

        # 1. 读取音频并重采样
        waveform, sr = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
            
        if sr != self.target_sr:
            if sr not in self.resamplers:
                self.resamplers[sr] = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr)
            waveform = self.resamplers[sr](waveform)

        waveform = waveform.squeeze(0) # 变成 1D 张量 [Time]
        
        # 🚨 核心修改：已删除所有计算 start_frame 和 end_frame 的“开卷”逻辑，直接把整段音频交给模型自己找！

        # 2. 获取拼音 ID
        pinyin_id = self.pinyin2id.get(pinyin, self.pinyin2id["[UNK]"])

        return {
            "id": item.get("id", str(idx)),
            "waveform": waveform,
            "pinyin_id": torch.tensor(pinyin_id, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long)
        }

# ==========================================
# 3. 数据打包器 (Collate Function)
# ==========================================
def collate_fn_hubert(batch):
    ids = [item['id'] for item in batch]
    waveforms = [item['waveform'] for item in batch]
    
    # 🚨 新增：记录被 Padding 之前的原始长度（采样点数），传给模型算 Mask
    lengths = torch.tensor([len(w) for w in waveforms], dtype=torch.long)
    
    # HuBERT 动态补齐
    padded_waveforms = pad_sequence(waveforms, batch_first=True, padding_value=0.0)
    
    pinyin_ids = torch.stack([item['pinyin_id'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])

    # 🚨 核心修改：返回的字典移除了 start_frames 和 end_frames，加入了 lengths
    return {
        "ids": ids,
        "waveforms": padded_waveforms,
        "lengths": lengths,      # 加入 lengths
        "pinyin_ids": pinyin_ids,
        "labels": labels
    }