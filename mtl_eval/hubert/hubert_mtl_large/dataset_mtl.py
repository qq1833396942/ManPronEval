import json
import torch
import torchaudio
from torch.utils.data import Dataset

class MTLDataset(Dataset):
    def __init__(self, json_file, pinyin2id, target_sample_rate=16000, max_duration=3.0):
        with open(json_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.pinyin2id = pinyin2id
        self.target_sample_rate = target_sample_rate
        # 计算最大允许的采样点数 (16000 * 3 = 48000)
        self.max_length = int(target_sample_rate * max_duration)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # ==========================================
        # 1. 加载并处理音频 (定义 waveform)
        # ==========================================
        waveform, sr = torchaudio.load(item['audio_path'])
        
        # 如果是双声道，转为单声道
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # 采样率对齐
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sample_rate)
            waveform = resampler(waveform)
        
        # 去除多余的维度，变成 1D Tensor (length,)
        waveform = waveform.squeeze(0)
        
        # 截断过长的音频，防止 OOM
        if waveform.shape[0] > self.max_length:
            waveform = waveform[:self.max_length]
            
        # ==========================================
        # 2. 文本 ID 转换 (包含 <unk> 保护逻辑)
        # ==========================================
        # 从词表中获取 <unk> 的 ID (如果找不到默认用 3)
        unk_id = self.pinyin2id.get('<unk>', 3)
        
        # 安全获取 ID，遇到词表外的拼音直接标为 <unk>
        target_id = self.pinyin2id.get(item['target_pinyin'], unk_id) 
        actual_id = self.pinyin2id.get(item['actual_pinyin'], unk_id) 

        # ==========================================
        # 3. 标签提取
        # ==========================================
        mdd_label = item['mdd_label']
        scores = [item['score_total'], item['score_initial'], item['score_final']]

        return {
            "input_values": waveform,
            "target_pinyin_id": torch.tensor(target_id, dtype=torch.long),
            "actual_pinyin_id": torch.tensor(actual_id, dtype=torch.long),
            "mdd_label": torch.tensor(mdd_label, dtype=torch.long),
            "scores": torch.tensor(scores, dtype=torch.float)
        }

def mtl_collate_fn(batch):
    # 音频 padding (因为每个音频长度不同，需要用 0 补齐到 batch 内最长长度)
    waveforms = [item['input_values'] for item in batch]
    input_values = torch.nn.utils.rnn.pad_sequence(waveforms, batch_first=True, padding_value=0.0)
    
    # 堆叠其他定长标签
    target_pinyin_ids = torch.stack([item['target_pinyin_id'] for item in batch])
    actual_pinyin_ids = torch.stack([item['actual_pinyin_id'] for item in batch])
    mdd_labels = torch.stack([item['mdd_label'] for item in batch])
    scores = torch.stack([item['scores'] for item in batch])

    return {
        "input_values": input_values,
        "target_pinyin_ids": target_pinyin_ids,
        "actual_pinyin_ids": actual_pinyin_ids,
        "mdd_labels": mdd_labels,
        "scores": scores
    }