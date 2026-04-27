import os
import json
import torch
import soundfile as sf
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from scipy.signal import resample_poly
class MTLDataset(Dataset):
    def __init__(self, json_file, pinyin2id, target_sample_rate=16000, max_duration=3.0):
        with open(json_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.pinyin2id = pinyin2id
        self.target_sample_rate = target_sample_rate
        self.max_length = int(target_sample_rate * max_duration)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        waveform, sr = sf.read(item['audio_path'], dtype='float32')

        # 多声道转单声道
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)

        # 重采样
        if sr != self.target_sample_rate:
            waveform = resample_poly(waveform, self.target_sample_rate, sr).astype('float32')

        waveform = torch.from_numpy(waveform).float()

        if waveform.shape[0] > self.max_length:
            waveform = waveform[:self.max_length]

        unk_id = self.pinyin2id.get('<unk>', 3)
        target_id = self.pinyin2id.get(item['target_pinyin'], unk_id)
        actual_id = self.pinyin2id.get(item['actual_pinyin'], unk_id)

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
    # 取出每条音频
    waveforms = [item['input_values'] for item in batch]

    # 每条原始长度
    lengths = torch.tensor([w.shape[0] for w in waveforms], dtype=torch.long)

    # padding 到 batch 内最长长度
    input_values = torch.nn.utils.rnn.pad_sequence(
        waveforms,
        batch_first=True,
        padding_value=0.0
    )  # [B, T]

    # 生成 attention_mask: [B, T]
    attention_mask = torch.arange(input_values.size(1)).unsqueeze(0) < lengths.unsqueeze(1)
    attention_mask = attention_mask.long()

    # 其他标签正常堆叠
    target_pinyin_ids = torch.stack([item['target_pinyin_id'] for item in batch])
    actual_pinyin_ids = torch.stack([item['actual_pinyin_id'] for item in batch])
    mdd_labels = torch.stack([item['mdd_label'] for item in batch])
    scores = torch.stack([item['scores'] for item in batch])

    return {
        "input_values": input_values,
        "attention_mask": attention_mask,
        "input_lengths": lengths,   # 可选，但建议保留
        "target_pinyin_ids": target_pinyin_ids,
        "actual_pinyin_ids": actual_pinyin_ids,
        "mdd_labels": mdd_labels,
        "scores": scores
    }