import os
import json
import torch
import soundfile as sf
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

def build_pinyin_vocab(json_paths):
    unique_pinyins = set()
    for path in json_paths:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    unique_pinyins.add(item.get('target_pinyin', '[UNK]'))

    pinyin2id = {"[PAD]": 0, "[UNK]": 1}
    for i, p in enumerate(sorted(list(unique_pinyins))):
        pinyin2id[p] = i + 2
    return pinyin2id

class APA_Wav2Vec2_Dataset(Dataset):
    def __init__(self, json_path, pinyin2id, target_sr=16000):
        super().__init__()
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.pinyin2id = pinyin2id
        self.target_sr = target_sr

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 1. 读音频
        speech, sr = sf.read(item['audio_path'])
        if sr != self.target_sr:
            import resampy
            speech = resampy.resample(speech, sr, self.target_sr)
        if len(speech.shape) > 1: speech = speech[:, 0]
        
        audio_length_sec = len(speech) / self.target_sr

        # 2. 提取并扩展时间边界 (+50ms 上下文)
        raw_start = item.get('start_time', 0.0)
        raw_end = item.get('end_time', audio_length_sec)
        start_time = max(0.0, raw_start - 0.05)
        end_time = min(audio_length_sec, raw_end + 0.05)

        start_frame = int(start_time * self.target_sr / 320)
        end_frame = int(end_time * self.target_sr / 320)
        if end_frame <= start_frame: end_frame = start_frame + 1

        # 3. 提取分数 (多任务回归目标)
        # 将原始分数除以 10.0，使目标值落在 [0, 1] 区间
        score_initial = item.get('score_initial', 0.0) / 10.0
        score_final = item.get('score_final', 0.0) / 10.0
        score_total = item.get('score_total', 0.0) / 10.0
        
        pinyin_id = self.pinyin2id.get(item.get('target_pinyin', '[UNK]'), 1)

        return {
            "waveform": torch.tensor(speech, dtype=torch.float32),
            "length": torch.tensor(len(speech), dtype=torch.long),
            "start_frame": torch.tensor(start_frame, dtype=torch.long),
            "end_frame": torch.tensor(end_frame, dtype=torch.long),
            "pinyin_id": torch.tensor(pinyin_id, dtype=torch.long),
            "score_initial": torch.tensor([score_initial], dtype=torch.float32),
            "score_final": torch.tensor([score_final], dtype=torch.float32),
            "score_total": torch.tensor([score_total], dtype=torch.float32)
        }

def collate_fn_apa(batch):
    waveforms = [item['waveform'] for item in batch]
    padded_waveforms = pad_sequence(waveforms, batch_first=True, padding_value=0.0)
    
    return {
        "waveforms": padded_waveforms,
        "lengths": torch.stack([item['length'] for item in batch]),
        "start_frames": torch.stack([item['start_frame'] for item in batch]),
        "end_frames": torch.stack([item['end_frame'] for item in batch]),
        "pinyin_ids": torch.stack([item['pinyin_id'] for item in batch]),
        "score_initial": torch.stack([item['score_initial'] for item in batch]),
        "score_final": torch.stack([item['score_final'] for item in batch]),
        "score_total": torch.stack([item['score_total'] for item in batch])
    }