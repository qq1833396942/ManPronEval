import os
import json
import torch
import torchaudio
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
    pinyin2id = {"[UNK]": 0}
    for i, p in enumerate(sorted(list(unique_pinyins))):
        pinyin2id[p] = i + 1
    return pinyin2id

class MDD_WavLM_Dataset(Dataset):
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
        waveform, sr = torchaudio.load(item['audio_path'])
        
        # 转单声道
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
            
        # 重采样
        if sr != self.target_sr:
            if sr not in self.resamplers:
                self.resamplers[sr] = torchaudio.transforms.Resample(sr, self.target_sr)
            waveform = self.resamplers[sr](waveform)
            
        waveform = waveform.squeeze(0)
        pinyin = item.get('target_pinyin', '[UNK]')
        pinyin_id = self.pinyin2id.get(pinyin, self.pinyin2id["[UNK]"])

        return {
            "waveform": waveform,
            "length": torch.tensor(len(waveform), dtype=torch.long),
            "pinyin_id": torch.tensor(pinyin_id, dtype=torch.long),
            "label": torch.tensor(item['label'], dtype=torch.long)
        }

def collate_fn_wavlm(batch):
    waveforms = [item['waveform'] for item in batch]
    padded_waveforms = pad_sequence(waveforms, batch_first=True, padding_value=0.0)
    lengths = torch.stack([item['length'] for item in batch])
    pinyin_ids = torch.stack([item['pinyin_id'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    return {
        "waveforms": padded_waveforms,
        "lengths": lengths,
        "pinyin_ids": pinyin_ids,
        "labels": labels
    }