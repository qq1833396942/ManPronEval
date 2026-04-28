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
        self.max_length = int(target_sample_rate * max_duration)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        waveform, sr = torchaudio.load(item['audio_path'])

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sample_rate)
            waveform = resampler(waveform)

        waveform = waveform.squeeze(0)

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
            "scores": torch.tensor(scores, dtype=torch.float),
        }


def mtl_collate_fn(batch):
    waveforms = [item['input_values'] for item in batch]
    input_values = torch.nn.utils.rnn.pad_sequence(waveforms, batch_first=True, padding_value=0.0)

    target_pinyin_ids = torch.stack([item['target_pinyin_id'] for item in batch])
    actual_pinyin_ids = torch.stack([item['actual_pinyin_id'] for item in batch])
    mdd_labels = torch.stack([item['mdd_label'] for item in batch])
    scores = torch.stack([item['scores'] for item in batch])

    return {
        "input_values": input_values,
        "target_pinyin_ids": target_pinyin_ids,
        "actual_pinyin_ids": actual_pinyin_ids,
        "mdd_labels": mdd_labels,
        "scores": scores,
    }
