import json
import torch
import torchaudio
from pathlib import Path, PurePosixPath, PureWindowsPath
from torch.utils.data import Dataset
from transformers import WhisperFeatureExtractor


class MTLDataset(Dataset):
    def __init__(
        self,
        json_file,
        pinyin2id,
        target_sample_rate=16000,
        audio_root=None,
        max_audio_seconds=5,
    ):
        with open(json_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.pinyin2id = pinyin2id
        self.unk_id = pinyin2id.get('<unk>')
        self.target_sample_rate = target_sample_rate
        self.audio_root = Path(audio_root).expanduser() if audio_root else None
        self.max_audio_len = int(max_audio_seconds * target_sample_rate)
        # 🌟 优化：预先定义 Resampler 提高效率
        self.resamplers = {}

    def __len__(self):
        return len(self.data)

    def _resolve_audio_path(self, raw_path):
        path = Path(raw_path).expanduser()
        if path.exists():
            return str(path)

        if not self.audio_root:
            return raw_path

        pure_path = PureWindowsPath(raw_path) if "\\" in raw_path else PurePosixPath(raw_path)
        parts = pure_path.parts
        for split_name in ("train", "val", "test"):
            if split_name in parts:
                split_idx = parts.index(split_name)
                candidate = self.audio_root.joinpath(*parts[split_idx:])
                if candidate.exists():
                    return str(candidate)

        candidate = self.audio_root / pure_path.name
        return str(candidate) if candidate.exists() else raw_path

    def _pinyin_id(self, item, key):
        pinyin = item.get(key)
        if pinyin in self.pinyin2id:
            return self.pinyin2id[pinyin]
        if self.unk_id is None:
            raise KeyError(f"词表缺少拼音 {pinyin!r}，且没有 <unk> 兜底。")
        return self.unk_id

    def __getitem__(self, idx):
        item = self.data[idx]
        audio_path = self._resolve_audio_path(item['audio_path'])
        try:
            waveform, sr = torchaudio.load(audio_path)

            # 1. 转单声道
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # 2. 重采样优化
            if sr != self.target_sample_rate:
                if sr not in self.resamplers:
                    self.resamplers[sr] = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sample_rate)
                waveform = self.resamplers[sr](waveform)

            # 🌟 3. 长度限制：单音节任务不需要30秒这么长，这里强制截断超过 5 秒的音频（可选）
            # 防止脏数据中有超长音频导致后续计算溢出
            if waveform.shape[1] > self.max_audio_len:
                waveform = waveform[:, :self.max_audio_len]

            waveform = waveform.squeeze(0).numpy()
        except Exception as e:
            print(f"\n⚠️ 音频加载失败，使用静音兜底: {audio_path} | Error: {e}")
            waveform = torch.zeros(self.target_sample_rate).numpy()

        target_id = self._pinyin_id(item, 'target_pinyin')
        actual_id = self._pinyin_id(item, 'actual_pinyin')

        return {
            "waveform": waveform,
            "target_pinyin_id": torch.tensor(target_id, dtype=torch.long),
            "actual_pinyin_id": torch.tensor(actual_id, dtype=torch.long),
            "mdd_label": torch.tensor(item['mdd_label'], dtype=torch.long),
            "scores": torch.tensor([item['score_total'], item['score_initial'], item['score_final']], dtype=torch.float)
        }


class MTLCollateFn:
    def __init__(self, model_path):
        # 它会自动处理 80 还是 128 通道
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(model_path)

    def __call__(self, batch):
        waveforms = [item['waveform'] for item in batch]

        # Whisper 核心处理
        # return_tensors="pt" 已经帮我们转成了张量
        inputs = self.feature_extractor(
            waveforms,
            sampling_rate=16000,
            return_tensors="pt",
            padding="max_length",  # 强制填充到 3000 帧（即30秒）
            truncation=True,
        )

        return {
            "input_features": inputs.input_features,  # 张量形状: [batch, 80/128, 3000]
            "target_pinyin_ids": torch.stack([item['target_pinyin_id'] for item in batch]),
            "actual_pinyin_ids": torch.stack([item['actual_pinyin_id'] for item in batch]),
            "mdd_labels": torch.stack([item['mdd_label'] for item in batch]),
            "scores": torch.stack([item['scores'] for item in batch])
        }
