import torch
import torch.nn as nn
from transformers import WavLMModel

class MDD_WavLM_Model2(nn.Module):
    def __init__(self, num_pinyins, wavlm_path):
        super().__init__()
        print("🧬 Model 2 初始化中... [WavLM 绝对公平版]")
        self.wavlm = WavLMModel.from_pretrained(wavlm_path)
        
        # 🛡️ 稳定性：禁用内部掩码概率
        self.wavlm.config.mask_time_prob = 0.0
        self.wavlm.config.mask_feature_prob = 0.0

        # 🚨 公平性：冻结 CNN 提取器
        if hasattr(self.wavlm, "feature_extractor"):
            self.wavlm.feature_extractor._freeze_parameters()

        audio_dim = self.wavlm.config.hidden_size # 768
        hidden_dim = 256

        self.local_conv = nn.Sequential(
            nn.Conv1d(audio_dim, audio_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm1d(audio_dim)
        )

        self.pinyin_embedding = nn.Embedding(num_pinyins + 1, hidden_dim)
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.text_proj = nn.Linear(hidden_dim, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 3)
        )

    def forward(self, waveforms, lengths, pinyin_ids):
        device = waveforms.device
        
        # A. 提取底座特征 (不传 attention_mask 保持绝对公平)
        outputs = self.wavlm(input_values=waveforms)
        audio_seq = outputs.last_hidden_state 

        # B. 🛡️ 修复 NaN 的核心：安全 Mask 计算
        batch_size, max_frames, _ = audio_seq.shape
        feature_lengths = lengths // 320 
        # 强制有效帧数至少为 1，防止全屏蔽
        feature_lengths = torch.clamp(feature_lengths, min=1)

        seq_range = torch.arange(max_frames, device=device)
        key_padding_mask = seq_range.unsqueeze(0) >= feature_lengths.unsqueeze(1)

        # C. 局部增强
        x = audio_seq.transpose(1, 2)
        x = self.local_conv(x)
        audio_seq = x.transpose(1, 2)

        # D. 交叉注意力
        query = self.text_proj(self.pinyin_embedding(pinyin_ids).unsqueeze(1))
        key = value = self.audio_proj(audio_seq)

        attn_output, _ = self.cross_attn(
            query=query, key=key, value=value, 
            key_padding_mask=key_padding_mask
        )

        logits = self.classifier(attn_output.squeeze(1))
        return logits