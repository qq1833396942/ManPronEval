import torch
import torch.nn as nn
from transformers import WavLMModel
from peft import LoraConfig, get_peft_model


class MDD_WavLM_Model2(nn.Module):
    def __init__(self, num_pinyins, wavlm_path):
        super().__init__()
        print("🧬 Model 2 初始化中... [WavLM-Large + LoRA 版]")

        # 1. 加载 WavLM-Large
        self.wavlm = WavLMModel.from_pretrained(wavlm_path)

        # 2. 稳定性：禁用内部遮盖
        self.wavlm.config.mask_time_prob = 0.0
        self.wavlm.config.mask_feature_prob = 0.0

        # 3. 冻结前端 CNN
        if hasattr(self.wavlm, "feature_extractor"):
            self.wavlm.feature_extractor._freeze_parameters()

        # 4. 冻结整个 WavLM 主体
        for param in self.wavlm.parameters():
            param.requires_grad = False

        # 5. 挂 LoRA
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.1,
            bias="none"
        )
        self.wavlm = get_peft_model(self.wavlm, lora_config)
        self.wavlm.print_trainable_parameters()

        # large 通常 hidden_size = 1024
        audio_dim = self.wavlm.config.hidden_size
        hidden_dim = 256

        # 6. 局部增强模块
        self.local_conv = nn.Sequential(
            nn.Conv1d(audio_dim, audio_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm1d(audio_dim)
        )

        # 7. 拼音条件模块
        self.pinyin_embedding = nn.Embedding(num_pinyins + 1, hidden_dim)
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.text_proj = nn.Linear(hidden_dim, hidden_dim)

        self.cross_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads=8,
            batch_first=True
        )

        # 8. 分类头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 3)
        )

    def forward(self, waveforms, lengths, pinyin_ids):
        device = waveforms.device

        # A. 提取底座特征
        outputs = self.wavlm(input_values=waveforms)
        audio_seq = outputs.last_hidden_state

        # B. 安全 Mask
        _, max_frames, _ = audio_seq.shape
        feature_lengths = lengths // 320
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
            query=query,
            key=key,
            value=value,
            key_padding_mask=key_padding_mask
        )

        # E. 分类
        logits = self.classifier(attn_output.squeeze(1))
        return logits
