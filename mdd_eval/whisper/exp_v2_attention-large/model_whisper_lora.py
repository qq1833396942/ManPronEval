import torch
import torch.nn as nn
from transformers import WhisperModel, WhisperFeatureExtractor
from peft import LoraConfig, get_peft_model


class MDD_Whisper_Large_LoRA_Model(nn.Module):
    def __init__(self, num_pinyins, whisper_version):
        super().__init__()
        print(f"🧬 Model 3 (Large + LoRA) 初始化中... [类型对齐版]")

        # 1. 加载本地 Whisper-Large 核心
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(whisper_version)
        self.whisper = WhisperModel.from_pretrained(whisper_version)

        # 🚀 注入 LoRA 配置
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.1,
            bias="none"
        )
        # 为 Encoder 注入适配器
        self.encoder = get_peft_model(self.whisper.encoder, lora_config)
        self.encoder.print_trainable_parameters()

        audio_dim = self.whisper.config.d_model  # Large 为 1280
        hidden_dim = 256

        # 2. 局部卷积
        self.local_conv = nn.Sequential(
            nn.Conv1d(audio_dim, audio_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm1d(audio_dim)
        )

        # 3. 投影与注意力
        self.pinyin_embedding = nn.Embedding(num_pinyins + 1, hidden_dim)
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.text_proj = nn.Linear(hidden_dim, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)

        # 4. 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 3)
        )

    def forward(self, waveforms, lengths, pinyin_ids):
        device = waveforms.device

        # A. 预处理
        inputs = self.feature_extractor(waveforms.cpu().numpy(), sampling_rate=16000, return_tensors="pt")
        # 🛡️ 确保输入数据类型与 encoder 权重一致 (通常是 float16)
        input_features = inputs.input_features.to(device).to(self.encoder.dtype)

        # B. 提取特征
        encoder_outputs = self.encoder(input_features)
        audio_seq = encoder_outputs.last_hidden_state  # [B, 1500, 1280]

        # 🚀 【核心修复】将特征转回 float32，对接下游自定义层
        audio_seq = audio_seq.float()

        # C. 卷积局部增强 (现在 audio_seq 是 float32，local_conv 也是 float32，完美对齐)
        x = audio_seq.transpose(1, 2)
        x = self.local_conv(x)
        audio_seq = x.transpose(1, 2)

        # D. 计算 Whisper 专用动态掩码
        feature_lengths = lengths // 320
        max_len = 1500
        seq_range = torch.arange(max_len, device=device)
        key_padding_mask = seq_range.unsqueeze(0) >= feature_lengths.unsqueeze(1)

        # E. 交叉注意力与分类输出
        query = self.text_proj(self.pinyin_embedding(pinyin_ids).unsqueeze(1))
        key = value = self.audio_proj(audio_seq)

        attn_output, _ = self.cross_attn(
            query=query,
            key=key,
            value=value,
            key_padding_mask=key_padding_mask
        )

        # F. 最终分类
        return self.classifier(attn_output.squeeze(1))