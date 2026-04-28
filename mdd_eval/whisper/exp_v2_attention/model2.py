import torch
import torch.nn as nn
from transformers import WhisperModel, WhisperFeatureExtractor


class MDD_Whisper_Attention_Model(nn.Module):
    # 删除了 freeze_encoder 参数，大家一起放开微调 Transformer
    def __init__(self, num_pinyins, whisper_version="openai/whisper-base"):
        super().__init__()
        print(f"🧬 Model 2 (纯净版对齐) 初始化中... [Whisper 严格对照版]")

        # 1. 载入 Whisper 核心组件
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(whisper_version)
        self.whisper = WhisperModel.from_pretrained(whisper_version)
        self.encoder = self.whisper.encoder

        audio_dim = self.whisper.config.d_model  # base模型为 512
        hidden_dim = 256  # 统一投影空间维度

        # 2. 局部特征提取器 (1D 卷积)
        self.local_conv = nn.Sequential(
            nn.Conv1d(audio_dim, audio_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm1d(audio_dim)
        )

        # 3. 文本/音频投影层
        # 🚨 修复 2：Embedding 容量对齐 (num_pinyins + 1)
        self.pinyin_embedding = nn.Embedding(num_pinyins + 1, hidden_dim)
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.text_proj = nn.Linear(hidden_dim, hidden_dim)

        # 4. 交叉注意力层
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True
        )

        # 5. 分类决策头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            # 🚨 修复 3：Dropout 对齐至 0.2
            nn.Dropout(0.2),
            nn.Linear(128, 3)
        )

    def forward(self, waveforms, lengths, pinyin_ids):
        device = waveforms.device

        # A. Whisper 必须凑齐 30 秒
        inputs = self.feature_extractor(
            waveforms.cpu().numpy(),
            sampling_rate=16000,
            return_tensors="pt"
        )
        input_features = inputs.input_features.to(device)

        # B. 提取音频序列特征
        encoder_outputs = self.encoder(input_features)
        audio_seq = encoder_outputs.last_hidden_state  # 永远是 [B, 1500, 512]

        # C. 1D 卷积增强局部特征
        x = audio_seq.transpose(1, 2)
        x = self.local_conv(x)
        audio_seq = x.transpose(1, 2)

        # =======================================
        # 🚨 修复 1 核心：计算 Whisper 的精确 Mask
        # =======================================
        # Whisper 的下采样率和 Wav2Vec2/HuBERT 竟然奇迹般地一致，也是 320！
        feature_lengths = lengths // 320
        max_len = 1500  # Whisper 固定的最大帧数

        # 生成布尔掩码：超出真实音频长度的 30秒 Padding 区域全部遮蔽
        seq_range = torch.arange(max_len, device=device)
        key_padding_mask = seq_range.unsqueeze(0) >= feature_lengths.unsqueeze(1)

        # D. 交叉注意力
        query = self.text_proj(self.pinyin_embedding(pinyin_ids).unsqueeze(1))
        key = value = self.audio_proj(audio_seq)

        # 🚨 传入 Mask，让 Whisper 闭上眼睛不看那些垃圾静音帧
        attn_output, _ = self.cross_attn(
            query=query,
            key=key,
            value=value,
            key_padding_mask=key_padding_mask
        )

        # E. 分类预测
        logits = self.classifier(attn_output.squeeze(1))

        return logits