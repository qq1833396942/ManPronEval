import torch
import torch.nn as nn
from transformers import HubertModel

class MDD_HuBERT_Attention_Model(nn.Module):
    def __init__(self, num_pinyins, freeze_encoder=False, hubert_version="facebook/hubert-base-ls960"):
        super().__init__()
        print(f"🧬 Model 2 (Local + Attention) 初始化中... [HuBERT 严格对照版]")

        # 1. 载入 HuBERT 核心组件
        self.hubert = HubertModel.from_pretrained(hubert_version)
        
        # 👑 极其关键：禁用遮盖配置，防止短音频报错
        self.hubert.config.mask_time_prob = 0.0
        self.hubert.config.mask_feature_prob = 0.0

        # 🚨 内鬼 2 已清除：只冻结前端 7 层 CNN 特征提取器，保护基础声学特征
        self.hubert.feature_extractor._freeze_parameters()
        
        # （可选）如果想和 W2V 一样整体冻结 Transformer 编码器，可以通过传参控制
        if freeze_encoder:
            print("❄️ HuBERT Transformer 骨干网络已冻结")
            for param in self.hubert.encoder.parameters():
                param.requires_grad = False

        # 获取音频特征维度 (HuBERT base 为 768)
        audio_dim = self.hubert.config.hidden_size  
        hidden_dim = 256  # 统一注意力投影空间维度

        # 2. 局部特征提取器 (1D 卷积，kernel_size=3 捕捉瞬时发音变化)
        self.local_conv = nn.Sequential(
            nn.Conv1d(audio_dim, audio_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm1d(audio_dim)
        )

        # 3. 文本/音频 投影层 
        # 🚨 内鬼 4 已清除：Embedding 维度增加余量 (num_pinyins + 1)
        self.pinyin_embedding = nn.Embedding(num_pinyins + 1, hidden_dim)
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.text_proj = nn.Linear(hidden_dim, hidden_dim) # 配合 embedding 维度修改为 hidden_dim

        # 4. 交叉注意力层 (Cross-Attention)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True
        )

        # 5. 分类决策头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            # 🚨 内鬼 3 已清除：Dropout 严格对齐至 0.2
            nn.Dropout(0.2), 
            nn.Linear(128, 3) # 输出 0, 1, 2
        )

    # 🚨 内鬼 1 已清除：接收 lengths 并计算 Mask
    def forward(self, input_values, lengths, target_syllable_ids):
        # =======================================
        # A. 提取音频序列特征
        # =======================================
        outputs = self.hubert(input_values)
        audio_seq = outputs.last_hidden_state  # 形状: [Batch, Time_Frames, 768]

        # =======================================
        # B. 🚨 内鬼 1 核心修复：计算 Attention Mask
        # =======================================
        batch_size, max_frames, _ = audio_seq.shape
        # HuBERT 的 CNN 下采样率也是 320，和 Wav2Vec2 完全一致
        feature_lengths = lengths // 320 
        
        # 生成布尔掩码：超出真实长度的部分标记为 True (表示需要被 Attention 忽略的 Padding)
        seq_range = torch.arange(max_frames, device=audio_seq.device)
        key_padding_mask = seq_range.unsqueeze(0) >= feature_lengths.unsqueeze(1)

        # =======================================
        # C. 1D 卷积增强局部特征
        # =======================================
        x = audio_seq.transpose(1, 2)
        x = self.local_conv(x)
        audio_seq = x.transpose(1, 2)  # 换回 [Batch, Time_Frames, 768]

        # =======================================
        # D. 交叉注意力：带 Mask 的安全对齐
        # =======================================
        text_vec = self.pinyin_embedding(target_syllable_ids).unsqueeze(1)
        query = self.text_proj(text_vec)           # [Batch, 1, 256]
        key = value = self.audio_proj(audio_seq)   # [Batch, Time_Frames, 256]

        # 🚨 内鬼 1 核心修复：传入 key_padding_mask，防止注意力被静音稀释
        attn_output, _ = self.cross_attn(
            query=query, 
            key=key, 
            value=value, 
            key_padding_mask=key_padding_mask  
        )

        # =======================================
        # E. 分类预测
        # =======================================
        logits = self.classifier(attn_output.squeeze(1))  # 挤掉序列维度 -> [Batch, 3]

        return logits