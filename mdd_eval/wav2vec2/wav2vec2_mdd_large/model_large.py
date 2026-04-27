import torch
import torch.nn as nn
from transformers import Wav2Vec2Model
from peft import LoraConfig, get_peft_model  # 🚀 新增 PEFT 库

class MDD_Wav2vec_Large_LoRA_Model(nn.Module):
    def __init__(self, num_pinyins, wav2vec2_version="facebook_wav2vec2_large_960"):
        super().__init__()
        print(f"🧬 Model 3 (Large + LoRA + Attention) 初始化中... [wav2vec2 版]")

        # 1. 载入 wav2vec2 核心组件
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(wav2vec2_version)
        
        # 👑 禁用遮盖配置，防止短音频报错
        self.wav2vec2.config.mask_time_prob = 0.0
        self.wav2vec2.config.mask_feature_prob = 0.0

        # ==========================================
        # 🚀 核心升级：应用 LoRA 微调机制
        # ==========================================
        print("✨ 正在注入 LoRA 适配器...")
        lora_config = LoraConfig(
            r=16, 
            lora_alpha=32, 
            target_modules=["q_proj", "v_proj"], # 针对注意力矩阵的 Q 和 V 投影层
            lora_dropout=0.1, 
            bias="none"
        )
        self.wav2vec2 = get_peft_model(self.wav2vec2, lora_config)
        self.wav2vec2.print_trainable_parameters() # 打印一下参数量，你会发现只有约 1% 被训练

        # 获取音频特征维度 (wav2vec2 Large 为 1024)
        audio_dim = self.wav2vec2.config.hidden_size  
        hidden_dim = 256  # 统一注意力投影空间维度

        # 2. 局部特征提取器 (1D 卷积动态适配 1024 维)
        self.local_conv = nn.Sequential(
            nn.Conv1d(audio_dim, audio_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm1d(audio_dim)
        )

        # 3. 文本/音频 投影层 
        self.pinyin_embedding = nn.Embedding(num_pinyins + 1, hidden_dim)
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.text_proj = nn.Linear(hidden_dim, hidden_dim) 

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
            nn.Dropout(0.2), 
            nn.Linear(128, 3) # 输出 0, 1, 2
        )

    def forward(self, input_values, lengths, target_syllable_ids):
        # A. 提取音频序列特征
        outputs = self.wav2vec2(input_values)
        audio_seq = outputs.last_hidden_state  # 形状: [Batch, Time_Frames, 1024]

        # B. 计算 Attention Mask
        batch_size, max_frames, _ = audio_seq.shape
        feature_lengths = lengths // 320 
        
        seq_range = torch.arange(max_frames, device=audio_seq.device)
        key_padding_mask = seq_range.unsqueeze(0) >= feature_lengths.unsqueeze(1)

        # C. 1D 卷积增强局部特征
        x = audio_seq.transpose(1, 2)
        x = self.local_conv(x)
        audio_seq = x.transpose(1, 2)  # 换回 [Batch, Time_Frames, 1024]

        # D. 交叉注意力：带 Mask 的安全对齐
        text_vec = self.pinyin_embedding(target_syllable_ids).unsqueeze(1)
        query = self.text_proj(text_vec)           # [Batch, 1, 256]
        key = value = self.audio_proj(audio_seq)   # [Batch, Time_Frames, 256]

        attn_output, _ = self.cross_attn(
            query=query, 
            key=key, 
            value=value, 
            key_padding_mask=key_padding_mask  
        )

        # E. 分类预测
        logits = self.classifier(attn_output.squeeze(1))  # [Batch, 3]
        return logits