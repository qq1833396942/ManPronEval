import torch
import torch.nn as nn
from transformers import Wav2Vec2Model
from peft import LoraConfig, get_peft_model

class AttentionPooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, hidden_states):
        attn_weights = self.attention(hidden_states)
        attn_weights = torch.softmax(attn_weights, dim=1)
        pooled_output = torch.sum(hidden_states * attn_weights, dim=1)
        return pooled_output

class MultiTaskHubertLarge(nn.Module):
    def __init__(self, model_path, vocab_size):
        super().__init__()
        print(f"🚀 初始化 [2/3 冻结 + LoRA 微调] Hubert-Large MTL 模型...")
        self.hubert = Wav2Vec2Model.from_pretrained(model_path)
        hidden_size = self.hubert.config.hidden_size # Large 为 1024

        # --- 策略修改：冻结前 16 层 (2/3) ---
        # Hubert-Large 一共有 24 层 EncoderLayers
        for i, layer in enumerate(self.hubert.encoder.layers):
            if i < 16:
                for param in layer.parameters():
                    param.requires_grad = False
        print(f"✅ 已冻结前 16 层编码器，仅允许后 8 层及 LoRA 更新")

        # --- LoRA 配置 ---
        self.hubert.config.mask_time_prob = 0.0
        self.hubert.config.mask_feature_prob = 0.0
        
        lora_config = LoraConfig(
            r=16, 
            lora_alpha=32, 
            target_modules=["q_proj", "v_proj"], 
            lora_dropout=0.1, 
            bias="none"
        )
        self.hubert = get_peft_model(self.hubert, lora_config)

        self.text_embedding = nn.Embedding(vocab_size, hidden_size)

        # 池化器
        self.apa_pooler_L16 = AttentionPooler(hidden_size) # 专用冻结层池化
        self.apa_pooler_L24 = AttentionPooler(hidden_size) # 专用微调顶层池化
        self.mdd_pooler = AttentionPooler(hidden_size)
        self.asr_pooler = AttentionPooler(hidden_size)

        # 任务头
        self.apa_head = nn.Sequential(
            nn.Linear(hidden_size * 3, 512), 
            nn.GELU(),                       
            nn.Dropout(0.2),
            nn.Linear(512, 3) 
        )

        self.mdd_head = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 3)  
        )

        self.asr_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, vocab_size) 
        )

    def forward(self, input_values, target_pinyin_ids):
        outputs = self.hubert(input_values, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        
        # 提取第 16 层（冻结特征）和 第 24 层（微调特征）
        hidden_L16 = hidden_states[16]  
        hidden_L24 = hidden_states[24]

        text_features = self.text_embedding(target_pinyin_ids)

        # APA 任务：融合 L16(detach) 和 L24
        # 使用 detach 确保梯度绝对不会渗透进前 16 层
        apa_pooled_16 = self.apa_pooler_L16(hidden_L16.detach())
        apa_pooled_24 = self.apa_pooler_L24(hidden_L24)
        
        apa_fused = torch.cat([apa_pooled_16, apa_pooled_24, text_features], dim=1)
        apa_scores = self.apa_head(apa_fused)

        # MDD & ASR 任务：仅使用微调后的顶层特征
        mdd_pooled = self.mdd_pooler(hidden_L24)
        mdd_fused = torch.cat([mdd_pooled, text_features], dim=1)
        mdd_logits = self.mdd_head(mdd_fused)

        asr_pooled = self.asr_pooler(hidden_L24)
        asr_logits = self.asr_head(asr_pooled)

        return apa_scores, mdd_logits, asr_logits