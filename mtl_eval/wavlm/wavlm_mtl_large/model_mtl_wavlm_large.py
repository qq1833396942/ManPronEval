import torch
import torch.nn as nn
from transformers import WavLMModel
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


class MultiTaskWavLMLarge(nn.Module):
    def __init__(self, model_path, vocab_size):
        super().__init__()
        print("🚀 初始化 [2/3 冻结 + LoRA 微调] WavLM-Large MTL 模型...")
        self.wavlm = WavLMModel.from_pretrained(model_path, local_files_only=True)
        hidden_size = self.wavlm.config.hidden_size  # Large 为 1024

        # --- 策略：冻结前 16 层 (2/3)，与 HuBERT-Large 严格对照 ---
        for i, layer in enumerate(self.wavlm.encoder.layers):
            if i < 16:
                for param in layer.parameters():
                    param.requires_grad = False
        print("✅ 已冻结前 16 层编码器，仅允许后 8 层及 LoRA 更新")

        # --- LoRA 配置：与 HuBERT-Large 保持一致 ---
        self.wavlm.config.mask_time_prob = 0.0
        self.wavlm.config.mask_feature_prob = 0.0

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.1,
            bias="none"
        )
        self.wavlm = get_peft_model(self.wavlm, lora_config)

        self.text_embedding = nn.Embedding(vocab_size, hidden_size)

        # 池化器
        self.apa_pooler_L16 = AttentionPooler(hidden_size)  # 冻结层池化
        self.apa_pooler_L24 = AttentionPooler(hidden_size)  # 顶层池化
        self.mdd_pooler = AttentionPooler(hidden_size)
        self.asr_pooler = AttentionPooler(hidden_size)

        # 任务头：与 HuBERT-Large 保持一致
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
        outputs = self.wavlm(input_values, output_hidden_states=True)
        hidden_states = outputs.hidden_states

        # 提取第 16 层（冻结特征）和第 24 层（顶层特征）
        hidden_L16 = hidden_states[16]
        hidden_L24 = hidden_states[24]

        text_features = self.text_embedding(target_pinyin_ids)

        # APA：融合 L16(detach) 和 L24，与 HuBERT-Large 保持一致
        apa_pooled_16 = self.apa_pooler_L16(hidden_L16.detach())
        apa_pooled_24 = self.apa_pooler_L24(hidden_L24)

        apa_fused = torch.cat([apa_pooled_16, apa_pooled_24, text_features], dim=1)
        apa_scores = self.apa_head(apa_fused)

        # MDD 与 ASR：使用顶层特征
        mdd_pooled = self.mdd_pooler(hidden_L24)
        mdd_fused = torch.cat([mdd_pooled, text_features], dim=1)
        mdd_logits = self.mdd_head(mdd_fused)

        asr_pooled = self.asr_pooler(hidden_L24)
        asr_logits = self.asr_head(asr_pooled)

        return apa_scores, mdd_logits, asr_logits
