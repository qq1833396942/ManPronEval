import torch
import torch.nn as nn
from transformers import WhisperModel
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
        # hidden_states: [batch, seq_len, hidden_size]
        attn_weights = self.attention(hidden_states)
        attn_weights = torch.softmax(attn_weights, dim=1)
        pooled_output = torch.sum(hidden_states * attn_weights, dim=1)
        return pooled_output


class MultiTaskWhisperLarge(nn.Module):
    def __init__(self, model_path, vocab_size):
        super().__init__()
        print(f"🚀 初始化 [2/3 冻结 + LoRA] Whisper-Large MTL 模型...")

        # 加载 Whisper 核心模型
        self.whisper = WhisperModel.from_pretrained(model_path)
        self.encoder = self.whisper.encoder
        hidden_size = self.whisper.config.d_model  # Large 通常为 1280

        # --- 策略修改：冻结前 21 层 (约 2/3，共 32 层) ---
        # 冻结卷积层和位置编码
        self.encoder.conv1.requires_grad_(False)
        self.encoder.conv2.requires_grad_(False)
        self.encoder.embed_positions.requires_grad_(False)

        for i, layer in enumerate(self.encoder.layers):
            if i < 21:
                for param in layer.parameters():
                    param.requires_grad = False
        print(f"✅ 已冻结前 21 层编码器，仅允许后 11 层参与微调")

        # --- LoRA 配置 ---
        # target_modules 针对 Whisper 的线性层名
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "out_proj"],
            lora_dropout=0.1,
            bias="none"
        )
        self.encoder = get_peft_model(self.encoder, lora_config)

        self.text_embedding = nn.Embedding(vocab_size, hidden_size)

        # 池化器
        self.apa_pooler_L21 = AttentionPooler(hidden_size)  # 专用冻结层池化
        self.apa_pooler_L32 = AttentionPooler(hidden_size)  # 专用顶层池化
        self.mdd_pooler = AttentionPooler(hidden_size)
        self.asr_pooler = AttentionPooler(hidden_size)

        # 任务头 (输入维度根据 hidden_size 自动适配)
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

    def forward(self, input_features, target_pinyin_ids):
        # input_features: [batch, 80, 3000] 或 [batch, 128, 3000]
        outputs = self.encoder(input_features, output_hidden_states=True)
        hidden_states = outputs.hidden_states

        # 提取第 21 层（冻结特征）和 第 32 层（微调特征）
        # hidden_states[0] 是 embedding 层，所以索引需注意
        hidden_L21 = hidden_states[21]
        hidden_L32 = hidden_states[-1]

        text_features = self.text_embedding(target_pinyin_ids)

        # APA 任务：融合 L21(detach) 和 L32 以及文本特征
        apa_pooled_21 = self.apa_pooler_L21(hidden_L21.detach())
        apa_pooled_32 = self.apa_pooler_L32(hidden_L32)

        apa_fused = torch.cat([apa_pooled_21, apa_pooled_32, text_features], dim=1)
        apa_scores = self.apa_head(apa_fused)

        # MDD 任务
        mdd_pooled = self.mdd_pooler(hidden_L32)
        mdd_fused = torch.cat([mdd_pooled, text_features], dim=1)
        mdd_logits = self.mdd_head(mdd_fused)

        # ASR 任务
        asr_pooled = self.asr_pooler(hidden_L32)
        asr_logits = self.asr_head(asr_pooled)

        return apa_scores, mdd_logits, asr_logits