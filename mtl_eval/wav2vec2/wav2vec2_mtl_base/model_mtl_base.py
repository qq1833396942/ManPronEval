import torch
import torch.nn as nn
from transformers import Wav2Vec2Model


class AttentionPooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, hidden_states, attention_mask=None):
        # hidden_states: [B, T, H]
        attn_scores = self.attention(hidden_states).squeeze(-1)  # [B, T]

        if attention_mask is not None:
            attn_scores = attn_scores.masked_fill(attention_mask == 0, float("-inf"))

        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)  # [B, T, 1]
        pooled_output = torch.sum(hidden_states * attn_weights, dim=1)  # [B, H]
        return pooled_output


class MultiTaskWav2vec2(nn.Module):
    def __init__(self, model_path, vocab_size):
        super().__init__()
        print("🚀 初始化 Wav2Vec2 MTL 模型 (APA 融合 L4+L8 特征)...")

        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_path)
        hidden_size = self.wav2vec2.config.hidden_size

        # =========================
        # 冻结策略
        # =========================
        self.wav2vec2.feature_extractor._freeze_parameters()

        for param in self.wav2vec2.feature_projection.parameters():
            param.requires_grad = False

        for param in self.wav2vec2.encoder.pos_conv_embed.parameters():
            param.requires_grad = False

        # Wav2Vec2-Base: 12层
        # 冻结前6层，训练后6层
        for i in range(6):
            for param in self.wav2vec2.encoder.layers[i].parameters():
                param.requires_grad = False

        for i in range(6, 12):
            for param in self.wav2vec2.encoder.layers[i].parameters():
                param.requires_grad = True

        self.text_embedding = nn.Embedding(vocab_size, hidden_size)

        # APA: 声学层 + 音素层
        self.apa_pooler_L4 = AttentionPooler(hidden_size)
        self.apa_pooler_L8 = AttentionPooler(hidden_size)

        self.mdd_pooler = AttentionPooler(hidden_size)
        self.asr_pooler = AttentionPooler(hidden_size)

        # =========================
        # 任务头
        # =========================
        # L4 + L8 + text = 3 * hidden_size
        self.apa_head = nn.Sequential(
            nn.Linear(hidden_size * 3, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 3)
        )

        # MDD: phoneme + text
        self.mdd_head = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 3)
        )

        # ASR: top layer
        self.asr_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, vocab_size)
        )

    def forward(self, input_values, target_pinyin_ids, attention_mask=None):
        outputs = self.wav2vec2(
            input_values,
            output_hidden_states=True
        )
        hidden_states = outputs.hidden_states

        # Wav2Vec2-Base:
        # hidden_states[0]  = conv/embedding output
        # hidden_states[1]  = encoder layer 1
        # ...
        # hidden_states[12] = encoder layer 12
        hidden_L4 = hidden_states[4]
        hidden_L8 = hidden_states[8]
        hidden_L12 = hidden_states[12]

        text_features = self.text_embedding(target_pinyin_ids)  # [B, H]

        # 将 waveform 级别 mask 映射到 feature 序列长度
        feature_mask = None
        if attention_mask is not None:
            feature_mask = self.wav2vec2._get_feature_vector_attention_mask(
                hidden_L12.shape[1], attention_mask
            ).long()

        # APA 分支：只读特征，不回传到底座
        apa_pooled_4 = self.apa_pooler_L4(hidden_L4.detach(), feature_mask)
        apa_pooled_8 = self.apa_pooler_L8(hidden_L8.detach(), feature_mask)

        # MDD / ASR 分支正常训练
        mdd_pooled = self.mdd_pooler(hidden_L8, feature_mask)
        asr_pooled = self.asr_pooler(hidden_L12, feature_mask)

        # APA: acoustic + phoneme + target text
        apa_fused = torch.cat([apa_pooled_4, apa_pooled_8, text_features], dim=1)
        apa_scores = self.apa_head(apa_fused)

        # MDD: phoneme + target text
        mdd_fused = torch.cat([mdd_pooled, text_features], dim=1)
        mdd_logits = self.mdd_head(mdd_fused)

        # ASR: top semantic/task layer
        asr_logits = self.asr_head(asr_pooled)

        return apa_scores, mdd_logits, asr_logits