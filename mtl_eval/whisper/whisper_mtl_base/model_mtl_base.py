import torch
import torch.nn as nn
from transformers import WhisperModel


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


class MultiTaskWhisper(nn.Module):
    def __init__(self, model_path, vocab_size):
        super().__init__()
        print(f"🚀 初始化 Whisper-Base MTL 模型...")

        self.whisper = WhisperModel.from_pretrained(model_path)
        self.encoder = self.whisper.encoder
        hidden_size = self.whisper.config.d_model  # Whisper-base 是 512

        # 🧊 冻结策略 (针对 6 层的 Base: 冻 4 层，开 2 层)
        self.encoder.conv1.requires_grad_(False)
        self.encoder.conv2.requires_grad_(False)
        self.encoder.embed_positions.requires_grad_(False)

        for i in range(4):
            for param in self.encoder.layers[i].parameters():
                param.requires_grad = False
        for i in range(4, 6):
            for param in self.encoder.layers[i].parameters():
                param.requires_grad = True

        self.text_embedding = nn.Embedding(vocab_size, hidden_size)

        # 池化层
        self.apa_pooler_mid = AttentionPooler(hidden_size)
        self.mdd_pooler = AttentionPooler(hidden_size)
        self.asr_pooler = AttentionPooler(hidden_size)

        # 🌟 APA 交互头 (4倍维度输入)
        self.apa_head = nn.Sequential(
            nn.Linear(hidden_size * 4, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 3),
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
        # input_features: [batch, 80, 3000]
        outputs = self.encoder(input_features, output_hidden_states=True)
        hidden_states = outputs.hidden_states

        hidden_mid = hidden_states[4]
        hidden_top = hidden_states[-1]

        text_features = self.text_embedding(target_pinyin_ids)

        apa_v_feat = self.apa_pooler_mid(hidden_mid.detach())
        mdd_v_feat = self.mdd_pooler(hidden_top)
        asr_v_feat = self.asr_pooler(hidden_top)

        # 显式交互: u, v, |u-v|, u*v
        u, v = text_features, apa_v_feat
        apa_fused = torch.cat([u, v, torch.abs(u - v), u * v], dim=1)
        apa_scores = self.apa_head(apa_fused)

        mdd_fused = torch.cat([mdd_v_feat, text_features], dim=1)
        mdd_logits = self.mdd_head(mdd_fused)

        asr_logits = self.asr_head(asr_v_feat)

        return apa_scores, mdd_logits, asr_logits