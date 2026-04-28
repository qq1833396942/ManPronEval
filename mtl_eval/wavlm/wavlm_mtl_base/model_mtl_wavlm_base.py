import torch
import torch.nn as nn
from transformers import WavLMModel


class AttentionPooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False),
        )

    def forward(self, hidden_states):
        attn_weights = self.attention(hidden_states)
        attn_weights = torch.softmax(attn_weights, dim=1)
        pooled_output = torch.sum(hidden_states * attn_weights, dim=1)
        return pooled_output


class MultiTaskWavLMBase(nn.Module):
    def __init__(self, model_path, vocab_size):
        super().__init__()
        print("🚀 初始化 WavLM-Base 多任务模型 (严格对照 HuBERT-Base 版本)...")
        self.wavlm = WavLMModel.from_pretrained(model_path, local_files_only=True)
        hidden_size = self.wavlm.config.hidden_size

        self.wavlm.feature_extractor._freeze_parameters()

        for param in self.wavlm.feature_projection.parameters():
            param.requires_grad = False

        for param in self.wavlm.encoder.pos_conv_embed.parameters():
            param.requires_grad = False

        for i in range(8):
            for param in self.wavlm.encoder.layers[i].parameters():
                param.requires_grad = False

        for i in range(8, 12):
            for param in self.wavlm.encoder.layers[i].parameters():
                param.requires_grad = True

        self.text_embedding = nn.Embedding(vocab_size, hidden_size)

        self.apa_pooler_L8 = AttentionPooler(hidden_size)
        self.mdd_pooler = AttentionPooler(hidden_size)
        self.asr_pooler = AttentionPooler(hidden_size)

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
            nn.Linear(256, 3),
        )

        self.asr_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, vocab_size),
        )

    def forward(self, input_values, target_pinyin_ids):
        outputs = self.wavlm(input_values, output_hidden_states=True)
        hidden_states = outputs.hidden_states

        hidden_L8 = hidden_states[8]
        hidden_L12 = hidden_states[12]

        text_features = self.text_embedding(target_pinyin_ids)

        apa_pooled_8 = self.apa_pooler_L8(hidden_L8.detach())
        mdd_pooled = self.mdd_pooler(hidden_L12)
        asr_pooled = self.asr_pooler(hidden_L12)

        u = text_features
        v_feat = apa_pooled_8
        diff = torch.abs(u - v_feat)
        prod = u * v_feat

        apa_fused = torch.cat([u, v_feat, diff, prod], dim=1)
        apa_scores = self.apa_head(apa_fused)

        mdd_fused = torch.cat([mdd_pooled, text_features], dim=1)
        mdd_logits = self.mdd_head(mdd_fused)

        asr_logits = self.asr_head(asr_pooled)

        return apa_scores, mdd_logits, asr_logits
