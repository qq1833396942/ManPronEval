import torch
import torch.nn as nn
from transformers import WavLMModel
from torch.nn.utils.rnn import pad_sequence


class APA_WavLM_Base_Model(nn.Module):
    def __init__(self, num_pinyins, embed_dim=128, wavlm_version="microsoft/wavlm-base"):
        super().__init__()
        print("🧬 APA 考官模型 (WavLM Base 版 + 显式交互 + 冻结底座 + Layer 8特征) 初始化中...")

        # 和 HuBERT Base 严格对齐：开启中间层输出
        self.wavlm = WavLMModel.from_pretrained(wavlm_version, output_hidden_states=True)

        # 关闭遮盖，和 HuBERT 版保持一致
        self.wavlm.config.mask_time_prob = 0.0
        self.wavlm.config.mask_feature_prob = 0.0

        # 冻结底座参数
        for param in self.wavlm.parameters():
            param.requires_grad = False
        print("🔒 WavLM Base 底座参数已完全冻结！")

        self.pinyin_embedding = nn.Embedding(num_embeddings=num_pinyins, embedding_dim=embed_dim)

        # WavLM Base hidden_size = 768
        audio_dim = self.wavlm.config.hidden_size

        # 与 HuBERT Base 严格保持一致
        self.attn_dim = 256
        self.num_heads = 4

        self.proj_q = nn.Linear(embed_dim, self.attn_dim)
        self.proj_k = nn.Linear(audio_dim, self.attn_dim)
        self.proj_v = nn.Linear(audio_dim, self.attn_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.attn_dim,
            num_heads=self.num_heads,
            batch_first=True
        )

        self.shared_mlp = nn.Sequential(
            nn.Linear(self.attn_dim * 4, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3)
        )

        self.head_initial = nn.Sequential(nn.Linear(512, 128), nn.GELU(), nn.Linear(128, 1))
        self.head_final = nn.Sequential(nn.Linear(512, 128), nn.GELU(), nn.Linear(128, 1))
        self.head_total = nn.Sequential(nn.Linear(512, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, waveforms, start_frames, end_frames, pinyin_ids):
        device = waveforms.device

        # 1. 提取声学特征
        outputs = self.wavlm(waveforms)

        # 和 HuBERT Base 严格对齐：也取第 8 层特征
        # hidden_states[0] 是前端输出，hidden_states[1]~[...] 是各层 Transformer 输出
        audio_features = outputs.hidden_states[8]   # [B, Seq_Len, 768]

        # 2. 根据标注边界切出目标音段
        audio_segments = []
        valid_lens = []

        for i in range(audio_features.size(0)):
            s_idx = start_frames[i].item()
            e_idx = end_frames[i].item()
            segment = audio_features[i, s_idx:e_idx, :]

            if segment.size(0) == 0:
                segment = audio_features[i, 0:1, :]

            audio_segments.append(segment)
            valid_lens.append(segment.size(0))

        padded_audio = pad_sequence(audio_segments, batch_first=True)

        max_len = padded_audio.size(1)
        valid_lens_t = torch.tensor(valid_lens, device=device)
        key_padding_mask = torch.arange(max_len, device=device).unsqueeze(0) >= valid_lens_t.unsqueeze(1)

        # 3. 文本查询向量
        text_vec = self.pinyin_embedding(pinyin_ids)   # [B, 128]

        q = self.proj_q(text_vec).unsqueeze(1)         # [B, 1, 256]
        k = self.proj_k(padded_audio)                  # [B, T, 256]
        v = self.proj_v(padded_audio)                  # [B, T, 256]

        attn_output, _ = self.cross_attn(
            query=q,
            key=k,
            value=v,
            key_padding_mask=key_padding_mask
        )

        attn_vec = attn_output.squeeze(1)              # [B, 256]

        # 4. 显式交互特征
        u = q.squeeze(1)
        v_feat = attn_vec

        diff = torch.abs(u - v_feat)
        prod = u * v_feat

        fusion_vec = torch.cat([u, v_feat, diff, prod], dim=-1)
        shared_feat = self.shared_mlp(fusion_vec)

        score_ini = torch.sigmoid(self.head_initial(shared_feat))
        score_fin = torch.sigmoid(self.head_final(shared_feat))
        score_tot = torch.sigmoid(self.head_total(shared_feat))

        return score_ini, score_fin, score_tot
