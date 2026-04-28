import torch
import torch.nn as nn
from transformers import WavLMModel
from torch.nn.utils.rnn import pad_sequence


class APA_WavLM_Large_Model(nn.Module):
    def __init__(self, num_pinyins, embed_dim=128, wavlm_version="microsoft/wavlm-large"):
        super().__init__()
        print("🧬 APA 考官模型 (WavLM Large 版 + 显式交互 + 冻结底座) 初始化中...")

        self.wavlm = WavLMModel.from_pretrained(wavlm_version, output_hidden_states=True)
        # 关闭 SpecAugment 遮盖，避免短音频报错；与原 HuBERT 版保持一致
        self.wavlm.config.mask_time_prob = 0.0
        self.wavlm.config.mask_feature_prob = 0.0

        # 严格对照：冻结底座参数，只训练交互层和打分头
        for param in self.wavlm.parameters():
            param.requires_grad = False
        print("🔒 WavLM Large 底座参数已完全冻结！")

        self.pinyin_embedding = nn.Embedding(num_embeddings=num_pinyins, embedding_dim=embed_dim)

        # WavLM Large hidden_size = 1024
        audio_dim = self.wavlm.config.hidden_size

        # 与原 HuBERT Large 版保持一致
        self.attn_dim = 384
        self.num_heads = 6

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

        # 严格对照原 Large 版：仍取第 16 层 hidden state
        # hidden_states: embedding 输出 + 每层 Transformer 输出
        audio_features = outputs.hidden_states[16]    # [B, Seq_Len, 1024]

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

        q = self.proj_q(text_vec).unsqueeze(1)         # [B, 1, 384]
        k = self.proj_k(padded_audio)                  # [B, T, 384]
        v = self.proj_v(padded_audio)                  # [B, T, 384]

        attn_output, _ = self.cross_attn(
            query=q,
            key=k,
            value=v,
            key_padding_mask=key_padding_mask
        )

        attn_vec = attn_output.squeeze(1)              # [B, 384]

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
