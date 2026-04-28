import torch
import torch.nn as nn
from transformers import WhisperModel, WhisperFeatureExtractor
from torch.nn.utils.rnn import pad_sequence


class APA_Whisper_Base_Model(nn.Module):
    def __init__(self, num_pinyins, embed_dim=128, whisper_version="openai/whisper-base"):
        super().__init__()
        print(f"🧬 APA 考官模型 (Whisper-Base 版 + 显式交互 + 冻结底座) 初始化中...")

        # 1. 加载 Whisper-Base 核心特征提取器和底座
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(whisper_version)
        self.whisper = WhisperModel.from_pretrained(whisper_version)

        # ====================================================
        # 🔒 公平对齐 1：冻结 Whisper 底座参数，防止灾难性遗忘！
        # ====================================================
        for param in self.whisper.parameters():
            param.requires_grad = False
        print("🔒 Whisper 底座参数已完全冻结！")

        audio_dim = self.whisper.config.d_model  # Whisper-Base 为 512

        self.pinyin_embedding = nn.Embedding(num_embeddings=num_pinyins, embedding_dim=embed_dim)

        # ====================================================
        # 👑 核心魔法：交叉注意力组件 (Cross-Attention)
        # ====================================================
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

        # ====================================================
        # 🧠 公平对齐 2 & 3：升级版融合层 (Explicit Interaction & LayerNorm)
        # ====================================================
        # 拼接 u, v, |u-v|, u*v -> 4 * 256 = 1024维
        self.shared_mlp = nn.Sequential(
            nn.Linear(self.attn_dim * 4, 512),
            nn.LayerNorm(512),  # 统一使用 LayerNorm 防止初期方差过大
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3)
        )

        # 兵分三路：多任务打分头
        self.head_initial = nn.Sequential(nn.Linear(512, 128), nn.GELU(), nn.Linear(128, 1))
        self.head_final = nn.Sequential(nn.Linear(512, 128), nn.GELU(), nn.Linear(128, 1))
        self.head_total = nn.Sequential(nn.Linear(512, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, waveforms, start_frames, end_frames, pinyin_ids):
        device = waveforms.device

        # 1. 提特征
        inputs = self.feature_extractor(
            waveforms.cpu().numpy(),
            sampling_rate=16000,
            return_tensors="pt"
        )
        input_features = inputs.input_features.to(device).to(self.whisper.dtype)

        # 2. Whisper Encoder 提取序列特征
        # 🛡️ 加上 torch.no_grad()，因为底座已冻结，能极大地省显存并提速
        with torch.no_grad():
            encoder_outputs = self.whisper.encoder(input_features)
            audio_features = encoder_outputs.last_hidden_state  # [B, 1500, 512]

        audio_features = audio_features.float()

        # 3. 音频切片与 Padding
        audio_segments = []
        valid_lens = []

        for i in range(audio_features.size(0)):
            s_idx = start_frames[i].item()
            e_idx = end_frames[i].item()

            s_idx = min(s_idx, 1499)
            e_idx = min(e_idx, 1500)
            if e_idx <= s_idx:
                e_idx = s_idx + 1

            segment = audio_features[i, s_idx:e_idx, :]
            audio_segments.append(segment)
            valid_lens.append(segment.size(0))

        padded_audio = pad_sequence(audio_segments, batch_first=True)
        max_len = padded_audio.size(1)
        valid_lens_t = torch.tensor(valid_lens, device=device)
        key_padding_mask = torch.arange(max_len, device=device).unsqueeze(0) >= valid_lens_t.unsqueeze(1)

        # 4. 文本提示与交叉注意力
        text_vec = self.pinyin_embedding(pinyin_ids)

        q = self.proj_q(text_vec).unsqueeze(1)
        k = self.proj_k(padded_audio)
        v = self.proj_v(padded_audio)

        attn_output, _ = self.cross_attn(query=q, key=k, value=v, key_padding_mask=key_padding_mask)
        attn_vec = attn_output.squeeze(1)

        # ====================================================
        # 🔍 5. 显式计算差异与相似度并融合 (完全复刻 HuBERT 逻辑)
        # ====================================================
        u = q.squeeze(1)  # 映射后的文本特征
        v_feat = attn_vec  # 提纯后的音频特征

        diff = torch.abs(u - v_feat)  # 寻找发音错误 (不匹配项)
        prod = u * v_feat  # 强化正确发音 (重合度项)

        fusion_vec = torch.cat([u, v_feat, diff, prod], dim=-1)  # [B, 1024]

        shared_feat = self.shared_mlp(fusion_vec)

        # 6. 多维度打分
        score_ini = torch.sigmoid(self.head_initial(shared_feat))
        score_fin = torch.sigmoid(self.head_final(shared_feat))
        score_tot = torch.sigmoid(self.head_total(shared_feat))

        return score_ini, score_fin, score_tot