import torch
import torch.nn as nn
from transformers import WhisperModel, WhisperFeatureExtractor
from torch.nn.utils.rnn import pad_sequence


class APA_Whisper_Large_Model(nn.Module):
    def __init__(self, num_pinyins, embed_dim=128,
                 whisper_version=r"C:\Users\14183\OneDrive\Desktop\MDD-whisper\whisper-large-local"):
        super().__init__()
        print(f"🧬 APA 考官模型 (Whisper-Large 版 + 显式交互 + 冻结底座) 初始化中...")

        # 1. 加载 Whisper-Large 核心特征提取器和底座
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(whisper_version)
        self.whisper = WhisperModel.from_pretrained(whisper_version)

        # ====================================================
        # 🔒 公平对齐 1：冻结 Whisper-Large 底座参数
        # ====================================================
        for param in self.whisper.parameters():
            param.requires_grad = False
        print("🔒 Whisper Large 底座参数已完全冻结！(省显存保平安)")

        self.pinyin_embedding = nn.Embedding(num_embeddings=num_pinyins, embedding_dim=embed_dim)

        # 自动获取 Large 版的维度 (Whisper-Large 为 1280)
        audio_dim = self.whisper.config.d_model

        # ====================================================
        # 👑 公平对齐 2：核心交叉注意力组件 (完全对齐 HuBERT)
        # ====================================================
        self.attn_dim = 384
        self.num_heads = 6  # 384 可以被 6 整除

        self.proj_q = nn.Linear(embed_dim, self.attn_dim)
        self.proj_k = nn.Linear(audio_dim, self.attn_dim)
        self.proj_v = nn.Linear(audio_dim, self.attn_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.attn_dim,
            num_heads=self.num_heads,
            batch_first=True
        )

        # ====================================================
        # 🧠 公平对齐 3：升级版融合层 (Explicit Interaction)
        # ====================================================
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

        # 兵分三路：多任务打分头
        self.head_initial = nn.Sequential(nn.Linear(512, 128), nn.GELU(), nn.Linear(128, 1))
        self.head_final = nn.Sequential(nn.Linear(512, 128), nn.GELU(), nn.Linear(128, 1))
        self.head_total = nn.Sequential(nn.Linear(512, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, waveforms, start_frames, end_frames, pinyin_ids):
        device = waveforms.device

        # 1. 提特征预处理 (波形转 Mel 频谱)
        inputs = self.feature_extractor(
            waveforms.cpu().numpy(),
            sampling_rate=16000,
            return_tensors="pt"
        )
        # 🛡️ 类型安全守护：确保输入特征与 Whisper 底座 (通常是 fp16) 精度匹配
        input_features = inputs.input_features.to(device).to(self.whisper.dtype)

        # 2. Whisper Encoder 提取序列特征
        # 🛡️ 加上 no_grad 彻底阻断底座计算图，极大节省 4070Ti 显存
        with torch.no_grad():
            # 注意：传入 output_hidden_states=True 才能拿到中间层
            encoder_outputs = self.whisper.encoder(input_features, output_hidden_states=True)

            # ====================================================
            # 🎯 公平对齐 4：抽取特定深度的网络层
            # HuBERT (24层) 抽了第 16 层。
            # Whisper Large (32层) 我们按同等深度比例抽取第 24 层！
            # ====================================================
            audio_features = encoder_outputs.hidden_states[24]  # [B, 1500, 1280]

        # 🚀 强制转回 float32，对接下游自定义组件防止报错
        audio_features = audio_features.float()

        # 3. 音频切片与 Padding (Whisper 帧数上限为 1500)
        audio_segments = []
        valid_lens = []

        for i in range(audio_features.size(0)):
            s_idx = start_frames[i].item()
            e_idx = end_frames[i].item()

            # Whisper 防越界保护
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

        # 4. 准备 Text Query
        text_vec = self.pinyin_embedding(pinyin_ids)  # [B, 128]

        q = self.proj_q(text_vec).unsqueeze(1)  # [B, 1, 384]
        k = self.proj_k(padded_audio)  # [B, max_len, 384]
        v = self.proj_v(padded_audio)  # [B, max_len, 384]

        attn_output, _ = self.cross_attn(
            query=q,
            key=k,
            value=v,
            key_padding_mask=key_padding_mask
        )

        attn_vec = attn_output.squeeze(1)  # [B, 384]

        # ====================================================
        # 🔍 5. 显式计算差异与相似度并融合 (严格对齐)
        # ====================================================
        u = q.squeeze(1)
        v_feat = attn_vec

        diff = torch.abs(u - v_feat)
        prod = u * v_feat

        # 384 * 4 = 1536 维，无缝对接下游 512 维的 MLP
        fusion_vec = torch.cat([u, v_feat, diff, prod], dim=-1)  # [B, 1536]

        shared_feat = self.shared_mlp(fusion_vec)

        score_ini = torch.sigmoid(self.head_initial(shared_feat))
        score_fin = torch.sigmoid(self.head_final(shared_feat))
        score_tot = torch.sigmoid(self.head_total(shared_feat))

        return score_ini, score_fin, score_tot