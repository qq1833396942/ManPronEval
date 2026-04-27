import torch
import torch.nn as nn
from transformers import HubertModel
from torch.nn.utils.rnn import pad_sequence

class APA_HuBERT_Base_Model(nn.Module):
    def __init__(self, num_pinyins, embed_dim=128, hubert_version="facebook/hubert-base-ls960"):
        super().__init__()
        print(f"🧬 APA 考官模型 (Base 版 + 显式交互 + 冻结底座) 初始化中...")

        self.hubert = HubertModel.from_pretrained(hubert_version)
        # 关闭遮盖防止短音频报错
        self.hubert.config.mask_time_prob = 0.0
        self.hubert.config.mask_feature_prob = 0.0

        # ====================================================
        # 🔒 救命操作：冻结 HuBERT 参数，防止灾难性遗忘！
        # ====================================================
        for param in self.hubert.parameters():
            param.requires_grad = False
        print("🔒 HuBERT 底座参数已完全冻结！")

        self.pinyin_embedding = nn.Embedding(num_embeddings=num_pinyins, embedding_dim=embed_dim)
        
        audio_dim = self.hubert.config.hidden_size # Base版是 768
        
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
        # 🧠 升级版融合层 (Explicit Interaction)
        # ====================================================
        # 拼接 u, v, |u-v|, u*v -> 4 * 256 = 1024维
        self.shared_mlp = nn.Sequential(
            nn.Linear(self.attn_dim * 4, 512),
            nn.LayerNorm(512),  # 替换 BatchNorm 防止初期方差过大导致的崩溃
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
        
        # 1. 提特征 (因为底座被冻结，这里只过前向传播，不计算底座梯度)
        outputs = self.hubert(waveforms)
        audio_features = outputs.last_hidden_state  # [B, Seq_Len, 768]

        # 2. 将每个音频段切片，并组装成带 Padding 的批次向量
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

        # 3. 准备 Text Query
        text_vec = self.pinyin_embedding(pinyin_ids) # [B, 128]
        
        q = self.proj_q(text_vec).unsqueeze(1)      # [B, 1, 256]
        k = self.proj_k(padded_audio)               # [B, max_len, 256]
        v = self.proj_v(padded_audio)               # [B, max_len, 256]
        
        attn_output, _ = self.cross_attn(
            query=q, 
            key=k, 
            value=v, 
            key_padding_mask=key_padding_mask
        )
        
        attn_vec = attn_output.squeeze(1) # [B, 256]
        
        # ====================================================
        # 🔍 4. 显式计算差异与相似度并融合
        # ====================================================
        u = q.squeeze(1)      # 映射后的文本特征
        v_feat = attn_vec     # 提纯后的音频特征
        
        diff = torch.abs(u - v_feat)  # 寻找发音错误 (不匹配项)
        prod = u * v_feat             # 强化正确发音 (重合度项)
        
        fusion_vec = torch.cat([u, v_feat, diff, prod], dim=-1) # [B, 1024]

        shared_feat = self.shared_mlp(fusion_vec)

        # 多维度预测
        score_ini = torch.sigmoid(self.head_initial(shared_feat))
        score_fin = torch.sigmoid(self.head_final(shared_feat))
        score_tot = torch.sigmoid(self.head_total(shared_feat))

        return score_ini, score_fin, score_tot