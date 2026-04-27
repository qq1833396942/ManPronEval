#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import tgt
import librosa
import numpy as np
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
import json
from sklearn.metrics import accuracy_score

# --- 1. 基础配置 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dir = r"../../data/train"
vocab_path = r"syllable_vocab.json"
model_path = r"facebook_wav2vec2_large"

with open(vocab_path, "r", encoding="utf-8") as f:
    phoneme_vocab = json.load(f)
inv_vocab = {v: k for k, v in phoneme_vocab.items()}
vocab_size = len(phoneme_vocab)

# --- 2. 核心架构重写：单音节强制分类器 + LoRA ---
class SingleSyllableWav2Vec2(nn.Module):
    def __init__(self, model_path, vocab_size, use_lora=True, lora_r=8, lora_alpha=16):
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_path)

        if use_lora:
            # 应用LoRA微调
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.1,
                bias="none"
            )
            self.wav2vec2 = get_peft_model(self.wav2vec2, lora_config)
            print("LoRA配置已应用:")
            self.wav2vec2.print_trainable_parameters()
        else:
            # 完全冻结主干
            self.wav2vec2.feature_extractor._freeze_parameters()
            for param in self.wav2vec2.parameters():
                param.requires_grad = False

        hidden_size = self.wav2vec2.config.hidden_size

        # MLP 分类头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 1024),
            nn.GELU(),
            nn.LayerNorm(1024),
            nn.Dropout(0.1),
            nn.Linear(1024, vocab_size)
        )

        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values).last_hidden_state
        pooled_features = outputs.mean(dim=1)
        logits = self.classifier(pooled_features)
        return logits

model = SingleSyllableWav2Vec2(model_path, vocab_size, use_lora=True, lora_r=8, lora_alpha=16).to(device)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)

# --- 3. 数据处理 (加入 Trim 和防报错机制) ---
class PhonemeDataset(Dataset):
    def __init__(self, data_dir, feature_extractor, vocab):
        self.samples = []
        for root, _, files in os.walk(data_dir):
            for f in files:
                if f.endswith('.wav'):
                    p = os.path.join(root, f)
                    tg = p.replace('.wav', '.TextGrid').replace('.textgrid', '.TextGrid')
                    if os.path.exists(tg): self.samples.append((p, tg))
        self.fe = feature_extractor
        self.vocab = vocab
        print(f"成功加载样本数: {len(self.samples)}")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        wav, tg_p = self.samples[idx]
        try:
            s, _ = librosa.load(wav, sr=16000)
            
            # 加入你提议的 Trim，截去首尾 25dB 以下的静音
            s_trimmed, _ = librosa.effects.trim(s, top_db=25)
            
            # 安全防护：防止修剪后音频变空 (小于 10ms 强制填充)
            if len(s_trimmed) < 160: 
                s = np.zeros(16000) 
            else:
                s = s_trimmed
                
            s = (s - s.mean()) / (s.std() + 1e-6)
            input_values = self.fe(s, return_tensors="pt", sampling_rate=16000).input_values.squeeze(0)
            
            # 读取标签
            tg = tgt.io.read_textgrid(tg_p, encoding='utf-8')
            txt = tg.get_tier_by_name('syllables').intervals[0].text.strip().lower() 
            lbl = self.vocab.get(txt, 1) # 返回一个 int
            
        except Exception:
            input_values = torch.zeros(1600)
            lbl = 1 # UNK
            
        return {'input_values': input_values, 'labels': torch.tensor(lbl, dtype=torch.long)}

def collate_fn(batch):
    inputs = feature_extractor.pad(
        {"input_values": [x['input_values'] for x in batch]}, 
        padding=True, 
        return_tensors="pt"
    ).input_values
    # 注意这里标签变了，不再是序列，而是一个一维数组 [batch_size]
    labels = torch.stack([x['labels'] for x in batch])
    return {'input_values': inputs.to(device), 'labels': labels.to(device)}

train_loader = DataLoader(
    PhonemeDataset(data_dir, feature_extractor, phoneme_vocab), 
    batch_size=16, 
    shuffle=True, 
    collate_fn=collate_fn
)

# --- 4. 训练设置：分组学习率 ---
# 收集LoRA参数和分类头参数
lora_params = [p for p in model.wav2vec2.parameters() if p.requires_grad]
classifier_params = model.classifier.parameters()

optimizer = optim.AdamW([
    {'params': classifier_params, 'lr': 1e-3},  # CE头用较大学习率
    {'params': lora_params, 'lr': 5e-5}  # LoRA用小学习率
])
criterion = nn.CrossEntropyLoss()
epochs = 50

# --- 5. 训练循环 ---
for epoch in range(epochs):
    model.train()
    total_loss = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
    
    for batch in pbar:
        optimizer.zero_grad()
        
        # 1. 前向传播
        logits = model(batch['input_values']) 
        
        # 2. 计算普通的交叉熵损失
        loss = criterion(logits, batch['labels'])
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

    # --- 评估 (看准确率) ---
    model.eval()
    all_preds, all_refs = [], []
    with torch.no_grad():
        # 拿最后一个 batch 验证一下
        logits = model(batch['input_values'])
        pred_ids = torch.argmax(logits, dim=-1)
        
        all_preds = pred_ids.cpu().numpy()
        all_refs = batch['labels'].cpu().numpy()
        
        acc = accuracy_score(all_refs, all_preds)
        
        pred_txt = inv_vocab.get(all_preds[0], "[UNK]")
        ref_txt = inv_vocab.get(all_refs[0], "[UNK]")
        print(f"\n[DEBUG] 示例预测: '{pred_txt}' | 真实标签: '{ref_txt}'")
        print(f"Epoch {epoch+1} 训练集当前 Batch 准确率: {acc*100:.2f}%")

    torch.save(model.state_dict(), "wav2vec2_single_syllable.pth")

print("训练完成")