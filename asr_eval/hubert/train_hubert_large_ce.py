#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HuBERT Large + LoRA + CrossEntropy (CE)
加载最优模型继续训练脚本
"""

import os
# 设置 Hugging Face 国内镜像站
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json
import torch
import torch.nn as nn
import torch.optim as optim
import soundfile as sf
import numpy as np
import tgt
import librosa
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import HubertModel, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, PeftModel
from jiwer import wer 

# ==========================================
# 1. 基础配置
# ==========================================
class Config:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = r""#填入数据集目录 todo
    vocab_path = "syllable_vocab.json"
    
    # 🚨 已替换为 HuBERT Large 官方模型（自动下载，无需本地路径）
    model_path = "facebook/hubert-large-ls960-ft"  
    
    # 🚨 你的模型保存目录（建议新建，避免和base版本混淆）
    output_dir = "./train_out_hubert_large_lora_ce_new"
    
    batch_size = 16              
    gradient_accumulation_steps = 1
    epochs = 20  # 再跑 20 个 epoch
    learning_rate = 1e-4
    lora_r = 16
    num_workers = 2

# ==========================================  
# 2. 数据处理类（修正了标签对齐逻辑）  
# ==========================================  
class SingleSyllableDataset(Dataset):
    def __init__(self, paired_files, vocab, split_name="数据", min_duration=0.3, min_samples=8000):
        self.vocab = vocab
        self.samples = []
        filtered_count = 0
        total_count = 0

        print(f"🔍 正在进行 {split_name} 的单音节标签提取...")
        for audio_path, tg_path in tqdm(paired_files, desc=f"加载 {split_name}"):
            try:
                total_count += 1
                with sf.SoundFile(audio_path) as f:
                    duration = len(f) / f.samplerate
                    num_samples = len(f)

                # 过滤太短的音频（时长或采样点数）
                if duration < min_duration or num_samples < min_samples:
                    filtered_count += 1
                    continue

                try:
                    tg = tgt.io.read_textgrid(tg_path, encoding='utf-8')
                except:
                    tg = tgt.io.read_textgrid(tg_path, encoding='utf-16')

                tier = tg.get_tier_by_name('syllables')

                # 寻找这段音频里的有效音节（忽略静音）
                audio_label = self.vocab.get("[UNK]", 1)
                for interval in tier.intervals:
                    p = interval.text.strip()
                    if p and p.lower() not in ['sil', 'sp', 'none', '']:
                        audio_label = self.vocab.get(p, self.vocab.get("[UNK]", 1))
                        break # 找到了单音节，直接跳出循环！整段音频就用这1个标签

                self.samples.append((audio_path, audio_label))
            except Exception:
                filtered_count += 1
                continue

        print(f"✅ {split_name} 加载完成: {len(self.samples)}/{total_count} 个样本 (过滤: {filtered_count})")  

    def __len__(self): return len(self.samples)  
    
    def __getitem__(self, idx):  
        audio_path, label = self.samples[idx]  
        speech, _ = sf.read(audio_path)  
        if len(speech.shape) > 1: speech = speech[:, 0]  
        
        speech = (speech - np.mean(speech)) / (np.std(speech) + 1e-6)  
        # 注意这里标签直接就是一个 int 整数了  
        return {"input_values": torch.tensor(speech, dtype=torch.float32), "labels": label}

def ce_collate_fn(batch):
    # 依然需要对输入音频进行 Padding
    max_input_len = max(item['input_values'].shape[0] for item in batch)
    input_values = [torch.cat([item['input_values'], torch.zeros(max_input_len - item['input_values'].shape[0])]) for item in batch]
    
    # 标签现在是单个整数，直接转换为 1D Tensor 即可！
    labels = torch.tensor([item['labels'] for item in batch], dtype=torch.long)
    
    return {"input_values": torch.stack(input_values), "labels": labels}

class HubertForCE(nn.Module):
    def __init__(self, model_path, num_classes, lora_config):
        super().__init__()
        self.hubert = HubertModel.from_pretrained(model_path)
        self.hubert = get_peft_model(self.hubert, lora_config)
        
        # 🚨 适配 Large 模型：输入维度从 768 → 1024
        self.classifier = nn.Sequential(
            nn.Linear(1024, 1024),  # 768 → 1024 (Large 特征维度)
            nn.GELU(),             
            nn.LayerNorm(1024),    
            nn.Dropout(0.1),
            nn.Linear(1024, num_classes)
        )

    def forward(self, input_values, labels=None):
        outputs = self.hubert(input_values)
        sequence_output = outputs.last_hidden_state
        
        # 核心：将帧特征拍扁为 1 个全局特征
        pooled_output = sequence_output.mean(dim=1)
        
        logits = self.classifier(pooled_output) # 形状: [batch_size, num_classes]
        
        loss = None
        if labels is not None:
            # 此时 labels 的形状是 [batch_size]，logits 的形状是 [batch_size, num_classes]
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            
        return {"loss": loss, "logits": logits}

def decode_frames(frame_ids, inv_vocab, pad_id):
    collapsed, last_id = [], -1
    for idx in frame_ids:
        idx = int(idx)
        if idx == -100: continue
        if idx != last_id:
            if idx != pad_id: collapsed.append(idx)
            last_id = idx
    return " ".join([inv_vocab.get(i, "[UNK]") for i in collapsed])

def get_paired_files(search_dir):
    paired = []
    for root, dirs, files in os.walk(search_dir):
        for file in files:
            if file.lower().endswith('.wav'):
                audio_path = os.path.join(root, file)
                tg_path = os.path.join(root, file.replace('.wav', '.TextGrid'))
                if not os.path.exists(tg_path): tg_path = os.path.join(root, file.replace('.wav', '.textgrid'))
                if os.path.exists(tg_path): paired.append((audio_path, tg_path))
    return paired

# ==========================================
# 3. 主程序 (包含加载逻辑)
# ==========================================
def main():
    cfg = Config()
    
    with open(cfg.vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    inv_vocab = {v: k for k, v in vocab.items()}
    pad_id = vocab.get("[PAD]", 0)

    # --- 数据准备 ---
    train_files = get_paired_files(os.path.join(cfg.data_dir, "train"))
    val_files = get_paired_files(os.path.join(cfg.data_dir, "val"))
    train_dataset = SingleSyllableDataset(train_files, vocab, "训练集")
    val_dataset = SingleSyllableDataset(val_files, vocab, "验证集")
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=ce_collate_fn, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, collate_fn=ce_collate_fn, num_workers=cfg.num_workers)

    # --- 模型初始化 ---
    lora_config = LoraConfig(
        r=cfg.lora_r, lora_alpha=32, target_modules=["q_proj", "v_proj"], 
        lora_dropout=0.1, bias="none"
    )
    model = HubertForCE(cfg.model_path, len(vocab), lora_config)

    # 重要：Large 模型不加载旧的 Base 模型权重，从头训练
    model.to(cfg.device)

    # 修改优化器部分：分层学习率
    lora_params = [p for n, p in model.named_parameters() if "lora" in n and p.requires_grad]
    classifier_params = model.classifier.parameters()

    optimizer = optim.AdamW([
        {'params': classifier_params, 'lr': 1e-3}, # 分类头：快跑
        {'params': lora_params, 'lr': 5e-5}       # LoRA层：慢走，保护预训练特征
    ])
    scaler = torch.cuda.amp.GradScaler()

    # 加入学习率调度器 (Warmup & Scheduler)
    num_training_steps = len(train_loader) * cfg.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * num_training_steps), # 前10%步数预热
        num_training_steps=num_training_steps
    )

    print(f"\n🚀 开始 HuBERT Large + LoRA 训练... 目标再跑 {cfg.epochs} 个 Epoch")
    best_per = float("inf")
    
    for epoch in range(cfg.epochs):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs} [训练]")
        for i, batch in enumerate(pbar):
            x = batch["input_values"].to(cfg.device); y = batch["labels"].to(cfg.device)
            with torch.autocast(device_type='cuda'):
                outputs = model(x, labels=y); loss = outputs["loss"] / cfg.gradient_accumulation_steps
            scaler.scale(loss).backward()
            if (i + 1) % cfg.gradient_accumulation_steps == 0:
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(); scheduler.step()
            train_loss += loss.item() * cfg.gradient_accumulation_steps
            pbar.set_postfix({"loss": f"{loss.item()*cfg.gradient_accumulation_steps:.4f}"})

        # 验证
        model.eval(); all_refs, all_hyps = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="验证中"):
                x = batch["input_values"].to(cfg.device); y = batch["labels"].to(cfg.device)
                with torch.autocast(device_type='cuda'):
                    outputs = model(x, labels=y); logits = outputs["logits"]
                pred_ids = torch.argmax(logits, dim=-1).cpu().numpy(); label_ids = y.cpu().numpy()
                for j in range(len(pred_ids)):
                    all_refs.append(inv_vocab.get(label_ids[j], "[UNK]"))
                    all_hyps.append(inv_vocab.get(pred_ids[j], "[UNK]"))
        
        # 采样展示
        print("\n📝 验证集采样展示 (前 3 个):")
        for k in range(min(3, len(all_refs))):
            if all_refs[k].strip():
                print(f"  [样本 {k+1}]")
                print(f"  预测 (Hyp): {all_hyps[k]}")
                print(f"  真实 (Ref): {all_refs[k]}")
                print("-" * 30)
 
        current_per = wer(all_refs, all_hyps)
        print(f"\n📊 Epoch {epoch+1} 总结: Loss: {train_loss/len(train_loader):.4f}, PER: {current_per:.4f}")
        
        if current_per < best_per:
            best_per = current_per
            model.hubert.save_pretrained(cfg.output_dir)
            torch.save(model.classifier.state_dict(), os.path.join(cfg.output_dir, "classifier_head.pth"))
            print(f"🌟 发现更好的 PER！模型已更新保存。\n")

if __name__ == "__main__":
    main()