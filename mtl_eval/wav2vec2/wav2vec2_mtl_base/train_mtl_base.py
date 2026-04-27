import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, f1_score

from dataset_mtl import MTLDataset, mtl_collate_fn
from model_mtl_base import MultiTaskWav2vec2

def train_and_validate_mtl():
    # ==========================================
    # ⚙️ 1. 配置区
    # ==========================================
    LOCAL_MODEL_PATH = r"facebook_wav2vec2_base_960"
    BASE_DIR = r"./"
    
    VOCAB_PATH = os.path.join(BASE_DIR, 'syllable_vocab.json')
    TRAIN_JSON_PATH = os.path.join(BASE_DIR, 'metadata_train_mtl.json')
    VAL_JSON_PATH = os.path.join(BASE_DIR, 'metadata_val_mtl.json') 
    
    epochs = 50
    batch_size = 16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 计算设备: {device} | 严格单学习率模式 (1e-4)")

    # ==========================================
    # 📚 2. 加载数据
    # ==========================================
    with open(VOCAB_PATH, 'r', encoding='utf-8') as f:
        pinyin2id = json.load(f)
    vocab_size = len(pinyin2id)

    train_dataset = MTLDataset(TRAIN_JSON_PATH, pinyin2id)
    val_dataset = MTLDataset(VAL_JSON_PATH, pinyin2id)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=mtl_collate_fn, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=mtl_collate_fn, num_workers=4)

    # ==========================================
    # 🚀 3. 模型与优化器 (严格对齐 1e-4)
    # ==========================================
    model = MultiTaskWav2vec2(model_path=LOCAL_MODEL_PATH, vocab_size=vocab_size).to(device)

    # 剔除了所有复杂的分组，统一使用你基线实验的 1e-4 学习率
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    criterion_apa = nn.MSELoss()
    criterion_mdd = nn.CrossEntropyLoss()
    criterion_asr = nn.CrossEntropyLoss()
    
    # 因为统一了学习率且解冻了后6层，可以恢复基础的 1:1:1 权重，保证各任务起步公平
    w_apa, w_mdd, w_asr = 1.0, 1.0, 1.0

    best_val_loss = float('inf') 
    
    # ==========================================
    # 🏃 4. 训练大循环
    # ==========================================
    for epoch in range(epochs):
        model.train()
        train_loss, train_apa_loss, train_mdd_loss, train_asr_loss = 0, 0, 0, 0

        pbar_train = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch in pbar_train:
            input_values = batch["input_values"].to(device)
            target_pinyin_ids = batch["target_pinyin_ids"].to(device)
            actual_pinyin_ids = batch["actual_pinyin_ids"].to(device)
            mdd_labels = batch["mdd_labels"].to(device)
            targets_scores = batch["scores"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            optimizer.zero_grad()
            apa_scores, mdd_logits, asr_logits = model(input_values, target_pinyin_ids , attention_mask)

            loss_apa = criterion_apa(apa_scores, targets_scores)
            loss_mdd = criterion_mdd(mdd_logits, mdd_labels)
            loss_asr = criterion_asr(asr_logits, actual_pinyin_ids)
            
            loss = (w_apa * loss_apa) + (w_mdd * loss_mdd) + (w_asr * loss_asr)
            loss.backward()
            
            # 单一学习率 1e-4 对微调 wav2vec2 来说冲击可能较大，加一点基础梯度截断防崩
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            train_apa_loss += loss_apa.item()
            train_mdd_loss += loss_mdd.item()
            train_asr_loss += loss_asr.item()

            pbar_train.set_postfix({'L': f"{loss.item():.2f}"})

        num_train_batches = len(train_loader)
        avg_train_loss = train_loss / num_train_batches

        # ------------------------------------------
        # 验证阶段
        # ------------------------------------------
        model.eval()
        val_loss = 0
        apa_true_total, apa_pred_total = [], []
        mdd_true, mdd_pred = [], []
        asr_true, asr_pred = [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]  ", leave=False):
                input_values = batch["input_values"].to(device)
                target_pinyin_ids = batch["target_pinyin_ids"].to(device)
                actual_pinyin_ids = batch["actual_pinyin_ids"].to(device)
                mdd_labels = batch["mdd_labels"].to(device)
                targets_scores = batch["scores"].to(device)

                apa_scores, mdd_logits, asr_logits = model(input_values, target_pinyin_ids)

                loss_apa = criterion_apa(apa_scores, targets_scores)
                loss_mdd = criterion_mdd(mdd_logits, mdd_labels)
                loss_asr = criterion_asr(asr_logits, actual_pinyin_ids)
                
                v_loss = (w_apa * loss_apa) + (w_mdd * loss_mdd) + (w_asr * loss_asr)
                val_loss += v_loss.item()

                apa_true_total.extend(targets_scores[:, 0].cpu().numpy())
                apa_pred_total.extend(apa_scores[:, 0].cpu().numpy())
                
                mdd_true.extend(mdd_labels.cpu().numpy())
                mdd_pred.extend(torch.argmax(mdd_logits, dim=1).cpu().numpy())
                
                asr_true.extend(actual_pinyin_ids.cpu().numpy())
                asr_pred.extend(torch.argmax(asr_logits, dim=1).cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)

        try:
            val_pcc_total, _ = pearsonr(apa_true_total, apa_pred_total)
        except:
            val_pcc_total = 0.0
            
        val_mdd_f1 = f1_score(mdd_true, mdd_pred, average='macro', zero_division=0)
        val_asr_acc = accuracy_score(asr_true, asr_pred)

        print(f"\n✅ [Epoch {epoch+1} 总结]")
        print(f"   📉 Train Loss: {avg_train_loss:.4f} | APA: {train_apa_loss/num_train_batches:.2f} | MDD: {train_mdd_loss/num_train_batches:.2f} | ASR: {train_asr_loss/num_train_batches:.2f}")
        print(f"   📈 Val   Loss: {avg_val_loss:.4f}  | 🎯 APA PCC: {val_pcc_total:.4f} | 🎯 MDD F1: {val_mdd_f1:.4f} | 🎯 ASR Acc: {val_asr_acc:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = os.path.join(BASE_DIR, "mtl_model_best.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"   🌟 发现最佳模型！权重已保存至: mtl_model_best.pth\n")
        else:
            print("\n")

if __name__ == "__main__":
    train_and_validate_mtl()