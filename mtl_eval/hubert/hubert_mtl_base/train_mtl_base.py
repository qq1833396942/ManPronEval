import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, f1_score

from dataset_mtl import MTLDataset,  mtl_collate_fn
from model_mtl_base import MultiTaskHubert

# 🧮 辅助函数：计算两个序列的编辑距离 (用于计算音素级 PER)
def get_edit_distance(seq1, seq2):
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
    return dp[m][n]

def train_and_validate_mtl():
    # ==========================================
    # ⚙️ 1. 配置区
    # ==========================================
    LOCAL_MODEL_PATH = r"C:\Users\Y9000P\.cache\huggingface\hub\models--facebook--hubert-base-ls960\snapshots\dba3bb02fda4248b6e082697eee756de8fe8aa8a"
    BASE_DIR = r"D:\develop\HuBERT_fun"
    
    VOCAB_PATH = os.path.join(BASE_DIR, 'syllable_vocab.json')
    TRAIN_JSON_PATH = os.path.join(BASE_DIR, 'metadata_train_mtl.json')
    VAL_JSON_PATH = os.path.join(BASE_DIR, 'metadata_val_mtl.json') 
    
    epochs = 30
    batch_size = 16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 计算设备: {device} | 严格单学习率模式 (1e-4)")

    # ==========================================
    # 📚 2. 加载数据
    # ==========================================
    with open(VOCAB_PATH, 'r', encoding='utf-8') as f:
        pinyin2id = json.load(f)
    vocab_size = len(pinyin2id)
    
    # 🌟 建立 id 到 pinyin 的反向映射，用于验证时还原音素计算 PER
    id2pinyin = {v: k for k, v in pinyin2id.items()}

    train_dataset = MTLDataset(TRAIN_JSON_PATH, pinyin2id)
    val_dataset = MTLDataset(VAL_JSON_PATH, pinyin2id)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=mtl_collate_fn, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=mtl_collate_fn, num_workers=4)

    # ==========================================
    # 🚀 3. 模型与优化器
    # ==========================================
    model = MultiTaskHubert(model_path=LOCAL_MODEL_PATH, vocab_size=vocab_size).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    criterion_apa = nn.MSELoss()
    criterion_mdd = nn.CrossEntropyLoss()
    criterion_asr = nn.CrossEntropyLoss()
    
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

            optimizer.zero_grad()
            apa_scores, mdd_logits, asr_logits = model(input_values, target_pinyin_ids)

            loss_apa = criterion_apa(apa_scores, targets_scores)
            loss_mdd = criterion_mdd(mdd_logits, mdd_labels)
            loss_asr = criterion_asr(asr_logits, actual_pinyin_ids)
            
            loss = (w_apa * loss_apa) + (w_mdd * loss_mdd) + (w_asr * loss_asr)
            loss.backward()
            
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

        # 🌟 计算 ASR PER (Phoneme Error Rate) 🌟
        total_edit_distance = 0
        total_phonemes = 0
        for t, p in zip(asr_true, asr_pred):
            true_pinyin = id2pinyin.get(t, "")
            pred_pinyin = id2pinyin.get(p, "")
            
            # 智能拆分音素：如果词表中带下划线/空格则按符号切分，否则直接转化为字符列表（满足声母韵母的细粒度验证）
            seq_t = true_pinyin.split('_') if '_' in true_pinyin else (true_pinyin.split() if ' ' in true_pinyin else list(true_pinyin))
            seq_p = pred_pinyin.split('_') if '_' in pred_pinyin else (pred_pinyin.split() if ' ' in pred_pinyin else list(pred_pinyin))
            
            total_edit_distance += get_edit_distance(seq_t, seq_p)
            total_phonemes += len(seq_t)
            
        val_asr_per = total_edit_distance / total_phonemes if total_phonemes > 0 else 0.0

        print(f"\n✅ [Epoch {epoch+1} 总结]")
        print(f"   📉 Train Loss: {avg_train_loss:.4f} | APA: {train_apa_loss/num_train_batches:.2f} | MDD: {train_mdd_loss/num_train_batches:.2f} | ASR: {train_asr_loss/num_train_batches:.2f}")
        print(f"   📈 Val   Loss: {avg_val_loss:.4f}  | 🎯 APA PCC: {val_pcc_total:.4f} | 🎯 MDD F1: {val_mdd_f1:.4f} | 🎯 ASR Acc: {val_asr_acc:.4f} | 🎯 ASR PER: {val_asr_per:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = os.path.join(BASE_DIR, "mtl_model_best_final.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"   🌟 发现最佳模型！权重已保存至: mtl_model_best_final.pth\n")
        else:
            print("\n")

if __name__ == "__main__":
    train_and_validate_mtl()