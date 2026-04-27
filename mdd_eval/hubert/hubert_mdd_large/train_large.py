import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import json

from dataset_hubert import build_pinyin_vocab, MDD_HuBERT_Dataset, collate_fn_hubert
# 🚨 导入新的 Large + LoRA 模型
from model_large import MDD_HuBERT_Large_LoRA_Model

# ==========================================
# 1. 核心配置 
# ==========================================
# 🚀 替换为 Large 版本的本地路径
LOCAL_HUBERT_PATH = r"facebook/hubert-large-ls960-ft"

TRAIN_JSON = 'metadata_train_cleaned.json'
VAL_JSON   = 'metadata_val_cleaned.json'
TEST_JSON  = 'metadata_test_cleaned.json' 

PARAMS = {
    "learning_rate": 1e-4,  # 🚀 统一使用一个学习率
    "batch_size": 2,        # 🚀 降低 Batch Size 防止 Large 模型显存溢出
    "epochs": 20,
    "patience": 3,
    "save_path": "best_hubert_large_lora_mdd.pth",
    "checkpoint_name": "hubert_large_lora_checkpoint.pth"
}

def evaluate(model, loader, device, criterion, desc="Evaluating"):
    model.eval()
    preds_list, trues_list = [], []
    total_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            logits = model(
                batch['waveforms'].to(device), 
                batch['lengths'].to(device),
                batch['pinyin_ids'].to(device)
            )
            labels = batch['labels'].to(device)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            preds_list.extend(preds.cpu().numpy())
            trues_list.extend(labels.cpu().numpy())

    acc = accuracy_score(trues_list, preds_list)
    f1 = f1_score(trues_list, preds_list, average='macro')
    return total_loss / len(loader), acc, f1

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 炼丹炉点火 (Large+LoRA)！当前设备: {device}")

    # ==========================================
    # 2. 数据流
    # ==========================================
    pinyin2id = build_pinyin_vocab([TRAIN_JSON, VAL_JSON, TEST_JSON])
    vocab_size = len(pinyin2id) + 1

    print(f"📦 正在加载数据... 词表大小: {vocab_size}")
    train_loader = DataLoader(MDD_HuBERT_Dataset(TRAIN_JSON, pinyin2id=pinyin2id), 
                              batch_size=PARAMS["batch_size"], shuffle=True, collate_fn=collate_fn_hubert)
    val_loader   = DataLoader(MDD_HuBERT_Dataset(VAL_JSON, pinyin2id=pinyin2id), 
                              batch_size=PARAMS["batch_size"], shuffle=False, collate_fn=collate_fn_hubert)
    
    # ==========================================
    # 3. 初始化与优化器配置
    # ==========================================
    model = MDD_HuBERT_Large_LoRA_Model(num_pinyins=vocab_size, hubert_version=LOCAL_HUBERT_PATH).to(device)
    
    # 🚀 核心修改：去掉分层逻辑，直接提取所有可训练参数统一设置 1e-4
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=PARAMS["learning_rate"])
    
    criterion = nn.CrossEntropyLoss()
    start_epoch, best_val_f1, epochs_no_improve = 0, 0.0, 0

    if os.path.exists(PARAMS["checkpoint_name"]):
        print(f"📦 恢复存档进度: {PARAMS['checkpoint_name']}")
        checkpoint = torch.load(PARAMS["checkpoint_name"])
        model.load_state_dict(checkpoint['model_state_dict'])
        #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_f1 = checkpoint['best_f1']
        print(f"✅ 从第 {start_epoch} 轮继续...")

    # ==========================================
    # 4. 训练大循环
    # ==========================================
    for epoch in range(start_epoch, PARAMS["epochs"]):
        print(f"\n━━━━━━━━ Epoch {epoch + 1}/{PARAMS['epochs']} ━━━━━━━━")
        model.train()
        train_loss = 0.0
        train_bar = tqdm(train_loader, desc="[Training]")
        
        for batch in train_bar:
            optimizer.zero_grad()
            logits = model(
                batch['waveforms'].to(device), 
                batch['lengths'].to(device),
                target_syllable_ids=batch['pinyin_ids'].to(device) 
            )
            loss = criterion(logits, batch['labels'].to(device))
            loss.backward()
            
            # 增加梯度裁剪，防止 LoRA 初期跑飞
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        v_loss, v_acc, v_f1 = evaluate(model, val_loader, device, criterion, desc="[Validating]")
        
        print(f"\n📊 成绩单:")
        print(f"   [Train] Loss: {train_loss / len(train_loader):.4f}")
        print(f"   [Val]   Loss: {v_loss:.4f} | Acc: {v_acc:.4f} | F1: {v_f1:.4f}")

        # 保存断点
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_f1': best_val_f1,
        }, PARAMS["checkpoint_name"])

        # 保存最优模型
        if v_f1 > best_val_f1:
            best_val_f1 = v_f1
            torch.save(model.state_dict(), PARAMS["save_path"])
            print(f"✨ 新纪录！已保存最优模型 {PARAMS['save_path']}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"⚠️ F1 停滞，连续 {epochs_no_improve} 轮无提升。")

        if epochs_no_improve >= PARAMS["patience"] + 5:
            print(f"\n✋ 早停触发！训练结束。")
            break

if __name__ == "__main__":
    main()