import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import json
import sys

# 让 Python 能找到上一级目录的 dataset.py
sys.path.append("..")
from dataset import build_pinyin_vocab, MDDDataset, collate_fn
# 🚨 确保你的模型类保存在 model_whisper_lora.py 中
from model_whisper_lora import MDD_Whisper_Large_LoRA_Model

# ==========================================
# 1. 核心配置 (严格对齐 HuBERT 版)
# ==========================================
# 🚀 使用你提供的 Whisper-Large 本地绝对路径
LOCAL_WHISPER_PATH = r"whisper-large"

# 实验结果存放路径
BASE_DIR = r"whisperV3_lora"
os.makedirs(BASE_DIR, exist_ok=True)

TRAIN_JSON = r"../metadata_train.json"
VAL_JSON = r"../metadata_val.json"
TEST_JSON = r"../metadata_test.json"

PARAMS = {
    "learning_rate": 1e-4,  # 🚀 对齐 HuBERT LoRA 学习率
    "batch_size": 2,  # 🚀 对齐 HuBERT 显存策略
    "epochs": 15,
    "patience": 5,  # 容忍度
    "save_path": os.path.join(BASE_DIR, "best_whisper_large_lora.pth"),
    "checkpoint_name": os.path.join(BASE_DIR, "whisper_large_lora_checkpoint.pth")
}


def evaluate(model, loader, device, criterion, desc="Evaluating"):
    model.eval()
    preds_list, trues_list = [], []
    total_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, leave=False):
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
    print(f"🔥 Whisper-Large LoRA 炼丹炉点火！")
    print(f"📍 实验目录: {BASE_DIR}")

    # ==========================================
    # 2. 数据准备
    # ==========================================
    # 从三个 JSON 中构建完整词表，确保对齐
    pinyin2id = build_pinyin_vocab([TRAIN_JSON, VAL_JSON, TEST_JSON])
    vocab_size = len(pinyin2id) + 1

    print(f"📦 加载数据中... 词表大小: {vocab_size}")
    train_loader = DataLoader(MDDDataset(TRAIN_JSON, pinyin2id=pinyin2id),
                              batch_size=PARAMS["batch_size"], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(MDDDataset(VAL_JSON, pinyin2id=pinyin2id),
                            batch_size=PARAMS["batch_size"], shuffle=False, collate_fn=collate_fn)

    # ==========================================
    # 3. 模型与优化器 (提取 LoRA + Head 的参数)
    # ==========================================
    model = MDD_Whisper_Large_LoRA_Model(num_pinyins=vocab_size, whisper_version=LOCAL_WHISPER_PATH).to(device)

    # 🚀 关键：只提取 requires_grad=True 的参数（即 LoRA 权重和下游分类头）
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=PARAMS["learning_rate"])

    criterion = nn.CrossEntropyLoss()
    start_epoch, best_val_f1, epochs_no_improve = 0, 0.0, 0

    # 存档恢复逻辑
    if os.path.exists(PARAMS["checkpoint_name"]):
        print(f"📦 正在恢复存档: {PARAMS['checkpoint_name']}")
        checkpoint = torch.load(PARAMS["checkpoint_name"])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_f1 = checkpoint['best_f1']
        print(f"✅ 从第 {start_epoch + 1} 轮继续训练...")

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
                pinyin_ids=batch['pinyin_ids'].to(device)
            )

            loss = criterion(logits, batch['labels'].to(device))
            loss.backward()

            # 🚀 对齐 HuBERT：增加梯度裁剪，防止训练初期震荡
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item()
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        # 验证
        v_loss, v_acc, v_f1 = evaluate(model, val_loader, device, criterion, desc="[Validating]")

        print(f"\n📊 阶段成绩:")
        print(f"   [Train] Avg Loss: {train_loss / len(train_loader):.4f}")
        print(f"   [Val]   Loss: {v_loss:.4f} | Acc: {v_acc:.4f} | F1: {v_f1:.4f}")

        # 保存断点
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_f1': best_val_f1,
        }, PARAMS["checkpoint_name"])

        # 保存最优
        if v_f1 > best_val_f1:
            best_val_f1 = v_f1
            torch.save(model.state_dict(), PARAMS["save_path"])
            print(f"✨ 刷新纪录！最优模型已保存。")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"⚠️ 性能未提升，已连续 {epochs_no_improve} 轮停滞。")

        if epochs_no_improve >= PARAMS["patience"]:
            print(f"\n✋ 早停触发，训练结束。")
            break


if __name__ == "__main__":
    main()