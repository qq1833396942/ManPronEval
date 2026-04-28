import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import sys

# 让 Python 能找到上一级目录的 dataset.py
sys.path.append("..")
from dataset import build_pinyin_vocab, MDDDataset, collate_fn
from model2 import MDD_Whisper_Attention_Model

# ==========================================
# 1. 绝对路径配置 (Model 2 专属)
# ==========================================
BASE_DIR = r"C:\Users\14183\OneDrive\Desktop\MDD-whisper\exp_v2_attention"
LOCAL_WHISPER = r"C:\Users\14183\OneDrive\Desktop\MDD-whisper\whisper-base-local"
TRAIN_JSON = r"C:\Users\14183\OneDrive\Desktop\MDD-whisper\metadata_train.json"
VAL_JSON = r"C:\Users\14183\OneDrive\Desktop\MDD-whisper\metadata_val.json"
TEST_JSON = r"C:\Users\14183\OneDrive\Desktop\MDD-whisper\metadata_test.json"

PARAMS = {
    "batch_size": 16,  # 4070 Ti 12G 建议 16
    "lr": 1e-5,
    "epochs": 15,
    "patience": 3,
    "checkpoint_name": "checkpoint_v2.pth"
}


def evaluate(model, loader, device, criterion, desc="Eval"):
    model.eval()
    preds_list, trues_list, total_loss = [], [], 0.0
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, leave=False):
            logits = model(batch['waveforms'].to(device), batch['lengths'].to(device), batch['pinyin_ids'].to(device))
            labels = batch['labels'].to(device)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            preds_list.extend(preds.cpu().numpy())
            trues_list.extend(labels.cpu().numpy())
    return total_loss / len(loader), accuracy_score(trues_list, preds_list), f1_score(trues_list, preds_list,average='macro')


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 Model 2 炼丹炉启动！位置: {BASE_DIR}")

    # 2. 准备词表并保存实验 Config
    pinyin2id = build_pinyin_vocab([TRAIN_JSON, VAL_JSON, TEST_JSON])
    vocab_size = len(pinyin2id) + 1

    config_dict = {**PARAMS, "vocab_size": vocab_size, "model_type": "Attention_Refined"}
    with open(os.path.join(BASE_DIR, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=4)

    # 3. 数据加载
    train_loader = DataLoader(MDDDataset(TRAIN_JSON, pinyin2id=pinyin2id), batch_size=PARAMS["batch_size"],
                              shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(MDDDataset(VAL_JSON, pinyin2id=pinyin2id), batch_size=PARAMS["batch_size"], shuffle=False,
                            collate_fn=collate_fn)
    test_loader = DataLoader(MDDDataset(TEST_JSON, pinyin2id=pinyin2id), batch_size=PARAMS["batch_size"], shuffle=False,
                             collate_fn=collate_fn)

    # 4. 初始化
    model = MDD_Whisper_Attention_Model(num_pinyins=vocab_size, whisper_version=LOCAL_WHISPER).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=PARAMS["lr"])
    criterion = nn.CrossEntropyLoss()

    # 断点续训检查
    ckpt_path = os.path.join(BASE_DIR, PARAMS["checkpoint_name"])
    start_epoch, best_v_f1 = 0, 0.0
    if os.path.exists(ckpt_path):
        print("📦 正在恢复 Model 2 训练进度...")
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_v_f1 = ckpt['best_f1']

    # 5. 训练循环
    patience_counter = 0
    for epoch in range(start_epoch, PARAMS["epochs"]):
        print(f"\n━━━━━━━━ Epoch {epoch + 1}/{PARAMS['epochs']} ━━━━━━━━")
        model.train()
        t_loss = 0.0
        pbar = tqdm(train_loader, desc="[Training]")
        for batch in pbar:
            optimizer.zero_grad()
            logits = model(batch['waveforms'].to(device), batch['lengths'].to(device), batch['pinyin_ids'].to(device))
            loss = criterion(logits, batch['labels'].to(device))
            loss.backward()
            optimizer.step()
            t_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # 评测
        v_loss, v_acc, v_f1 = evaluate(model, val_loader, device, criterion, "[Val]")
        _, t_acc, t_f1 = evaluate(model, test_loader, device, criterion, "[Test]")
        print(f"📊 Epoch {epoch + 1} 总结: Val F1: {v_f1:.4f} | Test F1: {t_f1:.4f}")

        # 存档与保存最优
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_f1': best_v_f1
        }, ckpt_path)

        if v_f1 > best_v_f1:
            best_v_f1 = v_f1
            torch.save(model.state_dict(), os.path.join(BASE_DIR, "best_model_v2.pth"))
            print(f"⭐ 刷新 Model 2 最佳记录！")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PARAMS["patience"]:
                print("🛑 停止重跑机制触发。")
                break


if __name__ == "__main__":
    main()