import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import pandas as pd

# 💡 复用 HuBERT 的数据集读取逻辑：
# 因为 Whisper 模型内部 (forward) 已经集成了波形转 Mel 频谱的操作
from dataset_whisper_apa import build_pinyin_vocab, APA_Whisper_Dataset,collate_fn_whisper_apa

# 🚨 导入刚刚写好的 Whisper-Large 考官模型
from model_whisper_apa_large import APA_Whisper_Large_Model

# ==========================================
# 1. 核心配置 (严格对齐)
# ==========================================
# 🚀 你的本地 Whisper-Large 路径
LOCAL_WHISPER_PATH = r""

TRAIN_JSON = 'metadata_train_apa.json'
VAL_JSON = 'metadata_val_apa.json'
TEST_JSON = 'metadata_test_apa.json'

PARAMS = {
    "learning_rate": 2e-5,  # 🎯 严格对齐 HuBERT-Large
    "batch_size": 4,  # ⚠️ 注意：如果 Whisper-Large 报 OOM(显存溢出)，请将此处改为 8 或 4
    "epochs": 30,
    "patience": 5,
    # ⚠️ 更改保存路径，防止覆盖 HuBERT 的实验结果
    "save_path": "best_whisper_large_v2_explicit.pth",
    "checkpoint_name": "whisper_large_apa_v2_checkpoint.pth"
}


def evaluate(model, loader, device, criterion, desc="Evaluating"):
    model.eval()

    preds_ini, trues_ini = [], []
    preds_fin, trues_fin = [], []
    preds_tot, trues_tot = [], []
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            pred_ini, pred_fin, pred_tot = model(
                batch['waveforms'].to(device),
                batch['start_frames'].to(device),
                batch['end_frames'].to(device),
                batch['pinyin_ids'].to(device)
            )

            true_ini = batch['score_initial'].to(device)
            true_fin = batch['score_final'].to(device)
            true_tot = batch['score_total'].to(device)

            loss = criterion(pred_ini, true_ini) + criterion(pred_fin, true_fin) + criterion(pred_tot, true_tot)
            total_loss += loss.item()

            preds_ini.extend(pred_ini.cpu().numpy().flatten())
            trues_ini.extend(true_ini.cpu().numpy().flatten())

            preds_fin.extend(pred_fin.cpu().numpy().flatten())
            trues_fin.extend(true_fin.cpu().numpy().flatten())

            preds_tot.extend(pred_tot.cpu().numpy().flatten())
            trues_tot.extend(true_tot.cpu().numpy().flatten())

    avg_loss = total_loss / len(loader)

    preds_ini_10, trues_ini_10 = np.array(preds_ini) * 10.0, np.array(trues_ini) * 10.0
    preds_fin_10, trues_fin_10 = np.array(preds_fin) * 10.0, np.array(trues_fin) * 10.0
    preds_tot_10, trues_tot_10 = np.array(preds_tot) * 10.0, np.array(trues_tot) * 10.0

    mse_ini = mean_squared_error(trues_ini_10, preds_ini_10)
    mse_fin = mean_squared_error(trues_fin_10, preds_fin_10)
    mse_tot = mean_squared_error(trues_tot_10, preds_tot_10)

    def safe_pcc(trues, preds):
        return pearsonr(trues, preds)[0] if len(set(preds)) > 1 and len(set(trues)) > 1 else 0.0

    pcc_ini = safe_pcc(trues_ini_10, preds_ini_10)
    pcc_fin = safe_pcc(trues_fin_10, preds_fin_10)
    pcc_tot = safe_pcc(trues_tot_10, preds_tot_10)

    return avg_loss, (mse_ini, mse_fin, mse_tot), (pcc_ini, pcc_fin, pcc_tot)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Whisper APA Large V2 架构启动 (显式交互版 + 严格对齐)！当前设备: {device}")

    # 1. 数据准备
    pinyin2id = build_pinyin_vocab([TRAIN_JSON, VAL_JSON, TEST_JSON])
    vocab_size = len(pinyin2id) + 1

    train_loader = DataLoader(APA_Whisper_Dataset(TRAIN_JSON, pinyin2id=pinyin2id),
                              batch_size=PARAMS["batch_size"], shuffle=True, collate_fn=collate_fn_whisper_apa)
    val_loader = DataLoader(APA_Whisper_Dataset(VAL_JSON, pinyin2id=pinyin2id),
                            batch_size=PARAMS["batch_size"], shuffle=False, collate_fn=collate_fn_whisper_apa)

    # 2. 模型初始化 (调用 Whisper-Large 版)
    model = APA_Whisper_Large_Model(num_pinyins=vocab_size, whisper_version=LOCAL_WHISPER_PATH).to(device)

    # 完美对齐：只提取 requires_grad=True 的参数（MLP 和 Attention）
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.AdamW(trainable_params, lr=PARAMS["learning_rate"])

    criterion = nn.MSELoss()

    best_pcc = -1.0
    epochs_no_improve = 0
    history = []

    if os.path.exists(PARAMS["checkpoint_name"]):
        print(f"📦 恢复存档进度: {PARAMS['checkpoint_name']}")
        checkpoint = torch.load(PARAMS["checkpoint_name"])
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_pcc = checkpoint['best_pcc']
        print(f"✅ 从第 {start_epoch} 轮继续...")
    else:
        print(f"🌟 检测到新架构，从 Epoch 0 开始纯净训练！")
        start_epoch = 0

    for epoch in range(start_epoch, PARAMS["epochs"]):
        print(f"\n━━━━━━━━ Epoch {epoch + 1}/{PARAMS['epochs']} ━━━━━━━━")
        model.train()
        train_loss = 0.0
        train_bar = tqdm(train_loader, desc="[Training]")

        step = 0
        for batch in train_bar:
            if step == 0 and epoch == 0:
                print(f"DEBUG - Label Sample (Total): {batch['score_total'][0].item()}")

            optimizer.zero_grad()
            step += 1

            pred_ini, pred_fin, pred_tot = model(
                batch['waveforms'].to(device),
                batch['start_frames'].to(device),
                batch['end_frames'].to(device),
                batch['pinyin_ids'].to(device)
            )

            loss = criterion(pred_ini, batch['score_initial'].to(device)) + \
                   criterion(pred_fin, batch['score_final'].to(device)) + \
                   criterion(pred_tot, batch['score_total'].to(device))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        # 验证
        v_loss, v_mses, v_pccs = evaluate(model, val_loader, device, criterion, desc="[Validating]")

        mse_ini, mse_fin, mse_tot = v_mses
        pcc_ini, pcc_fin, pcc_tot = v_pccs

        print(f"📊 成绩单: [Train 内部 MSE Loss: {train_loss / len(train_loader):.4f}]")
        print(f"   ➤ 声母 (Initial) - MSE: {mse_ini:.4f} | PCC: {pcc_ini:.4f}")
        print(f"   ➤ 韵母 (Final)   - MSE: {mse_fin:.4f} | PCC: {pcc_fin:.4f}")
        print(f"   ➤ 总分 (Total)   - MSE: {mse_tot:.4f} | PCC: {pcc_tot:.4f}")

        # ⚠️ 日志文件名做了区分，生成专属于 Whisper-Large 的 CSV
        history.append({
            "Epoch": epoch + 1,
            "MSE_Ini": round(mse_ini, 4), "PCC_Ini": round(pcc_ini, 4),
            "MSE_Fin": round(mse_fin, 4), "PCC_Fin": round(pcc_fin, 4),
            "MSE_Tot": round(mse_tot, 4), "PCC_Tot": round(pcc_tot, 4)
        })
        pd.DataFrame(history).to_csv("training_log_whisper_apa_large_v2.csv", index=False)

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'best_pcc': best_pcc,
        }, PARAMS["checkpoint_name"])

        if pcc_tot > best_pcc:
            best_pcc = pcc_tot
            torch.save(model.state_dict(), PARAMS["save_path"])
            print(f"✨ 突破新纪录！最优核心总分 PCC 达到 {best_pcc:.4f}，模型已保存。")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"⚠️ 核心总分 PCC 停滞 {epochs_no_improve} 轮。")

        if epochs_no_improve >= PARAMS["patience"]:
            print(f"\n✋ 触发早停机制，训练圆满结束！")
            break


if __name__ == "__main__":
    main()