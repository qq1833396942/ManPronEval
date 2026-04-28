import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from tqdm import tqdm

from dataset_wavlm import MDD_WavLM_Dataset, collate_fn_wavlm, build_pinyin_vocab
from model_wavlm_large_lora import MDD_WavLM_Model2


# =========================
# 配置
# =========================
WAVLM_PATH = r""
TRAIN_JSON = "metadata_train.json"
VAL_JSON = "metadata_val.json"

PARAMS = {
    "lr": 1e-4,
    "batch_size": 2,
    "epochs": 20,
    "save_path": "best_wavlm_large_lora_model.pth"
}


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            logits = model(
                batch['waveforms'].to(device),
                batch['lengths'].to(device),
                batch['pinyin_ids'].to(device)
            )
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch['labels'].cpu().numpy())

    return f1_score(all_labels, all_preds, average='macro')


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pinyin2id = build_pinyin_vocab([TRAIN_JSON, VAL_JSON])

    train_loader = DataLoader(
        MDD_WavLM_Dataset(TRAIN_JSON, pinyin2id),
        batch_size=PARAMS["batch_size"],
        shuffle=True,
        collate_fn=collate_fn_wavlm
    )

    val_loader = DataLoader(
        MDD_WavLM_Dataset(VAL_JSON, pinyin2id),
        batch_size=PARAMS["batch_size"],
        shuffle=False,
        collate_fn=collate_fn_wavlm
    )

    model = MDD_WavLM_Model2(len(pinyin2id), WAVLM_PATH).to(device)

    # 只优化可训练参数（LoRA + 任务头）
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=PARAMS["lr"])
    criterion = nn.CrossEntropyLoss()

    best_f1 = 0.0
    print("🚀 WavLM-Large + LoRA 炼丹炉启动！batch_size=8")

    for epoch in range(PARAMS["epochs"]):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{PARAMS['epochs']}")

        for batch in pbar:
            optimizer.zero_grad()

            logits = model(
                batch['waveforms'].to(device),
                batch['lengths'].to(device),
                batch['pinyin_ids'].to(device)
            )
            loss = criterion(logits, batch['labels'].to(device))

            if torch.isnan(loss):
                print("⚠️ Warning: NaN loss, skipping batch.")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        val_f1 = evaluate(model, val_loader, device)
        print(f"📊 Val F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), PARAMS["save_path"])
            print("⭐ 刷新纪录！")


if __name__ == "__main__":
    train()
