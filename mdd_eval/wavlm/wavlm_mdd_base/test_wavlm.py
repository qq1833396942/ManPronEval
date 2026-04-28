import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from dataset_wavlm import build_pinyin_vocab, MDD_WavLM_Dataset, collate_fn_wavlm
from model_wavlm import MDD_WavLM_Model2


# =========================
# 路径配置
# =========================
MODEL_PATH = r""
WAVLM_PATH = r""

TRAIN_JSON = "metadata_train.json"
VAL_JSON = "metadata_val.json"
TEST_JSON = "metadata_test.json"

BATCH_SIZE = 16


def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Testing"):
            waveforms = batch["waveforms"].to(device)
            lengths = batch["lengths"].to(device)
            pinyin_ids = batch["pinyin_ids"].to(device)
            labels = batch["labels"].to(device)

            logits = model(waveforms, lengths, pinyin_ids)
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    f1_per_class = f1_score(all_labels, all_preds, average=None, labels=[0, 1, 2])

    return acc, macro_f1, f1_per_class


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 测试设备: {device}")

    # 必须和训练时保持完全一致
    pinyin2id = build_pinyin_vocab([TRAIN_JSON, VAL_JSON])
    vocab_size = len(pinyin2id)

    print(f"📦 词表大小: {vocab_size}")

    test_loader = DataLoader(
        MDD_WavLM_Dataset(TEST_JSON, pinyin2id),
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn_wavlm
    )

    model = MDD_WavLM_Model2(
        num_pinyins=vocab_size,
        wavlm_path=WAVLM_PATH
    ).to(device)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print("✅ 模型加载完成")

    acc, macro_f1, f1_per_class = evaluate(model, test_loader, device)

    print("\n================= 📊 测试结果 =================")
    print(f"Accuracy      : {acc:.4f}")
    print(f"Macro F1      : {macro_f1:.4f}")
    print(f"F1 (label=0)  : {f1_per_class[0]:.4f}")
    print(f"F1 (label=1)  : {f1_per_class[1]:.4f}")
    print(f"F1 (label=2)  : {f1_per_class[2]:.4f}")
    print("===============================================")


if __name__ == "__main__":
    main()
