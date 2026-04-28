import os
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

# 让 Python 能找到上一级目录的 dataset.py
sys.path.append("..")
from dataset import build_pinyin_vocab, MDDDataset, collate_fn
from model_whisper_lora import MDD_Whisper_Large_LoRA_Model

# ==========================================
# 1. 路径配置 (必须与训练时完全一致)
# ==========================================
LOCAL_WHISPER_PATH = r"whisper-large"
BASE_DIR = r"whisperV3_lora"

TRAIN_JSON = r"../metadata_train.json"
VAL_JSON = r"../metadata_val.json"
TEST_JSON = r"../metadata_test.json"

MODEL_WEIGHTS = os.path.join(BASE_DIR, "best_whisper_large_lora.pth")
BATCH_SIZE = 8  # 测试时如果显存够也可以调大一点

# 根据你的 3 分类标签定义修改 (0, 1, 2)
CLASS_NAMES = ["Correct (0)", "Substitution (1)", "Deletion/Others (2)"]


def run_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔍 启动 Whisper-Large LoRA 测试引擎... 设备: {device}")

    # 1. 重建词表 (必须用同样的 JSON 组合)
    pinyin2id = build_pinyin_vocab([TRAIN_JSON, VAL_JSON, TEST_JSON])
    vocab_size = len(pinyin2id) + 1

    # 2. 加载 Test 数据集
    print("📦 正在加载测试集...")
    test_dataset = MDDDataset(TEST_JSON, pinyin2id=pinyin2id)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # 3. 初始化模型并加载权重
    model = MDD_Whisper_Large_LoRA_Model(num_pinyins=vocab_size, whisper_version=LOCAL_WHISPER_PATH).to(device)

    if not os.path.exists(MODEL_WEIGHTS):
        print(f"❌ 错误：未找到最优权重文件 {MODEL_WEIGHTS}")
        return

    # ⚠️ 加上 map_location 确保加载安全
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
    model.eval()
    print("✅ 最优模型权重加载成功！开始推理...\n")

    # 4. 推理过程
    all_preds, all_trues = [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="[Testing]"):
            logits = model(
                batch['waveforms'].to(device),
                batch['lengths'].to(device),
                batch['pinyin_ids'].to(device)
            )
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_trues.extend(batch['labels'].cpu().numpy())

    # ==========================================
    # 5. 生成报告
    # ==========================================
    print("\n" + "=" * 50)
    print("🏆 测试集最终性能报告")
    print("=" * 50)

    acc = accuracy_score(all_trues, all_preds)
    f1_macro = f1_score(all_trues, all_preds, average='macro')

    print(f"🎯 Total Accuracy : {acc:.4f}")
    print(f"🎯 F1-score (Macro): {f1_macro:.4f}")
    print("-" * 50)

    print("📝 详细分类指标 (Precision / Recall / F1):")
    print(classification_report(all_trues, all_preds, target_names=CLASS_NAMES, digits=4))

    # ==========================================
    # 6. 绘制并保存混淆矩阵
    # ==========================================
    cm = confusion_matrix(all_trues, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix - Whisper Large LoRA')

    cm_path = os.path.join(BASE_DIR, "test_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    print(f"\n🖼️ 混淆矩阵已保存至: {cm_path}")
    plt.show()


if __name__ == "__main__":
    run_test()