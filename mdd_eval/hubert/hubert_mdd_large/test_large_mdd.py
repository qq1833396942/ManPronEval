import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from tqdm import tqdm

from dataset_hubert import build_pinyin_vocab, MDD_HuBERT_Dataset, collate_fn_hubert
# 🚨 修改 1：引入 Large + LoRA 版的模型类名 (假设你的文件叫 model_large.py)
from model_large import MDD_HuBERT_Large_LoRA_Model

# ==========================================
# 1. 核心配置
# ==========================================
# 🚨 修改 2：指向 Large 版本的本地路径
LOCAL_HUBERT_PATH = r"facebook/hubert-large-ls960-ft"
TEST_JSON = 'metadata_test_cleaned.json'  

# 🚨 修改 3：指向你刚刚训练出来的 Large 版最优权重
MODEL_WEIGHTS = "best_hubert_large_lora_mdd.pth" 

# Large 版模型较大，推理时显存压力比训练小，但为了安全依然建议设为 8 或 16
BATCH_SIZE = 8

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 开始期末大考 (Large + LoRA 版)！当前设备: {device}")

    # 2. 加载词表和数据集
    pinyin2id = build_pinyin_vocab(['metadata_train_cleaned.json', 'metadata_val_cleaned.json', TEST_JSON])
    vocab_size = len(pinyin2id) + 1

    test_dataset = MDD_HuBERT_Dataset(TEST_JSON, pinyin2id=pinyin2id)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_hubert)
    print(f"✅ 测试集加载完毕，共 {len(test_dataset)} 条语音。")

    # 3. 初始化模型并加载权重
    print(f"📦 正在加载最优模型权重: {MODEL_WEIGHTS}")
    
    # 🚨 修改 4：使用 Large 模型类实例化 (注意参数匹配)
    model = MDD_HuBERT_Large_LoRA_Model(
        num_pinyins=vocab_size, 
        hubert_version=LOCAL_HUBERT_PATH
    )
    
    if not os.path.exists(MODEL_WEIGHTS):
        print(f"❌ 找不到权重文件 {MODEL_WEIGHTS}！请先完成训练。")
        return
        
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
    model.to(device)
    model.eval()

    # 4. 开始推理
    preds_list = []
    trues_list = []

    print("\n🎧 模型正在听音判卷中...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="[Testing]"):
            # 保持对齐：传入 waveforms, lengths, target_syllable_ids
            logits = model(
                batch['waveforms'].to(device),
                batch['lengths'].to(device),
                target_syllable_ids=batch['pinyin_ids'].to(device)
            )
            labels = batch['labels'].to(device)
            
            # 获取预测的类别 (0, 1, 2)
            preds = torch.argmax(logits, dim=-1)
            
            preds_list.extend(preds.cpu().numpy())
            trues_list.extend(labels.cpu().numpy())

    # ==========================================
    # 5. 计算指标与输出报告
    # ==========================================
    acc = accuracy_score(trues_list, preds_list)
    macro_f1 = f1_score(trues_list, preds_list, average='macro')
    
    print("\n" + "="*40)
    print("🏆 最终考试成绩单 (Large 版) 🏆")
    print("="*40)
    print(f"⭐ 整体准确率 (Accuracy) : {acc:.4f}") 
    print(f"⭐ 宏平均 F1 (Macro-F1) : {macro_f1:.4f}")
    
    print("\n📊 详细分类报告 (Classification Report):")
    target_names = ['0 (差)', '1 (中)', '2 (优)']
    print(classification_report(trues_list, preds_list, target_names=target_names, digits=4))

    # ==========================================
    # 6. 绘制并保存混淆矩阵
    # ==========================================
    print("🎨 正在生成混淆矩阵热力图...")
    cm = confusion_matrix(trues_list, preds_list)
    
    plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows 用黑体
    plt.rcParams['axes.unicode_minus'] = False
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names,
                annot_kws={"size": 14})
    
    plt.title('MDD Confusion Matrix (HuBERT Large + LoRA)', fontsize=16)
    plt.xlabel('Predicted Label (AI 判断)', fontsize=12)
    plt.ylabel('True Label (专家打分)', fontsize=12)
    
    # 🚨 修改 5：修改图片保存名称，防止覆盖 Base 版本的图
    save_path = "confusion_matrix_hubert_large.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 混淆矩阵已保存为图片: {save_path}")

if __name__ == "__main__":
    main()