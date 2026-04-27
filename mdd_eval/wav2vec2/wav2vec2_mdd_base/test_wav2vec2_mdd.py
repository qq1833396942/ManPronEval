import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from tqdm import tqdm

# 导入你写好的数据处理和模型类
from dataset_wav2vec2 import build_pinyin_vocab, MDD_Wav2vec2_Dataset, collate_fn_wav2vec2
# 🚨 修改 1：引入严格对照版的模型类名
from model_wav2vec2 import MDD_WAV2VEC2_Attention_Model

# ==========================================
# 1. 核心配置
# ==========================================
LOCAL_WAV2VEC2_PATH = r"facebook_wav2vec2_base_960"
TEST_JSON = 'metadata_test_cleaned.json'  # 拿没见过的数据来测试
MODEL_WEIGHTS = "best_wav2vec2_mdd_model.pth" # 训练出的最优权重

BATCH_SIZE = 16

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 开始期末大考！当前设备: {device}")

    # 2. 加载词表和数据集
    # 注意：为了词表一致性，最好还是把 train/val/test 一起传进去建词表
    pinyin2id = build_pinyin_vocab(['metadata_train_cleaned.json', 'metadata_val_cleaned.json', TEST_JSON])
    vocab_size = len(pinyin2id) + 1

    test_dataset = MDD_Wav2vec2_Dataset(TEST_JSON, pinyin2id=pinyin2id)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_wav2vec2)
    print(f"✅ 测试集加载完毕，共 {len(test_dataset)} 条语音。")

    # 3. 初始化模型并加载权重
    print(f"📦 正在加载最优模型权重: {MODEL_WEIGHTS}")
    # 🚨 修改 2：使用新的模型类名实例化
    model = MDD_WAV2VEC2_Attention_Model(num_pinyins=vocab_size, freeze_encoder=False, wav2vec2_version=LOCAL_WAV2VEC2_PATH)
    
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
            # 🚨 修改 3：对齐“盲测模式”的输入参数
            logits = model(
                input_values=batch['waveforms'].to(device), 
                lengths=batch['lengths'].to(device),
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
    print("🏆 最终考试成绩单 🏆")
    print("="*40)
    # 修改 5: 将整体准确率和宏平均F1保留4位小数
    print(f"⭐ 整体准确率 (Accuracy) : {acc:.4f}") 
    print(f"⭐ 宏平均 F1 (Macro-F1) : {macro_f1:.4f}")
    
    print("\n📊 详细分类报告 (Classification Report):")
    target_names = ['0 (差)', '1 (中)', '2 (优)']
    # 修改 6: 强制 classification_report 中的所有指标保留 4 位小数 (digits=4)
    print(classification_report(trues_list, preds_list, target_names=target_names, digits=4))

    # ==========================================
    # 6. 绘制并保存混淆矩阵
    # ==========================================
    print("🎨 正在生成混淆矩阵热力图...")
    cm = confusion_matrix(trues_list, preds_list)
    
    # 🚨 修改 4：把解决中文显示问题的代码移到画图之前
    plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows 用黑体
    plt.rcParams['axes.unicode_minus'] = False
    
    plt.figure(figsize=(8, 6))
    # 使用 seaborn 画热力图，颜色越深代表数量越多
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names,
                annot_kws={"size": 14})
    
    plt.title('MDD Confusion Matrix (wav2vec2)', fontsize=16)
    plt.xlabel('Predicted Label (AI 判断)', fontsize=12)
    plt.ylabel('True Label (专家打分)', fontsize=12)
    
    save_path = "confusion_matrix_wav2vec2.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 混淆矩阵已保存为图片: {save_path}")

if __name__ == "__main__":
    main()