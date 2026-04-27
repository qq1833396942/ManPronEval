import os
import json
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, mean_absolute_error, confusion_matrix
from jiwer import cer, wer  # 💡 引入 jiwer 计算 ASR 高阶指标

# 导入你的数据类和 Large 模型类
from dataset_mtl import MTLDataset, mtl_collate_fn
from model_mtl_large import MultiTaskHubertLarge

def test_best_model_large():
    # ==========================================
    # ⚙️ 1. 配置区
    # ==========================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 正在启动 Large 模型最终测试，使用设备: {device}")

    # 替换为 Large 模型路径
    LOCAL_MODEL_PATH = r"facebook_wav2vec2_large_960"
    BASE_DIR = r""
    
    TEST_JSON_PATH = os.path.join(BASE_DIR, 'metadata_test_mtl.json') 
    VOCAB_PATH = os.path.join(BASE_DIR, 'syllable_vocab.json')
    # 🚨 指向 Large 模型的保存权重
    BEST_WEIGHTS = os.path.join(BASE_DIR, 'mtl_model_large_best.pth')
    
    # ==========================================
    # 📚 2. 加载词表与数据
    # ==========================================
    with open(VOCAB_PATH, 'r', encoding='utf-8') as f:
        pinyin2id = json.load(f)
    vocab_size = len(pinyin2id)

    print(f"🔍 正在加载测试集: {TEST_JSON_PATH}")
    test_dataset = MTLDataset(TEST_JSON_PATH, pinyin2id)
    # Large 模型显存压力大，Batch size 可以适当设小一点，比如 8 或 4
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=mtl_collate_fn, num_workers=4)

    # ==========================================
    # 🚀 3. 初始化模型并加载最佳权重
    # ==========================================
    model = MultiTaskHubertLarge(model_path=LOCAL_MODEL_PATH, vocab_size=vocab_size).to(device)
    
    if not os.path.exists(BEST_WEIGHTS):
        raise FileNotFoundError(f"找不到权重文件: {BEST_WEIGHTS}，请确认路径！")
        
    print("📦 正在加载最佳权重 (mtl_model_large_best.pth)...")
    model.load_state_dict(torch.load(BEST_WEIGHTS, map_location=device))
    model.eval() # 开启测试模式

    # ==========================================
    # 🏃 4. 开始测试推理
    # ==========================================
    apa_true_total, apa_pred_total = [], []
    apa_true_init, apa_pred_init = [], []
    apa_true_final, apa_pred_final = [], []
    mdd_true, mdd_pred = [], []
    asr_true, asr_pred = [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing Model"):
            input_values = batch["input_values"].to(device)
            target_pinyin_ids = batch["target_pinyin_ids"].to(device)
            actual_pinyin_ids = batch["actual_pinyin_ids"].to(device)
            mdd_labels = batch["mdd_labels"].to(device)
            targets_scores = batch["scores"].to(device)

            # ⚡ 开启自动混合精度，防止 OOM
            with torch.autocast(device_type='cuda'):
                apa_scores, mdd_logits, asr_logits = model(input_values, target_pinyin_ids)

            # 收集 APA
            apa_true_total.extend(targets_scores[:, 0].cpu().numpy())
            apa_true_init.extend(targets_scores[:, 1].cpu().numpy())
            apa_true_final.extend(targets_scores[:, 2].cpu().numpy())
            apa_pred_total.extend(apa_scores[:, 0].cpu().numpy())
            apa_pred_init.extend(apa_scores[:, 1].cpu().numpy())
            apa_pred_final.extend(apa_scores[:, 2].cpu().numpy())
            
            # 收集 MDD & ASR
            mdd_true.extend(mdd_labels.cpu().numpy())
            mdd_pred.extend(torch.argmax(mdd_logits, dim=1).cpu().numpy())
            asr_true.extend(actual_pinyin_ids.cpu().numpy())
            asr_pred.extend(torch.argmax(asr_logits, dim=1).cpu().numpy())

# ==========================================
    # 🧮 5. 计算并打印最终报告 (修复版 PER)
    # ==========================================
    print("\n" + "="*50)
    print("🏆 Large 多任务模型 (MTL) 最终测试集报告")
    print("="*50)

    # --- 📌 任务 1: APA ---
    pcc_tot, _ = pearsonr(apa_true_total, apa_pred_total)
    pcc_ini, _ = pearsonr(apa_true_init, apa_pred_init)
    pcc_fin, _ = pearsonr(apa_true_final, apa_pred_final)
    mse_tot = mean_squared_error(apa_true_total, apa_pred_total)
    mae_tot = mean_absolute_error(apa_true_total, apa_pred_total)
    
    print(f"📌 [任务 1: APA 自动打分]")
    print(f"   - 总分 PCC:   {pcc_tot:.4f}")
    print(f"   - 声母 PCC:   {pcc_ini:.4f}")
    print(f"   - 韵母 PCC:   {pcc_fin:.4f}")
    print(f"   - 总分 MSE:   {mse_tot:.4f}")
    print(f"   - 总分 MAE:   {mae_tot:.4f}")
    print("-" * 50)

    # --- 📌 任务 2: MDD ---
    mdd_acc = accuracy_score(mdd_true, mdd_pred) # 💡 新增 MDD Accuracy
    mdd_f1 = f1_score(mdd_true, mdd_pred, average='macro', zero_division=0)
    
    cm = confusion_matrix(mdd_true, mdd_pred)
    if cm.shape[0] > 1:
        false_acceptances = np.sum(cm[1:, 0]) # 实际>0(错)，预测=0(对)
        actual_errors = np.sum(cm[1:, :])     # 所有的实际错误
        far = false_acceptances / actual_errors if actual_errors > 0 else 0.0
    else:
        far = 0.0

    print(f"📌 [任务 2: MDD 发音诊断]")
    print(f"   - 准确率 (Accuracy):  {mdd_acc:.4f}") # 💡 新增打印
    print(f"   - 宏 F1 (Macro-F1):   {mdd_f1:.4f}")
    print(f"   - FAR (错误接受率):   {far:.4%}")
    print("-" * 50)

    # --- 📌 任务 3: ASR (修复真正的 PER) ---
    asr_acc = accuracy_score(asr_true, asr_pred)
    
    # 1. 建立反向词典 (ID -> 拼音)
    id2pinyin = {v: k for k, v in pinyin2id.items()}
    
    # 2. 定义声母表 (注意长度降序排列，防止 zh、ch、sh 被截断成 z、c、s)
    INITIALS = ['zh', 'ch', 'sh', 'b', 'p', 'm', 'f', 'd', 't', 'n', 'l', 'g', 'k', 'h', 'j', 'q', 'x', 'r', 'z', 'c', 's', 'y', 'w']
    
    def split_syllable(pinyin_str):
        """将拼音拆分为 声母 和 韵母"""
        for i in INITIALS:
            if pinyin_str.startswith(i):
                return i, pinyin_str[len(i):]
        # 零声母情况 (如 "an", "o")
        return "", pinyin_str

    true_phonemes_list = []
    pred_phonemes_list = []

    for t_id, p_id in zip(asr_true, asr_pred):
        # 获取拼音文本 (假设词表中有这个ID)
        t_pinyin = id2pinyin.get(t_id, "")
        p_pinyin = id2pinyin.get(p_id, "")
        
        # 拆分声母和韵母
        t_i, t_f = split_syllable(t_pinyin)
        p_i, p_f = split_syllable(p_pinyin)
        
        # 用空格拼接，让 jiwer 把声母和韵母当做两个独立的 token 计算编辑距离
        true_phonemes_list.append(f"{t_i} {t_f}".strip())
        pred_phonemes_list.append(f"{p_i} {p_f}".strip())

    # CER: 对于单音节，字符错误率其实就是音节错误率 (SER)，严格等于 1 - Acc
    asr_cer = 1.0 - asr_acc 
    
    # PER: 真正的音素错误率
    asr_per = wer(true_phonemes_list, pred_phonemes_list)

    print(f"📌 [任务 3: ASR 实际发音识别]")
    print(f"   - 准确率 (Accuracy):  {asr_acc:.4f}")
    print(f"   - CER (音节错误率):   {asr_cer:.4%}")
    print(f"   - PER (音素错误率):   {asr_per:.4%}")
    print("="*50 + "\n")

if __name__ == "__main__":
    test_best_model_large()