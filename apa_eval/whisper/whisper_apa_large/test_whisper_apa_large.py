import os
import torch
from torch.utils.data import DataLoader
import numpy as np
from scipy.stats import pearsonr
# 🚨 新增：导入 mean_absolute_error
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm

# 复用已有的 Dataset 和 Model
from dataset_whisper_apa import build_pinyin_vocab, APA_Whisper_Dataset,collate_fn_whisper_apa
from model_whisper_apa_large import APA_Whisper_Large_Model

# ==========================================
# 1. 路径与配置
# ==========================================
LOCAL_WHISPER_PATH = r""

TRAIN_JSON = 'metadata_train_apa.json'
VAL_JSON = 'metadata_val_apa.json'
TEST_JSON = 'metadata_test_apa.json'

# 指向你刚刚训练保存的最优模型权重
MODEL_WEIGHTS = "best_whisper_large_v2_explicit.pth"
BATCH_SIZE = 8  # 测试时不需要算梯度，显存占用较小，8绝对安全


def run_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔍 启动 Whisper-Large APA 测试引擎！当前设备: {device}")

    # 1. 重建词表 (必须用同样的 JSON 组合保证 id 完全一致)
    pinyin2id = build_pinyin_vocab([TRAIN_JSON, VAL_JSON, TEST_JSON])
    vocab_size = len(pinyin2id) + 1

    # 2. 加载 Test 数据集
    print("📦 正在加载测试集...")
    test_dataset = APA_Whisper_Dataset(TEST_JSON, pinyin2id=pinyin2id)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_whisper_apa)

    # 3. 初始化模型并加载权重
    model = APA_Whisper_Large_Model(num_pinyins=vocab_size, whisper_version=LOCAL_WHISPER_PATH).to(device)

    if not os.path.exists(MODEL_WEIGHTS):
        print(f"❌ 错误：未找到最优权重文件 {MODEL_WEIGHTS}")
        return

    # 安全加载权重
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
    model.eval()
    print("✅ 最优模型权重加载成功！开始推理...\n")

    # 4. 推理过程
    preds_ini, trues_ini = [], []
    preds_fin, trues_fin = [], []
    preds_tot, trues_tot = [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="[Testing]"):
            pred_ini, pred_fin, pred_tot = model(
                batch['waveforms'].to(device),
                batch['start_frames'].to(device),
                batch['end_frames'].to(device),
                batch['pinyin_ids'].to(device)
            )

            # 收集数据 (转为一维数组)
            preds_ini.extend(pred_ini.cpu().numpy().flatten())
            trues_ini.extend(batch['score_initial'].numpy().flatten())

            preds_fin.extend(pred_fin.cpu().numpy().flatten())
            trues_fin.extend(batch['score_final'].numpy().flatten())

            preds_tot.extend(pred_tot.cpu().numpy().flatten())
            trues_tot.extend(batch['score_total'].numpy().flatten())

    # ==========================================
    # 5. 还原 10 分制并计算指标
    # ==========================================
    preds_ini_10, trues_ini_10 = np.array(preds_ini) * 10.0, np.array(trues_ini) * 10.0
    preds_fin_10, trues_fin_10 = np.array(preds_fin) * 10.0, np.array(trues_fin) * 10.0
    preds_tot_10, trues_tot_10 = np.array(preds_tot) * 10.0, np.array(trues_tot) * 10.0

    # 计算 MSE
    mse_ini = mean_squared_error(trues_ini_10, preds_ini_10)
    mse_fin = mean_squared_error(trues_fin_10, preds_fin_10)
    mse_tot = mean_squared_error(trues_tot_10, preds_tot_10)

    # 🚨 新增：计算 MAE
    mae_ini = mean_absolute_error(trues_ini_10, preds_ini_10)
    mae_fin = mean_absolute_error(trues_fin_10, preds_fin_10)
    mae_tot = mean_absolute_error(trues_tot_10, preds_tot_10)

    def safe_pcc(trues, preds):
        return pearsonr(trues, preds)[0] if len(set(preds)) > 1 and len(set(trues)) > 1 else 0.0

    # 计算 PCC
    pcc_ini = safe_pcc(trues_ini_10, preds_ini_10)
    pcc_fin = safe_pcc(trues_fin_10, preds_fin_10)
    pcc_tot = safe_pcc(trues_tot_10, preds_tot_10)

    # 6. 打印终极成绩单
    print("\n" + "=" * 60)
    print("🏆 Whisper-Large 测试集最终性能报告 (Test Set)")
    print("=" * 60)
    # 🚨 打印结果中加入 MAE
    print(f"   ➤ 声母 (Initial) - MSE: {mse_ini:.4f} | MAE: {mae_ini:.4f} | PCC: {pcc_ini:.4f}")
    print(f"   ➤ 韵母 (Final)   - MSE: {mse_fin:.4f} | MAE: {mae_fin:.4f} | PCC: {pcc_fin:.4f}")
    print(f"   ➤ 总分 (Total)   - MSE: {mse_tot:.4f} | MAE: {mae_tot:.4f} | PCC: {pcc_tot:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    run_test()