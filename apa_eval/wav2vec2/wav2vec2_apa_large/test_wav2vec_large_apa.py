import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm
import pandas as pd

# 导入你的 Large 模型和数据处理类
from dataset_wav2vec2_apa import build_pinyin_vocab, APA_Wav2Vec2_Dataset, collate_fn_apa
from mdd_eval.wav2vec2.wav2vec2_mdd_large.test_large_mdd import LOCAL_WAV2VEC2_PATH
from model_wav2vec2_apa_large import APA_Wav2Vec2_Large_Model

# ==========================================
# 1. 核心配置
# ==========================================
LOCAL_WAV2VEC2_PATH = r"facebook_wav2vec2_large_960"
TEST_JSON = 'metadata_test_apa.json' 
MODEL_WEIGHTS = "best_wav2vec2_large_v2_explicit.pth" # 确保这是你刚刚练出来的包含三头打分的最新权重

BATCH_SIZE = 16

def plot_regression(trues, preds, title, save_path):
    """绘制回归散点图"""
    plt.figure(figsize=(8, 6))
    sns.regplot(x=trues, y=preds, scatter_kws={'alpha':0.3, 's':10}, line_kws={'color':'red'})
    plt.xlabel('True Score (Expert)', fontsize=12)
    plt.ylabel('Predicted Score (AI)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def safe_pcc(trues, preds):
    return pearsonr(trues, preds)[0] if len(set(preds)) > 1 and len(set(trues)) > 1 else 0.0

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 开始 APA Large 版本多维度终极测试！设备: {device}")

    # 2. 加载词表和数据集
    pinyin2id = build_pinyin_vocab(['metadata_train_apa.json', 'metadata_val_apa.json', TEST_JSON])
    vocab_size = len(pinyin2id) + 1

    test_dataset = APA_Wav2Vec2_Dataset(TEST_JSON, pinyin2id=pinyin2id)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_apa)
    print(f"✅ 测试集加载完毕，共 {len(test_dataset)} 条数据。")

    # 3. 初始化模型并加载权重
    model = APA_Wav2Vec2_Large_Model(num_pinyins=vocab_size, wav2vec2_version=LOCAL_WAV2VEC2_PATH)
    
    if not os.path.exists(MODEL_WEIGHTS):
        print(f"❌ 找不到权重文件 {MODEL_WEIGHTS}！")
        return
        
    checkpoint = torch.load(MODEL_WEIGHTS, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("📦 成功从存档点加载权重。")
    else:
        model.load_state_dict(checkpoint, strict=True) 
        print("📦 成功加载纯权重文件。")
        
    model.to(device)
    model.eval()

    # 4. 准备存储 3 组分数的列表
    preds_ini, trues_ini = [], []
    preds_fin, trues_fin = [], []
    preds_tot, trues_tot = [], []

    print("\n🎧 AI 考官正在对测试集进行多维度打分...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="[Testing]"):
            # 获取模型的 3 个输出 (0~1 之间)
            p_ini, p_fin, p_tot = model(
                batch['waveforms'].to(device), 
                batch['start_frames'].to(device),
                batch['end_frames'].to(device),
                batch['pinyin_ids'].to(device)
            )
            
            # 获取真实的 3 个标签 (Dataset 里已经除以 10 了)
            t_ini = batch['score_initial'] 
            t_fin = batch['score_final'] 
            t_tot = batch['score_total'] 
            
            # 存入列表
            preds_ini.extend(p_ini.cpu().numpy().flatten())
            trues_ini.extend(t_ini.cpu().numpy().flatten())
            
            preds_fin.extend(p_fin.cpu().numpy().flatten())
            trues_fin.extend(t_fin.cpu().numpy().flatten())
            
            preds_tot.extend(p_tot.cpu().numpy().flatten())
            trues_tot.extend(t_tot.cpu().numpy().flatten())

    # 5. 指标计算 (全部还原到 10 分制)
    p_ini_10, t_ini_10 = np.array(preds_ini) * 10.0, np.array(trues_ini) * 10.0
    p_fin_10, t_fin_10 = np.array(preds_fin) * 10.0, np.array(trues_fin) * 10.0
    p_tot_10, t_tot_10 = np.array(preds_tot) * 10.0, np.array(trues_tot) * 10.0

    # 计算各项指标
    metrics = {
        "Initial (声母)": {
            "PCC": safe_pcc(t_ini_10, p_ini_10),
            "MSE": mean_squared_error(t_ini_10, p_ini_10),
            "MAE": mean_absolute_error(t_ini_10, p_ini_10)
        },
        "Final (韵母)": {
            "PCC": safe_pcc(t_fin_10, p_fin_10),
            "MSE": mean_squared_error(t_fin_10, p_fin_10),
            "MAE": mean_absolute_error(t_fin_10, p_fin_10)
        },
        "Total (总分)": {
            "PCC": safe_pcc(t_tot_10, p_tot_10),
            "MSE": mean_squared_error(t_tot_10, p_tot_10),
            "MAE": mean_absolute_error(t_tot_10, p_tot_10)
        }
    }

    # 6. 华丽地打印成绩单
    print("\n" + "="*50)
    print("🏆 APA Large 多维度测试集最终成绩单 🏆")
    print("="*50)
    for name, m in metrics.items():
        print(f"📌 【{name}】")
        print(f"   ⭐ PCC (相关性): {m['PCC']:.4f}")
        print(f"   ⭐ MSE (均方误差): {m['MSE']:.4f}")
        print(f"   ⭐ MAE (绝对误差): {m['MAE']:.4f} 分")
        print("-" * 50)

    # 7. 可视化 (画 3 张图)
    print("🎨 正在生成 3 张打分回归分析图...")
    plot_regression(t_ini_10, p_ini_10, f'Initial Score (PCC={metrics["Initial (声母)"]["PCC"]:.3f})', 'apa_test_regression_initial.png')
    plot_regression(t_fin_10, p_fin_10, f'Final Score (PCC={metrics["Final (韵母)"]["PCC"]:.3f})', 'apa_test_regression_final.png')
    plot_regression(t_tot_10, p_tot_10, f'Total Score (PCC={metrics["Total (总分)"]["PCC"]:.3f})', 'apa_test_regression_total.png')
    
    # 8. 保存极其详细的预测结果
    results_df = pd.DataFrame({
        'True_Initial': t_ini_10,  'Pred_Initial': p_ini_10,  'Error_Initial': np.abs(t_ini_10 - p_ini_10),
        'True_Final': t_fin_10,    'Pred_Final': p_fin_10,    'Error_Final': np.abs(t_fin_10 - p_fin_10),
        'True_Total': t_tot_10,    'Pred_Total': p_tot_10,    'Error_Total': np.abs(t_tot_10 - p_tot_10)
    })
    results_df.to_csv("test_multiscore_results_detail.csv", index=False)
    print(f"✅ 详细预测对比已保存至: test_multiscore_results_detail.csv")

if __name__ == "__main__":
    main()