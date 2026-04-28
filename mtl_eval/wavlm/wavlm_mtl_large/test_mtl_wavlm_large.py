import os
import json
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, mean_absolute_error

from dataset_mtl_wavlm import MTLDataset, mtl_collate_fn
from model_mtl_wavlm_large import MultiTaskWavLMLarge


def get_edit_distance(seq1, seq2):
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
    return dp[m][n]


def pinyin_split(pinyin_str):
    if not pinyin_str:
        return []
    initials = [
        'zh', 'ch', 'sh', 'b', 'p', 'm', 'f', 'd', 't', 'n', 'l',
        'g', 'k', 'h', 'j', 'q', 'x', 'r', 'z', 'c', 's', 'y', 'w'
    ]
    res, tone = [], ""
    if pinyin_str[-1].isdigit():
        tone = pinyin_str[-1]
        pinyin_str = pinyin_str[:-1]
    for length in [2, 1]:
        if len(pinyin_str) >= length and pinyin_str[:length] in initials:
            res.append(pinyin_str[:length])
            pinyin_str = pinyin_str[length:]
            break
    if pinyin_str:
        res.append(pinyin_str)
    if tone:
        res.append(tone)
    return res


def test_best_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CODE_DIR = os.path.dirname(os.path.abspath(__file__))
    LOCAL_MODEL_PATH = r""

    TEST_JSON_PATH = os.path.join(CODE_DIR, "metadata_test_mtl.json")
    VOCAB_PATH = os.path.join(CODE_DIR, "syllable_vocab.json")
    BEST_WEIGHTS = os.path.join(CODE_DIR, "mtl_wavlm_large_best.pth")

    with open(VOCAB_PATH, 'r', encoding='utf-8') as f:
        pinyin2id = json.load(f)
    id2pinyin = {v: k for k, v in pinyin2id.items()}

    test_loader = DataLoader(
        MTLDataset(TEST_JSON_PATH, pinyin2id),
        batch_size=4,
        shuffle=False,
        collate_fn=mtl_collate_fn,
    )
    model = MultiTaskWavLMLarge(model_path=LOCAL_MODEL_PATH, vocab_size=len(pinyin2id)).to(device)
    model.load_state_dict(torch.load(BEST_WEIGHTS, map_location=device))
    model.eval()

    apa_res = {"t_true": [], "t_pred": [], "i_true": [], "i_pred": [], "f_true": [], "f_pred": []}
    mdd_res = {"true": [], "pred": []}
    asr_res = {"true_ids": [], "pred_ids": []}

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="🔍 执行评估中"):
            apa_scores, mdd_logits, asr_logits = model(
                batch["input_values"].to(device),
                batch["target_pinyin_ids"].to(device),
            )

            apa_res["t_true"].extend(batch["scores"][:, 0].numpy())
            apa_res["t_pred"].extend(apa_scores[:, 0].cpu().numpy())
            apa_res["i_true"].extend(batch["scores"][:, 1].numpy())
            apa_res["i_pred"].extend(apa_scores[:, 1].cpu().numpy())
            apa_res["f_true"].extend(batch["scores"][:, 2].numpy())
            apa_res["f_pred"].extend(apa_scores[:, 2].cpu().numpy())

            mdd_res["true"].extend(batch["mdd_labels"].numpy())
            mdd_res["pred"].extend(torch.argmax(mdd_logits, dim=1).cpu().numpy())

            asr_res["true_ids"].extend(batch["actual_pinyin_ids"].numpy())
            asr_res["pred_ids"].extend(torch.argmax(asr_logits, dim=1).cpu().numpy())

    pcc_tot, _ = pearsonr(apa_res["t_true"], apa_res["t_pred"])
    pcc_ini, _ = pearsonr(apa_res["i_true"], apa_res["i_pred"])
    pcc_fin, _ = pearsonr(apa_res["f_true"], apa_res["f_pred"])
    mse_tot = mean_squared_error(apa_res["t_true"], apa_res["t_pred"])
    mae_tot = mean_absolute_error(apa_res["t_true"], apa_res["t_pred"])

    mdd_t_np, mdd_p_np = np.array(mdd_res["true"]), np.array(mdd_res["pred"])
    mdd_acc = accuracy_score(mdd_t_np, mdd_p_np)
    mdd_f1 = f1_score(mdd_t_np, mdd_p_np, average='macro')
    actual_wrong = (mdd_t_np > 0)
    far = np.sum(mdd_p_np[actual_wrong] == 0) / np.sum(actual_wrong) if np.sum(actual_wrong) > 0 else 0

    asr_t_ids, asr_p_ids = np.array(asr_res["true_ids"]), np.array(asr_res["pred_ids"])
    asr_acc = accuracy_score(asr_t_ids, asr_p_ids)
    cer = 1.0 - asr_acc

    total_dist_p, total_len_p = 0, 0
    for t_id, p_id in zip(asr_t_ids, asr_p_ids):
        t_phonemes = pinyin_split(id2pinyin.get(t_id, ""))
        p_phonemes = pinyin_split(id2pinyin.get(p_id, ""))
        total_dist_p += get_edit_distance(t_phonemes, p_phonemes)
        total_len_p += len(t_phonemes)
    per = total_dist_p / total_len_p if total_len_p > 0 else 0

    print("\n" + "═" * 65)
    print(f"{'🏆 WavLM-Large 单音节多任务模型最终评估报告':^58}")
    print("═" * 65)
    print(f"📊 [APA 任务]  PCC: {pcc_tot:.4f} (总) | {pcc_ini:.4f} (声) | {pcc_fin:.4f} (韵)")
    print(f"               MSE: {mse_tot:.4f} | MAE: {mae_tot:.4f}")
    print("-" * 65)
    print(f"📊 [MDD 任务]  Acc: {mdd_acc * 100:.2f}% | Macro-F1: {mdd_f1:.4f} | FAR: {far * 100:.2f}%")
    print("-" * 65)
    print(f"📊 [ASR 任务]  Syllable Acc (全对率): {asr_acc * 100:.2f}%")
    print(f"               CER (拼音错误率):      {cer * 100:.2f}%")
    print(f"               PER (部件错误率):      {per:.4f}")
    print("═" * 65 + "\n")


if __name__ == "__main__":
    test_best_model()
