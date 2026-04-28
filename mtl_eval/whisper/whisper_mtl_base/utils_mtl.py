import json
import math
from pathlib import Path


SPECIAL_TOKENS = ["<pad>", "<unk>"]


def load_or_create_vocab(vocab_path, metadata_paths):
    vocab_path = Path(vocab_path)
    if vocab_path.exists():
        with vocab_path.open("r", encoding="utf-8") as f:
            vocab = json.load(f)
        if "<unk>" not in vocab:
            raise ValueError(f"{vocab_path} 缺少 <unk>，无法安全处理未知拼音。")
        return vocab

    tokens = set()
    for metadata_path in metadata_paths:
        metadata_path = Path(metadata_path)
        if not metadata_path.exists():
            continue
        with metadata_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            for key in ("target_pinyin", "actual_pinyin"):
                value = item.get(key)
                if value:
                    tokens.add(value)

    if not tokens:
        raise FileNotFoundError(
            f"找不到词表 {vocab_path}，也无法从 metadata 文件生成词表。"
        )

    vocab = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}
    for token in sorted(tokens):
        if token not in vocab:
            vocab[token] = len(vocab)

    with vocab_path.open("w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print(f"✅ 已自动生成拼音词表: {vocab_path} | size={len(vocab)}")
    return vocab


def safe_pearsonr(pearsonr_fn, y_true, y_pred):
    if len(y_true) < 2 or len(y_pred) < 2:
        return 0.0
    try:
        value, _ = pearsonr_fn(y_true, y_pred)
    except Exception:
        return 0.0
    return 0.0 if math.isnan(value) else float(value)


def split_pinyin_units(pinyin):
    if "_" in pinyin:
        return [part for part in pinyin.split("_") if part]
    if " " in pinyin:
        return [part for part in pinyin.split() if part]
    return list(pinyin)


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


def pinyin_error_rate(true_ids, pred_ids, id2pinyin):
    total_edit_distance = 0
    total_units = 0
    for true_id, pred_id in zip(true_ids, pred_ids):
        true_pinyin = id2pinyin.get(int(true_id), "")
        pred_pinyin = id2pinyin.get(int(pred_id), "")
        true_units = split_pinyin_units(true_pinyin)
        pred_units = split_pinyin_units(pred_pinyin)
        total_edit_distance += get_edit_distance(true_units, pred_units)
        total_units += len(true_units)
    return total_edit_distance / total_units if total_units > 0 else 0.0
