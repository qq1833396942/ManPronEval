#!/usr/bin/env python3
import json
import os
import time

import numpy as np
import soundfile as sf
import torch
import tgt
from peft import PeftModel
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor


BASE_MODEL = os.environ.get("WHISPER_MODEL_NAME", "/whisper-large-v3")
LORA_DIR = os.environ.get(
    "LORA_DIR",
    "best_lora_model",
)
TEST_DIR = os.environ.get("SOLO_TEST_DIR", "/data/test")
OUT_JSON = os.environ.get(
    "OUT_JSON",
    "test_strict_cer_per.json",
)
OUT_JSONL = os.environ.get(
    "OUT_JSONL",
    "test_strict_predictions.jsonl",
)
BATCH_SIZE = int(os.environ.get("EVAL_BATCH_SIZE", "16"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "32"))
SAMPLE_RATE = 16000

SHENGMU_LIST = [
    "zh",
    "ch",
    "sh",
    "b",
    "p",
    "m",
    "f",
    "d",
    "t",
    "n",
    "l",
    "g",
    "k",
    "h",
    "j",
    "q",
    "x",
    "r",
    "z",
    "c",
    "s",
    "y",
    "w",
]


def parse_textgrid(tg_path):
    try:
        tg = tgt.io.read_textgrid(tg_path)
    except UnicodeDecodeError:
        try:
            tg = tgt.io.read_textgrid(tg_path, encoding="utf-16")
        except Exception:
            return None
    except Exception:
        return None

    try:
        tier = tg.tiers[1]
        return " ".join(
            [
                interval.text.strip()
                for interval in tier
                if interval.text.strip() and interval.text.strip() not in ["sp", "sil"]
            ]
        )
    except Exception:
        return None


def scan_solo_data(root_path):
    data = []
    if not os.path.exists(root_path):
        return data

    for char_dir in sorted(os.listdir(root_path)):
        char_path = os.path.join(root_path, char_dir)
        if not os.path.isdir(char_path):
            continue

        for sub_dir in sorted(os.listdir(char_path)):
            target_path = os.path.join(char_path, sub_dir)
            if not os.path.isdir(target_path):
                continue

            files = os.listdir(target_path)
            lower_to_name = {name.lower(): name for name in files}
            for file_name in sorted(files):
                if not file_name.lower().endswith(".wav"):
                    continue

                stem = os.path.splitext(file_name)[0]
                tg_name = lower_to_name.get((stem + ".textgrid").lower())
                if not tg_name:
                    continue

                audio_path = os.path.join(target_path, file_name)
                tg_path = os.path.join(target_path, tg_name)
                ref = parse_textgrid(tg_path)
                if ref:
                    data.append(
                        {
                            "audio": audio_path,
                            "ref": ref,
                            "char": char_dir,
                            "subdir": sub_dir,
                        }
                    )
    return data


def strip_prefix_only_tokens(raw_pred):
    text = raw_pred.strip().lower().replace("：", ":")
    if text.startswith("拼音:"):
        text = text[len("拼音:") :].strip()
    elif text.startswith("拼音"):
        text = text[len("拼音") :].strip()
        if text.startswith(":"):
            text = text[1:].strip()
    return text.split() if text else []


def ref_tokens(text):
    return text.strip().lower().split() if text and text.strip() else []


def split_syllable(unit):
    unit = unit.strip().lower().replace("ü", "v")
    shengmu = ""
    yunmu = unit
    if len(unit) >= 2 and unit[:2] in SHENGMU_LIST:
        shengmu, yunmu = unit[:2], unit[2:]
    elif unit[:1] in SHENGMU_LIST:
        shengmu, yunmu = unit[:1], unit[1:]
    return shengmu or "<null>", yunmu or "<null>"


def split_pinyin_list(tokens):
    parts = []
    for unit in tokens:
        shengmu, yunmu = split_syllable(unit)
        parts.extend([shengmu, yunmu])
    return parts


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(
                min(
                    prev[j] + 1,
                    cur[j - 1] + 1,
                    prev[j - 1] + (0 if x == y else 1),
                )
            )
        prev = cur
    return prev[-1]


def load_audio(path):
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav[:, 0]
    if sr != SAMPLE_RATE:
        import librosa

        wav = librosa.resample(wav, orig_sr=sr, target_sr=SAMPLE_RATE)
    return wav


def main():
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    print(f"base_model={BASE_MODEL}")
    print(f"lora_dir={LORA_DIR}")
    print(f"test_dir={TEST_DIR}")

    data = scan_solo_data(TEST_DIR)
    print(f"test_samples={len(data)}")
    if not data:
        raise SystemExit("No test samples found")

    processor = WhisperProcessor.from_pretrained(BASE_MODEL)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    base = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL, torch_dtype=dtype)
    model = PeftModel.from_pretrained(base, LORA_DIR)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    total_syl_dist = 0
    total_syl_ref = 0
    total_syl_hyp = 0
    total_per_dist = 0
    total_per_ref = 0
    total_per_hyp = 0
    exact_syl = 0
    empty_hyp = 0
    prefix_count = 0
    t0 = time.time()

    with open(OUT_JSONL, "w", encoding="utf-8") as fw:
        for start in tqdm(range(0, len(data), BATCH_SIZE), desc="Evaluating"):
            batch = data[start : start + BATCH_SIZE]
            wavs = [load_audio(item["audio"]) for item in batch]
            features = processor.feature_extractor(
                wavs, sampling_rate=SAMPLE_RATE, return_tensors="pt"
            ).input_features
            features = features.to(device=device, dtype=dtype)

            with torch.inference_mode():
                pred_ids = model.generate(features, max_new_tokens=MAX_NEW_TOKENS)
            preds = processor.batch_decode(pred_ids, skip_special_tokens=True)

            for item, raw_pred in zip(batch, preds):
                raw_pred = raw_pred.strip()
                if raw_pred.lower().replace("：", ":").startswith("拼音"):
                    prefix_count += 1

                rt = ref_tokens(item["ref"])
                ht = strip_prefix_only_tokens(raw_pred)
                if not ht:
                    empty_hyp += 1

                syl_dist = edit_distance(rt, ht)
                ref_parts = split_pinyin_list(rt)
                hyp_parts = split_pinyin_list(ht)
                per_dist = edit_distance(ref_parts, hyp_parts)

                total_syl_dist += syl_dist
                total_syl_ref += len(rt)
                total_syl_hyp += len(ht)
                total_per_dist += per_dist
                total_per_ref += len(ref_parts)
                total_per_hyp += len(hyp_parts)
                if syl_dist == 0 and len(rt) == len(ht):
                    exact_syl += 1

                fw.write(
                    json.dumps(
                        {
                            "audio": item["audio"],
                            "char": item["char"],
                            "subdir": item["subdir"],
                            "ref": item["ref"],
                            "raw_pred": raw_pred,
                            "hyp_tokens": ht,
                            "ref_tokens": rt,
                            "syl_edit_distance": syl_dist,
                            "per_edit_distance": per_dist,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    summary = {
        "base_model": BASE_MODEL,
        "lora_dir": LORA_DIR,
        "test_dir": TEST_DIR,
        "samples": len(data),
        "batch_size": BATCH_SIZE,
        "max_new_tokens": MAX_NEW_TOKENS,
        "metric_note": (
            "Strict edit distance. Only leading prompt prefix '拼音:'/'拼音：' is stripped "
            "from predictions. No regex cleanup; empty/illegal/extra tokens are counted. "
            "PER splits each syllable into initial+final, with tone kept inside final."
        ),
        "cer": total_syl_dist / total_syl_ref if total_syl_ref else None,
        "per": total_per_dist / total_per_ref if total_per_ref else None,
        "syllable_exact_accuracy": exact_syl / len(data) if data else None,
        "total_syllable_edit_distance": total_syl_dist,
        "total_ref_syllables": total_syl_ref,
        "total_hyp_syllables": total_syl_hyp,
        "total_per_edit_distance": total_per_dist,
        "total_ref_initial_final_tokens": total_per_ref,
        "total_hyp_initial_final_tokens": total_per_hyp,
        "empty_hypotheses": empty_hyp,
        "predictions_with_prompt_prefix": prefix_count,
        "seconds": time.time() - t0,
        "output_jsonl": OUT_JSONL,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("SUMMARY_JSON=" + json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
