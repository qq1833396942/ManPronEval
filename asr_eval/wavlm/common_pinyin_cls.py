# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import librosa
import numpy as np
import tgt
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from transformers import AutoFeatureExtractor, AutoModel, AutoProcessor


DEFAULT_CONFIG: Dict[str, Any] = {
    "train_dir": "./data/train",
    "val_dir": "./data/val",
    "test_dir": "./data/test",
    "vocab_path": "./syllable_vocab.json",
    "base_model": "jonatasgrosman/exp_w2v2t_zh-cn_wavlm_s677",
    "output_root": "./outputs_wavlm_pinyin_official_style",
    "eval_ckpt_dir": "",
    "seed": 42,
    "epochs": 20,
    "batch_size": 8,
    "eval_batch_size": 16,
    "num_workers": 0,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "fp16": True,
    "use_lora": True,
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "lora_bias": "none",
    "lora_target_preference": ["q_proj", "v_proj"],
    "classifier_hidden": 1024,
    "classifier_dropout": 0.1,
    "pooling_mode": "masked_mean",
    "encoder_lr": 5e-5,
    "classifier_lr": 1e-3,
    "weight_decay": 0.01,
    "grad_clip": 1.0,
    "sample_rate": 16000,

    # 音频预处理：更贴近 WavLM，默认不手工归一化，避免和 processor 重复
    "trim_top_db": None,
    "min_after_trim_samples": 160,
    "min_input_samples": 1600,
    "normalize_wave": False,

    # 标签读取
    # tier_name: 按 tier 名找
    # tier_index: 按固定层号找（1-based）
    "label_source_mode": "tier_name",
    "label_tier_index": 2,
    "tier_name": "syllables",
    "tier_name_candidates": ["syllables", "Syllables", "syllable", "SYLLABLES"],
    "label_read_mode": "first_valid",  # first_interval / first_non_empty / first_valid / longest_valid_duration
    "ignore_labels": ["sil", "sp", "spn", "pau", "noise", "<sil>", "<sp>", "<spn>"],
    "strip_internal_spaces": False,

    "probe_every_steps": 50,
    "probe_num_val_samples": 32,
    "print_top_label_count": 20,
    "balanced_sampling": False,
    "save_last": True,
    "test_progress_every_batches": 20,
}


def zh_print(msg: str) -> None:
    print(f"[中文提示] {msg}", flush=True)


def load_config(config_path: str | Path) -> Dict[str, Any]:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"未找到配置文件: {config_path}")
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    merged = dict(DEFAULT_CONFIG)
    merged.update(cfg)
    return merged


def save_json(path: str | Path, obj: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def save_jsonl(path: str | Path, rows: List[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_tsv(path: str | Path, header: List[str], row: List[Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", encoding="utf-8") as f:
            f.write("\t".join(header) + "\n")
    with path.open("a", encoding="utf-8") as f:
        f.write("\t".join(str(x) for x in row) + "\n")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_label(text: str, strip_internal_spaces: bool = False) -> str:
    text = (text or "").strip().lower()
    if strip_internal_spaces:
        text = "".join(text.split())
    return text


def format_num(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.2f}K"
    return str(n)


def safe_relpath(path: str | Path, start: str | Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(Path(start).resolve()))
    except Exception:
        return str(path)


def find_matching_textgrid(wav_path: str | Path) -> Optional[Path]:
    wav_path = Path(wav_path)
    for cand in wav_path.parent.glob(wav_path.stem + ".*"):
        if cand.suffix.lower() == ".textgrid":
            return cand
    for ext in [".TextGrid", ".textgrid", ".TEXTGRID"]:
        cand = wav_path.with_suffix(ext)
        if cand.exists():
            return cand
    return None


def read_textgrid_with_fallback(tg_path: str | Path):
    tg_path = Path(tg_path)
    candidates = ["utf-8", "utf-8-sig", "utf-16", "utf-16-le", "utf-16-be", "gb18030"]
    last_error = None
    for enc in candidates:
        try:
            tg = tgt.io.read_textgrid(str(tg_path), encoding=enc)
            return tg, enc
        except Exception as e:
            last_error = e
    raise RuntimeError(
        f"TextGrid 读取失败，已尝试编码={candidates} | 最后错误={type(last_error).__name__}: {str(last_error)}"
    )


def inspect_textgrid_tiers(tg_path: str | Path) -> List[str]:
    try:
        tg, used_encoding = read_textgrid_with_fallback(tg_path)
        tier_names = [getattr(t, "name", "") for t in getattr(tg, "tiers", [])]
        return [f"[encoding={used_encoding}]"] + tier_names
    except Exception as e:
        return [f"[读取tier列表失败] {type(e).__name__}: {str(e)}"]


def resolve_target_tier(
    tg,
    label_source_mode: str = "tier_name",
    tier_name: str = "syllables",
    tier_name_candidates: Optional[List[str]] = None,
    label_tier_index: int = 2,
    strip_internal_spaces: bool = False,
):
    if label_source_mode == "tier_index":
        tiers = list(getattr(tg, "tiers", []))
        idx0 = int(label_tier_index) - 1
        if idx0 < 0 or idx0 >= len(tiers):
            available = [getattr(t, "name", "") for t in tiers]
            raise IndexError(f"tier_index={label_tier_index} 越界 | 可用tiers={available}")
        return tiers[idx0]

    candidates = tier_name_candidates or [tier_name]
    candidates_norm = [normalize_label(x, strip_internal_spaces=strip_internal_spaces) for x in candidates]

    matched = None
    for t in getattr(tg, "tiers", []):
        t_name = getattr(t, "name", "")
        if normalize_label(t_name, strip_internal_spaces=strip_internal_spaces) in candidates_norm:
            matched = t
            break

    if matched is None:
        available = [getattr(t, "name", "") for t in getattr(tg, "tiers", [])]
        raise KeyError(f"未找到目标tier。期望候选={candidates} | 实际tiers={available}")

    return matched


def read_textgrid_label(
    tg_path: str | Path,
    label_source_mode: str = "tier_name",
    label_tier_index: int = 2,
    tier_name: str = "syllables",
    tier_name_candidates: Optional[List[str]] = None,
    label_read_mode: str = "first_valid",
    ignore_labels: Optional[List[str]] = None,
    strip_internal_spaces: bool = False,
) -> Dict[str, Any]:
    ignore_set = set(
        normalize_label(x, strip_internal_spaces=strip_internal_spaces)
        for x in (ignore_labels or [])
    )

    tg, used_encoding = read_textgrid_with_fallback(tg_path)
    target_tier = resolve_target_tier(
        tg=tg,
        label_source_mode=label_source_mode,
        tier_name=tier_name,
        tier_name_candidates=tier_name_candidates,
        label_tier_index=label_tier_index,
        strip_internal_spaces=strip_internal_spaces,
    )

    intervals = getattr(target_tier, "intervals", [])
    if not intervals:
        return {
            "label": "",
            "used_encoding": used_encoding,
            "matched_tier_name": getattr(target_tier, "name", ""),
            "selected_interval_index": None,
            "selected_interval_start": None,
            "selected_interval_end": None,
            "num_intervals": 0,
            "preview_intervals": [],
        }

    processed = []
    for idx, interval in enumerate(intervals):
        text_raw = getattr(interval, "text", "")
        text_norm = normalize_label(text_raw, strip_internal_spaces=strip_internal_spaces)
        processed.append(
            {
                "index": idx,
                "start_time": float(getattr(interval, "start_time", 0.0)),
                "end_time": float(getattr(interval, "end_time", 0.0)),
                "duration": float(getattr(interval, "end_time", 0.0) - getattr(interval, "start_time", 0.0)),
                "text_raw": text_raw,
                "text_norm": text_norm,
                "is_empty": (text_norm == ""),
                "is_ignored": (text_norm in ignore_set),
            }
        )

    selected = None
    if label_read_mode == "first_interval":
        selected = processed[0]
    elif label_read_mode == "first_non_empty":
        for row in processed:
            if not row["is_empty"]:
                selected = row
                break
    elif label_read_mode == "first_valid":
        for row in processed:
            if (not row["is_empty"]) and (not row["is_ignored"]):
                selected = row
                break
    elif label_read_mode == "longest_valid_duration":
        valid_rows = [row for row in processed if (not row["is_empty"]) and (not row["is_ignored"])]
        if valid_rows:
            selected = max(valid_rows, key=lambda x: x["duration"])
    else:
        raise ValueError(f"不支持的 label_read_mode: {label_read_mode}")

    if selected is None:
        label = ""
        selected_idx = None
        selected_start = None
        selected_end = None
    else:
        label = selected["text_norm"]
        selected_idx = selected["index"]
        selected_start = selected["start_time"]
        selected_end = selected["end_time"]

    return {
        "label": label,
        "used_encoding": used_encoding,
        "matched_tier_name": getattr(target_tier, "name", ""),
        "selected_interval_index": selected_idx,
        "selected_interval_start": selected_start,
        "selected_interval_end": selected_end,
        "num_intervals": len(processed),
        "preview_intervals": processed[:10],
    }


def load_vocab(vocab_path: str | Path) -> Tuple[Dict[str, int], Dict[int, str]]:
    vocab = json.loads(Path(vocab_path).read_text(encoding="utf-8"))
    vocab = {str(k): int(v) for k, v in vocab.items()}
    inv_vocab = {int(v): k for k, v in vocab.items()}
    return vocab, inv_vocab


def split_pinyin(syllable: str) -> List[str]:
    if syllable in ["[unk]", "[pad]", "[UNK]", "[PAD]", ""]:
        return [syllable]

    initials = [
        "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h",
        "j", "q", "x", "zh", "ch", "sh", "r", "z", "c", "s", "y", "w"
    ]
    for init in sorted(initials, key=len, reverse=True):
        if syllable.startswith(init):
            final = syllable[len(init):]
            return [init, final] if final else [init]
    return [syllable]


def calculate_cer(refs: List[str], preds: List[str]) -> float:
    if not refs:
        return 0.0
    errors = sum(1 for r, p in zip(refs, preds) if r != p)
    return errors / len(refs)


def calculate_per(refs: List[str], preds: List[str]) -> float:
    total_phonemes = 0
    errors = 0
    for ref, pred in zip(refs, preds):
        ref_tokens = split_pinyin(ref)
        pred_tokens = split_pinyin(pred)
        total_phonemes += len(ref_tokens)

        for r, p in zip(ref_tokens, pred_tokens):
            if r != p:
                errors += 1
        errors += abs(len(ref_tokens) - len(pred_tokens))

    if total_phonemes == 0:
        return 0.0
    return errors / total_phonemes


def load_audio_waveform(wav_path: str | Path, cfg: Dict[str, Any]) -> np.ndarray:
    wav, _ = librosa.load(str(wav_path), sr=int(cfg["sample_rate"]), mono=True)
    wav = wav.astype(np.float32)

    trim_top_db = cfg.get("trim_top_db", None)
    if trim_top_db is not None:
        raw = wav
        trimmed, _ = librosa.effects.trim(wav, top_db=float(trim_top_db))
        if len(trimmed) >= int(cfg["min_after_trim_samples"]):
            wav = trimmed
        else:
            wav = raw

    if wav.size == 0:
        wav = np.zeros(int(cfg["min_input_samples"]), dtype=np.float32)

    if bool(cfg.get("normalize_wave", False)):
        wav = (wav - float(wav.mean())) / (float(wav.std()) + 1e-6)

    min_input_samples = int(cfg["min_input_samples"])
    if len(wav) < min_input_samples:
        wav = np.pad(wav, (0, min_input_samples - len(wav)), mode="constant")

    return wav.astype(np.float32)


def load_audio_processor(base_model: str):
    errors = []
    try:
        processor = AutoProcessor.from_pretrained(base_model)
        return processor, {"processor_class": processor.__class__.__name__, "load_mode": "AutoProcessor"}
    except Exception as e:
        errors.append(f"AutoProcessor: {type(e).__name__}: {str(e)}")

    try:
        processor = AutoFeatureExtractor.from_pretrained(base_model)
        return processor, {
            "processor_class": processor.__class__.__name__,
            "load_mode": "AutoFeatureExtractor",
            "fallback_reason": errors,
        }
    except Exception as e:
        errors.append(f"AutoFeatureExtractor: {type(e).__name__}: {str(e)}")

    raise RuntimeError(f"无法加载音频 processor / feature extractor: {errors}")


def processor_supports_attention_mask(processor) -> bool:
    fe = getattr(processor, "feature_extractor", None)
    if fe is not None and hasattr(fe, "return_attention_mask"):
        return bool(getattr(fe, "return_attention_mask"))
    if hasattr(processor, "return_attention_mask"):
        return bool(getattr(processor, "return_attention_mask"))
    return False


class PinyinFolderDataset(Dataset):
    def __init__(self, split_dir: str | Path, vocab: Dict[str, int], cfg: Dict[str, Any], split_name: str):
        self.split_dir = Path(split_dir)
        self.vocab = vocab
        self.cfg = cfg
        self.split_name = split_name

        self.samples: List[Dict[str, Any]] = []
        self.label_counter: Counter = Counter()
        self.oov_counter: Counter = Counter()
        self.encoding_counter: Counter = Counter()
        self.matched_tier_counter: Counter = Counter()

        self.missing_textgrid_samples: List[Dict[str, Any]] = []
        self.tier_read_error_samples: List[Dict[str, Any]] = []
        self.empty_label_samples: List[Dict[str, Any]] = []
        self.oov_samples: List[Dict[str, Any]] = []

        total_wavs = 0
        paired_textgrids = 0
        missing_textgrid = 0
        empty_label = 0
        oov_label = 0
        tier_read_error = 0

        wav_paths = [p for p in self.split_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".wav"]
        wav_paths = sorted(wav_paths, key=lambda x: str(x))

        for wav_path in wav_paths:
            total_wavs += 1
            tg_path = find_matching_textgrid(wav_path)

            if tg_path is None:
                missing_textgrid += 1
                self.missing_textgrid_samples.append({
                    "split_name": split_name,
                    "utt_id": wav_path.stem,
                    "wav_path": str(wav_path),
                    "relative_path": safe_relpath(wav_path, self.split_dir),
                    "reason": "missing_textgrid",
                })
                continue

            paired_textgrids += 1

            try:
                read_info = read_textgrid_label(
                    tg_path=tg_path,
                    label_source_mode=str(cfg.get("label_source_mode", "tier_name")),
                    label_tier_index=int(cfg.get("label_tier_index", 2)),
                    tier_name=str(cfg.get("tier_name", "syllables")),
                    tier_name_candidates=list(cfg.get("tier_name_candidates", [])),
                    label_read_mode=str(cfg.get("label_read_mode", "first_valid")),
                    ignore_labels=list(cfg.get("ignore_labels", [])),
                    strip_internal_spaces=bool(cfg.get("strip_internal_spaces", False)),
                )
                label = read_info["label"]
            except Exception as e:
                tier_read_error += 1
                self.tier_read_error_samples.append({
                    "split_name": split_name,
                    "utt_id": wav_path.stem,
                    "wav_path": str(wav_path),
                    "textgrid_path": str(tg_path),
                    "relative_path": safe_relpath(wav_path, self.split_dir),
                    "label_source_mode": str(cfg.get("label_source_mode", "tier_name")),
                    "label_tier_index": int(cfg.get("label_tier_index", 2)),
                    "tier_name_expected": str(cfg.get("tier_name", "syllables")),
                    "tier_name_candidates": list(cfg.get("tier_name_candidates", [])),
                    "label_read_mode": str(cfg.get("label_read_mode", "first_valid")),
                    "ignore_labels": list(cfg.get("ignore_labels", [])),
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "available_tier_names": inspect_textgrid_tiers(tg_path),
                })
                continue

            if not label:
                empty_label += 1
                self.empty_label_samples.append({
                    "split_name": split_name,
                    "utt_id": wav_path.stem,
                    "wav_path": str(wav_path),
                    "textgrid_path": str(tg_path),
                    "relative_path": safe_relpath(wav_path, self.split_dir),
                    "reason": "empty_label",
                    "read_info": read_info,
                })
                continue

            if label not in self.vocab:
                oov_label += 1
                self.oov_counter[label] += 1
                self.oov_samples.append({
                    "split_name": split_name,
                    "utt_id": wav_path.stem,
                    "wav_path": str(wav_path),
                    "textgrid_path": str(tg_path),
                    "relative_path": safe_relpath(wav_path, self.split_dir),
                    "label": label,
                    "reason": "oov_label",
                    "read_info": read_info,
                })
                continue

            char_text = wav_path.parent.parent.name if len(wav_path.parts) >= 3 else ""
            self.samples.append({
                "utt_id": wav_path.stem,
                "wav_path": str(wav_path),
                "textgrid_path": str(tg_path),
                "ref_text": char_text,
                "ref_pinyin": label,
                "label_id": int(self.vocab[label]),
                "relative_path": safe_relpath(wav_path, self.split_dir),
                "textgrid_encoding": read_info["used_encoding"],
                "matched_tier_name": read_info["matched_tier_name"],
                "selected_interval_index": read_info["selected_interval_index"],
            })
            self.label_counter[label] += 1
            self.encoding_counter[read_info["used_encoding"]] += 1
            self.matched_tier_counter[read_info["matched_tier_name"]] += 1

        used = len(self.samples)
        coverage = used / total_wavs if total_wavs else 0.0
        self.stats = {
            "split_name": split_name,
            "split_dir": str(self.split_dir),
            "num_total_wavs": total_wavs,
            "num_paired_textgrids": paired_textgrids,
            "num_missing_textgrid": missing_textgrid,
            "num_tier_read_error": tier_read_error,
            "num_empty_label": empty_label,
            "num_oov_label": oov_label,
            "num_used": used,
            "coverage": coverage,
            "num_unique_labels": len(self.label_counter),
            "encoding_counts": dict(self.encoding_counter.most_common()),
            "matched_tier_counts": dict(self.matched_tier_counter.most_common()),
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        item = self.samples[idx]
        try:
            waveform = load_audio_waveform(item["wav_path"], self.cfg)
        except Exception as e:
            return {"load_failed": True, "error": str(e), "utt_id": item["utt_id"], "wav_path": item["wav_path"]}

        return {
            "utt_id": item["utt_id"],
            "waveform": waveform,
            "length": int(len(waveform)),
            "label_id": int(item["label_id"]),
            "ref_pinyin": item["ref_pinyin"],
            "ref_text": item["ref_text"],
            "relative_path": item["relative_path"],
        }


def build_collate_fn(processor, sample_rate: int = 16000):
    processor_can_return_am = processor_supports_attention_mask(processor)

    def collate_batch(batch: List[Optional[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
        batch2 = [x for x in batch if x is not None and not x.get("load_failed", False)]
        if not batch2:
            return None

        waveforms = [item["waveform"] for item in batch2]
        lengths = [int(item["length"]) for item in batch2]

        call_kwargs = {"sampling_rate": int(sample_rate), "padding": True, "return_tensors": "pt"}
        if processor_can_return_am:
            call_kwargs["return_attention_mask"] = True

        proc_out = processor(waveforms, **call_kwargs)
        input_values = proc_out["input_values"]
        encoder_attention_mask = proc_out.get("attention_mask", None) if processor_can_return_am else None

        if proc_out.get("attention_mask", None) is not None:
            pooling_attention_mask = proc_out["attention_mask"].long()
        else:
            pooling_attention_mask = torch.zeros(input_values.shape[0], input_values.shape[1], dtype=torch.long)
            for i, L in enumerate(lengths):
                pooling_attention_mask[i, :L] = 1

        return {
            "utt_ids": [item["utt_id"] for item in batch2],
            "input_values": input_values.float(),
            "encoder_attention_mask": encoder_attention_mask.long() if encoder_attention_mask is not None else None,
            "pooling_attention_mask": pooling_attention_mask.long(),
            "label_ids": torch.tensor([item["label_id"] for item in batch2], dtype=torch.long),
            "ref_pinyins": [item["ref_pinyin"] for item in batch2],
            "ref_texts": [item["ref_text"] for item in batch2],
            "relative_paths": [item["relative_path"] for item in batch2],
            "lengths": lengths,
        }

    return collate_batch


def build_loader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int, collate_fn, weighted: bool = False):
    pin_memory = torch.cuda.is_available()

    if weighted and hasattr(dataset, "samples") and len(getattr(dataset, "samples", [])) > 0:
        label_ids = [int(x["label_id"]) for x in dataset.samples]
        counter = Counter(label_ids)
        weights = [1.0 / counter[y] for y in label_ids]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=False, num_workers=num_workers, collate_fn=collate_fn, pin_memory=pin_memory)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn, pin_memory=pin_memory)


def resolve_lora_targets(model: nn.Module, target_preference: List[str]) -> Tuple[List[str], List[str]]:
    linear_leaf_names = set()
    matched_full_names = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            leaf = name.split(".")[-1]
            linear_leaf_names.add(leaf)

    targets = [x for x in target_preference if x in linear_leaf_names]
    if not targets:
        fallback_candidates = [["query", "value"], ["q_proj", "v_proj"], ["q", "v"]]
        for cand in fallback_candidates:
            if all(x in linear_leaf_names for x in cand):
                targets = cand
                break

    if not targets:
        raise ValueError(f"未找到可用于 LoRA 的目标模块。当前线性层末级名字示例: {sorted(list(linear_leaf_names))[:50]}")

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name.split(".")[-1] in targets:
            matched_full_names.append(name)

    return targets, matched_full_names


def get_core_encoder(model_or_peft: nn.Module) -> nn.Module:
    if hasattr(model_or_peft, "get_base_model"):
        try:
            return model_or_peft.get_base_model()
        except Exception:
            pass
    return model_or_peft


def build_feature_vector_attention_mask(encoder: nn.Module, hidden_len: int, raw_attention_mask: torch.Tensor) -> torch.Tensor:
    core = get_core_encoder(encoder)
    if hasattr(core, "_get_feature_vector_attention_mask"):
        return core._get_feature_vector_attention_mask(hidden_len, raw_attention_mask)
    m = raw_attention_mask.unsqueeze(1).float()
    m = F.interpolate(m, size=hidden_len, mode="nearest").squeeze(1)
    return (m > 0.5).long()


class SingleSyllableWavLM(nn.Module):
    def __init__(self, cfg: Dict[str, Any], vocab_size: int):
        super().__init__()
        self.cfg = cfg
        self.pooling_mode = str(cfg.get("pooling_mode", "masked_mean")).strip().lower()
        self.encoder = AutoModel.from_pretrained(str(cfg["base_model"]))
        hidden_size = int(self.encoder.config.hidden_size)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, int(cfg["classifier_hidden"])),
            nn.GELU(),
            nn.LayerNorm(int(cfg["classifier_hidden"])),
            nn.Dropout(float(cfg["classifier_dropout"])),
            nn.Linear(int(cfg["classifier_hidden"]), vocab_size),
        )

        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.lora_summary: Dict[str, Any] = {
            "enabled": bool(cfg.get("use_lora", True)),
            "target_modules": [],
            "matched_module_paths": [],
            "r": None,
            "alpha": None,
            "dropout": None,
            "bias": None,
            "note": "",
        }

        if bool(cfg.get("use_lora", True)):
            targets, matched_names = resolve_lora_targets(self.encoder, list(cfg.get("lora_target_preference", ["q_proj", "v_proj"])))
            lora_config = LoraConfig(
                r=int(cfg["lora_r"]),
                lora_alpha=int(cfg["lora_alpha"]),
                target_modules=targets,
                lora_dropout=float(cfg["lora_dropout"]),
                bias=str(cfg.get("lora_bias", "none")),
            )
            self.encoder = get_peft_model(self.encoder, lora_config)
            self.lora_summary = {
                "enabled": True,
                "target_modules": targets,
                "matched_module_paths": matched_names,
                "r": int(cfg["lora_r"]),
                "alpha": int(cfg["lora_alpha"]),
                "dropout": float(cfg["lora_dropout"]),
                "bias": str(cfg.get("lora_bias", "none")),
                "note": "已启用 LoRA，只训练 LoRA 参数与分类头。",
            }
        else:
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.lora_summary = {
                "enabled": False,
                "target_modules": [],
                "matched_module_paths": [],
                "r": None,
                "alpha": None,
                "dropout": None,
                "bias": None,
                "note": "未启用 LoRA，默认冻结 encoder，仅训练分类头。",
            }

    def forward(self, input_values: torch.Tensor, encoder_attention_mask: Optional[torch.Tensor] = None, pooling_attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        outputs = self.encoder(input_values=input_values, attention_mask=encoder_attention_mask)
        hidden = outputs.last_hidden_state

        if self.pooling_mode == "masked_mean" and pooling_attention_mask is not None:
            feat_mask = build_feature_vector_attention_mask(self.encoder, hidden.shape[1], pooling_attention_mask)
            m = feat_mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        else:
            pooled = hidden.mean(dim=1)

        logits = self.classifier(pooled)
        return logits


def collect_parameter_statistics(model: nn.Module) -> Dict[str, Any]:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    encoder_total = sum(p.numel() for p in model.encoder.parameters())
    encoder_trainable = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    classifier_total = sum(p.numel() for p in model.classifier.parameters())
    classifier_trainable = sum(p.numel() for p in model.classifier.parameters() if p.requires_grad)
    lora_trainable = sum(p.numel() for name, p in model.named_parameters() if p.requires_grad and "lora_" in name)

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_ratio": trainable_params / total_params if total_params else 0.0,
        "encoder_total": encoder_total,
        "encoder_trainable": encoder_trainable,
        "classifier_total": classifier_total,
        "classifier_trainable": classifier_trainable,
        "lora_trainable": lora_trainable,
    }


def print_model_summary(model: nn.Module) -> Dict[str, Any]:
    stats = collect_parameter_statistics(model)
    zh_print("========== 模型参数统计 ==========")
    zh_print(f"总参数量: {format_num(stats['total_params'])} ({stats['total_params']})")
    zh_print(f"可训练参数量: {format_num(stats['trainable_params'])} ({stats['trainable_params']})")
    zh_print(f"可训练占比: {stats['trainable_ratio'] * 100:.4f}%")
    zh_print(f"Encoder 总参数量: {format_num(stats['encoder_total'])} ({stats['encoder_total']})")
    zh_print(f"Encoder 可训练参数量: {format_num(stats['encoder_trainable'])} ({stats['encoder_trainable']})")
    zh_print(f"分类头总参数量: {format_num(stats['classifier_total'])} ({stats['classifier_total']})")
    zh_print(f"分类头可训练参数量: {format_num(stats['classifier_trainable'])} ({stats['classifier_trainable']})")
    zh_print(f"LoRA 可训练参数量: {format_num(stats['lora_trainable'])} ({stats['lora_trainable']})")

    zh_print("========== LoRA 设置 ==========")
    if getattr(model, "lora_summary", {}).get("enabled", False):
        summary = model.lora_summary
        zh_print(f"LoRA r = {summary['r']}")
        zh_print(f"LoRA alpha = {summary['alpha']}")
        zh_print(f"LoRA dropout = {summary['dropout']}")
        zh_print(f"LoRA bias = {summary['bias']}")
        zh_print(f"LoRA target_modules = {summary['target_modules']}")
        zh_print(f"LoRA 命中的模块数 = {len(summary['matched_module_paths'])}")
        preview = summary["matched_module_paths"][:20]
        for name in preview:
            zh_print(f"  - {name}")
        if len(summary["matched_module_paths"]) > len(preview):
            zh_print(f"  ... 其余 {len(summary['matched_module_paths']) - len(preview)} 个模块未展开")
    else:
        zh_print("LoRA 未启用")
    return stats


def decode_ids(ids: List[int], inv_vocab: Dict[int, str]) -> List[str]:
    return [inv_vocab.get(int(i), "[UNK]") for i in ids]


@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader, device: str, inv_vocab: Dict[int, str], max_batches: Optional[int] = None) -> Dict[str, Any]:
    model.eval()
    all_preds: List[str] = []
    all_refs: List[str] = []
    all_utt_ids: List[str] = []
    all_paths: List[str] = []

    used_batches = 0
    for batch in loader:
        if batch is None:
            continue
        input_values = batch["input_values"].to(device)
        encoder_attention_mask = batch["encoder_attention_mask"].to(device) if batch["encoder_attention_mask"] is not None else None
        pooling_attention_mask = batch["pooling_attention_mask"].to(device)
        label_ids = batch["label_ids"].to(device)

        logits = model(input_values=input_values, encoder_attention_mask=encoder_attention_mask, pooling_attention_mask=pooling_attention_mask)
        pred_ids = logits.argmax(dim=-1).detach().cpu().tolist()
        ref_ids = label_ids.detach().cpu().tolist()

        all_preds.extend(decode_ids(pred_ids, inv_vocab))
        all_refs.extend(decode_ids(ref_ids, inv_vocab))
        all_utt_ids.extend(batch["utt_ids"])
        all_paths.extend(batch["relative_paths"])

        used_batches += 1
        if max_batches is not None and used_batches >= max_batches:
            break

    acc = sum(1 for r, p in zip(all_refs, all_preds) if r == p) / len(all_refs) if all_refs else 0.0
    cer = calculate_cer(all_refs, all_preds)
    per = calculate_per(all_refs, all_preds)

    examples = []
    for idx in range(min(5, len(all_refs))):
        examples.append({
            "utt_id": all_utt_ids[idx],
            "relative_path": all_paths[idx],
            "ref_pinyin": all_refs[idx],
            "pred_pinyin": all_preds[idx],
            "ref_split": split_pinyin(all_refs[idx]),
            "pred_split": split_pinyin(all_preds[idx]),
            "match": bool(all_refs[idx] == all_preds[idx]),
        })

    return {
        "pinyin_acc": acc,
        "cer": cer,
        "per": per,
        "num_samples": len(all_refs),
        "examples": examples,
        "refs": all_refs,
        "preds": all_preds,
        "utt_ids": all_utt_ids,
        "relative_paths": all_paths,
    }


def create_probe_loader(val_dataset: Dataset, batch_size: int, num_workers: int, collate_fn, probe_num_samples: int) -> Optional[DataLoader]:
    if len(val_dataset) == 0 or probe_num_samples <= 0:
        return None
    n = min(len(val_dataset), int(probe_num_samples))
    subset = Subset(val_dataset, list(range(n)))
    return DataLoader(subset, batch_size=min(batch_size, n), shuffle=False, num_workers=num_workers, collate_fn=collate_fn, pin_memory=torch.cuda.is_available())


def save_dataset_artifacts(run_dir: str | Path, split_name: str, dataset: PinyinFolderDataset, top_n: int = 20) -> None:
    run_dir = Path(run_dir)
    save_json(run_dir / f"{split_name}_dataset_stats.json", dataset.stats)
    save_json(run_dir / f"{split_name}_oov_labels.json", {"split_name": split_name, "num_oov_types": len(dataset.oov_counter), "oov_counts": dict(dataset.oov_counter.most_common())})
    save_json(run_dir / f"{split_name}_label_counts.json", {"split_name": split_name, "num_label_types": len(dataset.label_counter), "top_label_counts": dict(dataset.label_counter.most_common(top_n))})
    save_jsonl(run_dir / f"{split_name}_missing_textgrid_samples.jsonl", dataset.missing_textgrid_samples)
    save_jsonl(run_dir / f"{split_name}_tier_read_error_samples.jsonl", dataset.tier_read_error_samples)
    save_jsonl(run_dir / f"{split_name}_empty_label_samples.jsonl", dataset.empty_label_samples)
    save_jsonl(run_dir / f"{split_name}_oov_samples.jsonl", dataset.oov_samples)


def print_dataset_summary(split_name: str, dataset: PinyinFolderDataset, top_n: int = 20) -> None:
    zh_print(f"========== {split_name} 数据集统计 ==========")
    for key, value in dataset.stats.items():
        zh_print(f"{key}: {value}")

    top_labels = dataset.label_counter.most_common(top_n)
    if top_labels:
        zh_print(f"{split_name} 标签 Top{min(top_n, len(top_labels))}:")
        for label, count in top_labels:
            zh_print(f"  - {label}: {count}")

    if dataset.oov_counter:
        zh_print(f"{split_name} OOV 标签 Top{min(top_n, len(dataset.oov_counter))}:")
        for label, count in dataset.oov_counter.most_common(top_n):
            zh_print(f"  - {label}: {count}")

    if dataset.tier_read_error_samples:
        zh_print(f"{split_name} tier读取失败样本预览 Top5:")
        for row in dataset.tier_read_error_samples[:5]:
            zh_print(f"  - utt={row['utt_id']} | rel={row['relative_path']} | err={row['error_type']} | msg={row['error_message']} | tiers={row['available_tier_names']}")


def get_run_dir(output_root: str | Path) -> Path:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (output_root / "latest_run.txt").write_text(str(run_dir), encoding="utf-8")
    return run_dir


def get_latest_run_dir(output_root: str | Path) -> Path:
    output_root = Path(output_root)
    pointer = output_root / "latest_run.txt"
    if not pointer.exists():
        raise FileNotFoundError(f"未找到 latest_run.txt: {pointer}")
    run_dir = Path(pointer.read_text(encoding="utf-8").strip())
    if not run_dir.exists():
        raise FileNotFoundError(f"latest_run.txt 指向的目录不存在: {run_dir}")
    return run_dir
