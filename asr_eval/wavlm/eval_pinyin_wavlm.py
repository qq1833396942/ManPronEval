# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch

from common_pinyin_cls import (
    build_collate_fn,
    build_loader,
    get_latest_run_dir,
    load_audio_processor,
    load_config,
    load_vocab,
    PinyinFolderDataset,
    save_dataset_artifacts,
    save_json,
    save_jsonl,
    SingleSyllableWavLM,
    split_pinyin,
    zh_print,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json", help="配置文件路径，默认同目录下 config.json")
    return parser.parse_args()


def main():
    args = parse_args()
    cli_cfg = load_config(args.config)

    ckpt_dir = Path(cli_cfg["eval_ckpt_dir"]).resolve() if str(cli_cfg.get("eval_ckpt_dir", "")).strip() else get_latest_run_dir(cli_cfg["output_root"])
    zh_print(f"本次评估使用的运行目录: {ckpt_dir}")

    snapshot_path = ckpt_dir / "config_snapshot.json"
    if snapshot_path.exists():
        cfg = json.loads(snapshot_path.read_text(encoding="utf-8"))
        zh_print("检测到训练时保存的 config_snapshot.json，将优先复用该配置。")
    else:
        cfg = cli_cfg
        zh_print("未检测到 config_snapshot.json，将使用当前 config.json。")

    device = str(cfg["device"])
    vocab, inv_vocab = load_vocab(cfg["vocab_path"])

    processor, processor_info = load_audio_processor(cfg["base_model"])
    collate_fn = build_collate_fn(processor, sample_rate=int(cfg["sample_rate"]))
    zh_print(f"音频处理器: {processor_info['processor_class']} | 加载方式: {processor_info['load_mode']}")

    test_ds = PinyinFolderDataset(cfg["test_dir"], vocab, cfg, "test")
    save_dataset_artifacts(ckpt_dir, "test", test_ds, int(cfg["print_top_label_count"]))
    save_jsonl(ckpt_dir / "test_manifest.jsonl", test_ds.samples)

    test_loader = build_loader(
        dataset=test_ds,
        batch_size=int(cfg["eval_batch_size"]),
        shuffle=False,
        num_workers=int(cfg["num_workers"]),
        collate_fn=collate_fn,
        weighted=False,
    )

    model = SingleSyllableWavLM(cfg, vocab_size=len(vocab)).to(device)
    ckpt = torch.load(ckpt_dir / "best.pt", map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    zh_print(f"模型已加载完成。最佳 epoch = {ckpt.get('epoch', -1)}")

    results_tsv = ckpt_dir / "test_results.tsv"
    start_all = time.time()

    used = 0
    cer_wrong = 0
    per_wrong = 0
    per_num = 0
    per_den = 0

    progress_every = max(1, int(cfg.get("test_progress_every_batches", 20)))

    with results_tsv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["utt_id", "relative_path", "ref_text", "ref_pinyin", "pred_pinyin", "ACC_strict", "CER_strict", "ref_split", "pred_split", "PER_sample", "PER_strict"])

        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader, start=1):
                if batch is None:
                    continue

                input_values = batch["input_values"].to(device)
                encoder_attention_mask = batch["encoder_attention_mask"].to(device) if batch["encoder_attention_mask"] is not None else None
                pooling_attention_mask = batch["pooling_attention_mask"].to(device)

                logits = model(
                    input_values=input_values,
                    encoder_attention_mask=encoder_attention_mask,
                    pooling_attention_mask=pooling_attention_mask,
                )
                pred_ids = logits.argmax(dim=-1).detach().cpu().tolist()
                pred_texts = [inv_vocab.get(int(i), "[UNK]") for i in pred_ids]

                for i in range(len(pred_texts)):
                    used += 1
                    ref_py = batch["ref_pinyins"][i]
                    pred_py = pred_texts[i]

                    cer_s = 0 if ref_py == pred_py else 1
                    cer_wrong += cer_s

                    ref_split = split_pinyin(ref_py)
                    pred_split = split_pinyin(pred_py)

                    err = 0
                    for r, p in zip(ref_split, pred_split):
                        if r != p:
                            err += 1
                    err += abs(len(ref_split) - len(pred_split))

                    per_sample = err / len(ref_split) if ref_split else 0.0
                    per_num += err
                    per_den += len(ref_split)

                    per_s = 0 if ref_split == pred_split else 1
                    per_wrong += per_s

                    writer.writerow([
                        batch["utt_ids"][i],
                        batch["relative_paths"][i],
                        batch["ref_texts"][i],
                        ref_py,
                        pred_py,
                        1 - cer_s,
                        cer_s,
                        " + ".join(ref_split),
                        " + ".join(pred_split),
                        f"{per_sample:.6f}",
                        per_s,
                    ])

                if batch_idx % progress_every == 0 or batch_idx == len(test_loader):
                    zh_print(f"测试进度: 已完成 {batch_idx}/{len(test_loader)} 个 batch，当前已统计 {used} 条样本。")

    summary = {
        "num_test_used": used,
        "pinyin_acc": 1.0 - (cer_wrong / used if used else 0.0),
        "CER_strict": (cer_wrong / used if used else 0.0),
        "PER_micro": (per_num / per_den) if per_den else 0.0,
        "PER_strict": (per_wrong / used if used else 0.0),
        "per_micro_num": per_num,
        "per_micro_den": per_den,
        "best_epoch": ckpt.get("epoch", -1),
        "processor_info": processor_info,
    }
    save_json(ckpt_dir / "test_summary.json", summary)

    summary_zh = "\n".join([
        "测试完成，中文摘要如下：",
        f"测试样本数: {used}",
        f"拼音准确率: {summary['pinyin_acc']:.6f}",
        f"CER_strict: {summary['CER_strict']:.6f}",
        f"PER_micro: {summary['PER_micro']:.6f}",
        f"PER_strict: {summary['PER_strict']:.6f}",
        f"结果明细: {results_tsv}",
        f"结果汇总: {ckpt_dir / 'test_summary.json'}",
        f"总耗时(秒): {time.time() - start_all:.1f}",
    ])
    (ckpt_dir / "test_summary_zh.txt").write_text(summary_zh, encoding="utf-8")

    zh_print("========== 测试完成 ==========")
    for line in summary_zh.splitlines():
        zh_print(line)


if __name__ == "__main__":
    main()
