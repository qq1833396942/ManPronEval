# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn

from common_pinyin_cls import (
    append_tsv,
    build_collate_fn,
    build_loader,
    calculate_cer,
    calculate_per,
    create_probe_loader,
    evaluate_model,
    get_run_dir,
    load_audio_processor,
    load_config,
    load_vocab,
    PinyinFolderDataset,
    print_dataset_summary,
    print_model_summary,
    save_dataset_artifacts,
    save_json,
    save_jsonl,
    set_seed,
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
    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))

    device = str(cfg["device"])
    run_dir = get_run_dir(cfg["output_root"])
    zh_print(f"当前运行目录: {run_dir}")
    save_json(run_dir / "config_snapshot.json", cfg)

    vocab, inv_vocab = load_vocab(cfg["vocab_path"])
    zh_print(f"词表大小: {len(vocab)}")

    processor, processor_info = load_audio_processor(cfg["base_model"])
    save_json(run_dir / "processor_info.json", processor_info)
    zh_print(f"音频处理器: {processor_info['processor_class']} | 加载方式: {processor_info['load_mode']}")

    collate_fn = build_collate_fn(processor, sample_rate=int(cfg["sample_rate"]))

    train_ds = PinyinFolderDataset(cfg["train_dir"], vocab, cfg, "train")
    val_ds = PinyinFolderDataset(cfg["val_dir"], vocab, cfg, "val")

    print_dataset_summary("train", train_ds, int(cfg["print_top_label_count"]))
    print_dataset_summary("val", val_ds, int(cfg["print_top_label_count"]))

    save_dataset_artifacts(run_dir, "train", train_ds, int(cfg["print_top_label_count"]))
    save_dataset_artifacts(run_dir, "val", val_ds, int(cfg["print_top_label_count"]))
    save_jsonl(run_dir / "train_manifest.jsonl", train_ds.samples)
    save_jsonl(run_dir / "val_manifest.jsonl", val_ds.samples)
    save_json(run_dir / "vocab_snapshot.json", vocab)

    train_loader = build_loader(
        dataset=train_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=not bool(cfg.get("balanced_sampling", False)),
        num_workers=int(cfg["num_workers"]),
        collate_fn=collate_fn,
        weighted=bool(cfg.get("balanced_sampling", False)),
    )
    val_loader = build_loader(
        dataset=val_ds,
        batch_size=int(cfg["eval_batch_size"]),
        shuffle=False,
        num_workers=int(cfg["num_workers"]),
        collate_fn=collate_fn,
        weighted=False,
    )
    probe_loader = create_probe_loader(
        val_dataset=val_ds,
        batch_size=int(cfg["eval_batch_size"]),
        num_workers=int(cfg["num_workers"]),
        collate_fn=collate_fn,
        probe_num_samples=int(cfg["probe_num_val_samples"]),
    )

    model = SingleSyllableWavLM(cfg, vocab_size=len(vocab)).to(device)
    param_stats = print_model_summary(model)
    save_json(
        run_dir / "model_summary.json",
        {
            "base_model": cfg["base_model"],
            "pooling_mode": cfg["pooling_mode"],
            "param_stats": param_stats,
            "lora_summary": getattr(model, "lora_summary", {}),
            "processor_info": processor_info,
        },
    )

    lora_params = [p for name, p in model.named_parameters() if p.requires_grad and "lora_" in name]
    classifier_params = [p for p in model.classifier.parameters() if p.requires_grad]

    optimizer_groups = []
    if classifier_params:
        optimizer_groups.append({"params": classifier_params, "lr": float(cfg["classifier_lr"]), "name": "classifier"})
    if lora_params:
        optimizer_groups.append({"params": lora_params, "lr": float(cfg["encoder_lr"]), "name": "lora"})

    if not optimizer_groups:
        raise RuntimeError("当前没有可训练参数，请检查 LoRA 与分类头配置。")

    optimizer = torch.optim.AdamW(
        [{"params": g["params"], "lr": g["lr"]} for g in optimizer_groups],
        weight_decay=float(cfg["weight_decay"]),
    )
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.get("fp16", True)) and device.startswith("cuda"))

    zh_print("========== 优化器设置 ==========")
    for g in optimizer_groups:
        num_params = sum(p.numel() for p in g["params"])
        zh_print(f"参数组 = {g['name']}, 学习率 = {g['lr']}, 参数量 = {num_params}")

    global_step = 0
    best_val_acc = -1.0
    best_epoch = -1
    best_path = run_dir / "best.pt"
    last_path = run_dir / "last.pt"

    history_header = ["epoch", "avg_train_loss", "val_acc", "val_cer", "val_per", "num_val_used", "epoch_seconds"]
    probe_header = [
        "global_step", "epoch", "batch_idx", "avg_loss", "batch_acc", "batch_cer", "batch_per",
        "probe_val_acc", "probe_val_cer", "probe_val_per", "probe_val_num_samples",
        "example_ref", "example_pred", "example_ref_split", "example_pred_split",
        "example_conf", "mean_seconds", "max_seconds", "classifier_lr", "encoder_or_lora_lr"
    ]

    zh_print("========== 开始训练 ==========")
    start_all = time.time()

    for epoch in range(1, int(cfg["epochs"]) + 1):
        model.train()
        running_loss = 0.0
        seen_batches = 0
        epoch_start = time.time()

        zh_print(f"---- 第 {epoch}/{cfg['epochs']} 轮训练开始 ----")
        for batch_idx, batch in enumerate(train_loader, start=1):
            if batch is None:
                continue

            input_values = batch["input_values"].to(device)
            encoder_attention_mask = batch["encoder_attention_mask"].to(device) if batch["encoder_attention_mask"] is not None else None
            pooling_attention_mask = batch["pooling_attention_mask"].to(device)
            label_ids = batch["label_ids"].to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                logits = model(
                    input_values=input_values,
                    encoder_attention_mask=encoder_attention_mask,
                    pooling_attention_mask=pooling_attention_mask,
                )
                loss = criterion(logits, label_ids)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip"]))
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            seen_batches += 1
            running_loss += float(loss.item())

            if global_step % int(cfg["probe_every_steps"]) == 0:
                pred_ids = logits.argmax(dim=-1).detach().cpu().tolist()
                ref_ids = label_ids.detach().cpu().tolist()
                pred_texts = [inv_vocab.get(int(i), "[UNK]") for i in pred_ids]
                ref_texts = [inv_vocab.get(int(i), "[UNK]") for i in ref_ids]

                batch_acc = sum(1 for r, p in zip(ref_texts, pred_texts) if r == p) / max(1, len(ref_texts))
                batch_cer = calculate_cer(ref_texts, pred_texts)
                batch_per = calculate_per(ref_texts, pred_texts)

                probe_val = {"pinyin_acc": -1.0, "cer": -1.0, "per": -1.0, "num_samples": 0}
                if probe_loader is not None:
                    probe_val = evaluate_model(model, probe_loader, device, inv_vocab)

                probs = torch.softmax(logits.detach(), dim=-1)
                conf = float(probs.max(dim=-1).values.mean().detach().cpu().item())
                mean_seconds = sum(batch["lengths"]) / max(1, len(batch["lengths"])) / float(cfg["sample_rate"])
                max_seconds = max(batch["lengths"]) / float(cfg["sample_rate"])

                classifier_lr = optimizer.param_groups[0]["lr"]
                encoder_lr = optimizer.param_groups[1]["lr"] if len(optimizer.param_groups) > 1 else 0.0

                ref_show = ref_texts[0] if ref_texts else ""
                pred_show = pred_texts[0] if pred_texts else ""
                ref_split_show = " + ".join(split_pinyin(ref_show)) if ref_show else ""
                pred_split_show = " + ".join(split_pinyin(pred_show)) if pred_show else ""

                append_tsv(
                    run_dir / "probe_train_log.tsv",
                    probe_header,
                    [
                        global_step,
                        epoch,
                        batch_idx,
                        f"{running_loss / max(1, seen_batches):.6f}",
                        f"{batch_acc:.6f}",
                        f"{batch_cer:.6f}",
                        f"{batch_per:.6f}",
                        f"{probe_val['pinyin_acc']:.6f}",
                        f"{probe_val['cer']:.6f}",
                        f"{probe_val['per']:.6f}",
                        probe_val["num_samples"],
                        ref_show,
                        pred_show,
                        ref_split_show,
                        pred_split_show,
                        f"{conf:.6f}",
                        f"{mean_seconds:.4f}",
                        f"{max_seconds:.4f}",
                        f"{classifier_lr:.8f}",
                        f"{encoder_lr:.8f}",
                    ],
                )

                zh_print(
                    f"[探针] step={global_step} | epoch={epoch} batch={batch_idx}/{len(train_loader)} | "
                    f"平均loss={running_loss / max(1, seen_batches):.4f} | "
                    f"当前batch acc={batch_acc:.4f} cer={batch_cer:.4f} per={batch_per:.4f} | "
                    f"验证小探针 acc={probe_val['pinyin_acc']:.4f} cer={probe_val['cer']:.4f} per={probe_val['per']:.4f} | "
                    f"lr(cls={classifier_lr:.8f}, enc={encoder_lr:.8f}) | "
                    f"示例: ref={ref_show} ({ref_split_show}) | pred={pred_show} ({pred_split_show}) | conf={conf:.4f}"
                )

        avg_train_loss = running_loss / max(1, seen_batches)
        val_metrics = evaluate_model(model, val_loader, device, inv_vocab)
        epoch_seconds = time.time() - epoch_start

        append_tsv(
            run_dir / "train_history.tsv",
            history_header,
            [
                epoch,
                f"{avg_train_loss:.6f}",
                f"{val_metrics['pinyin_acc']:.6f}",
                f"{val_metrics['cer']:.6f}",
                f"{val_metrics['per']:.6f}",
                val_metrics["num_samples"],
                f"{epoch_seconds:.2f}",
            ],
        )

        zh_print(
            f"[整轮验证] epoch={epoch} | avg_train_loss={avg_train_loss:.4f} | "
            f"val_acc={val_metrics['pinyin_acc']:.4f} | val_cer={val_metrics['cer']:.4f} | "
            f"val_per={val_metrics['per']:.4f} | val_num={val_metrics['num_samples']} | "
            f"本轮耗时={epoch_seconds:.1f}秒"
        )

        if bool(cfg.get("save_last", True)):
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "val_metrics": val_metrics, "config": cfg, "processor_info": processor_info},
                last_path,
            )

        if val_metrics["pinyin_acc"] > best_val_acc:
            best_val_acc = float(val_metrics["pinyin_acc"])
            best_epoch = epoch
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "val_metrics": val_metrics, "config": cfg, "processor_info": processor_info},
                best_path,
            )
            zh_print(f"出现新的最佳模型，已保存: {best_path}")
            save_json(
                run_dir / "best_info.json",
                {
                    "best_epoch": best_epoch,
                    "best_val_acc": best_val_acc,
                    "best_val_cer": val_metrics["cer"],
                    "best_val_per": val_metrics["per"],
                    "best_path": str(best_path),
                },
            )

    total_seconds = time.time() - start_all
    summary_zh = "\n".join([
        "训练完成",
        f"最佳 epoch: {best_epoch}",
        f"最佳验证集拼音准确率: {best_val_acc:.6f}",
        f"最佳模型路径: {best_path}",
        f"最后一轮模型路径: {last_path}",
        f"总耗时(秒): {total_seconds:.1f}",
        f"运行目录: {run_dir}",
    ])
    (run_dir / "best_info_zh.txt").write_text(summary_zh, encoding="utf-8")
    zh_print("========== 训练完成 ==========")
    for line in summary_zh.splitlines():
        zh_print(line)


if __name__ == "__main__":
    main()
