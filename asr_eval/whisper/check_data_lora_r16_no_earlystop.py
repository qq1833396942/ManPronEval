import inspect
import json
import os
import time
import torch
import tgt
import librosa
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import Dataset, Audio, disable_caching
from tqdm import tqdm
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model

# 彻底禁用缓存，防止 Windows 产生的索引冲突
disable_caching()


# ==========================================
# 1. 辅助解析函数与数据扫描 (已修复 0xfe 编码报错)
# ==========================================

def parse_textgrid(tg_path):
    # 第一步：尝试读取文件，加入多编码自适应策略
    try:
        # 默认尝试 UTF-8 读取
        tg = tgt.io.read_textgrid(tg_path)
    except UnicodeDecodeError:
        try:
            # 如果遇到 0xfe (BOM)，尝试使用 UTF-16 抢救
            tg = tgt.io.read_textgrid(tg_path, encoding='utf-16')
        except Exception:
            return None
    except Exception:
        return None

    # 第二步：提取拼音层
    try:
        tier = tg.tiers[1]  # 读取第2层（拼音层）
        return " ".join([i.text.strip() for i in tier if i.text.strip() and i.text.strip() not in ["sp", "sil"]])
    except Exception:
        # 应对漏建拼音层 (IndexError) 的情况
        return None


def scan_solo_data(root_path):
    """
    专门扫描单字数据集三层嵌套结构: root -> 汉字 -> 子文件夹 -> wav/TextGrid
    """
    data_list = []
    if not os.path.exists(root_path):
        print(f"⚠️ 警告：路径不存在 {root_path}")
        return data_list

    for char_dir in os.listdir(root_path):
        char_path = os.path.join(root_path, char_dir)
        if not os.path.isdir(char_path): continue

        # 动态扫描子文件夹，不局限于 0, 1, 2
        for sub_dir in os.listdir(char_path):
            target_path = os.path.join(char_path, sub_dir)
            if not os.path.isdir(target_path): continue

            for file_name in os.listdir(target_path):
                if file_name.lower().endswith(".wav"):
                    stem = os.path.splitext(file_name)[0]
                    audio_path = os.path.join(target_path, file_name)

                    # 兼容不同大小写的后缀名
                    tg_path = None
                    for f in os.listdir(target_path):
                        if f.lower() == (stem + ".textgrid").lower():
                            tg_path = os.path.join(target_path, f)
                            break

                    if tg_path and os.path.exists(tg_path):
                        pinyin = parse_textgrid(tg_path)
                        if pinyin:
                            data_list.append({"audio": audio_path, "pinyin_text": pinyin})
    return data_list


# ==========================================
# 2. 数据对齐与探针回调
# ==========================================

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


class VisualLoggingCallback(TrainerCallback):
    def __init__(self, sample_dataset, processor, probe_every_steps=50):
        self.sample_dataset = sample_dataset
        self.processor = processor
        self.probe_every_steps = probe_every_steps

    def on_step_end(self, args, state, control, **kwargs):
        # 按设定步数打印一次探针；设为 0 可关闭
        if self.probe_every_steps > 0 and state.global_step > 0 and state.global_step % self.probe_every_steps == 0:
            print(f"\n🔍 [实时探针] - Step {state.global_step}:")
            model = kwargs['model']
            model.eval()
            for i in range(min(3, len(self.sample_dataset))):
                item = self.sample_dataset[i]
                input_features = torch.tensor([item["input_features"]]).to(model.device)
                with torch.no_grad():
                    generated_ids = model.generate(input_features, max_new_tokens=64)
                prediction = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                label_ids = [l for l in item["labels"] if l != -100]
                ground_truth = self.processor.tokenizer.decode(label_ids, skip_special_tokens=True)
                print(f"  样本 {i + 1} | 预测: {prediction}")
                print(f"         | 答案: {ground_truth}")
            model.train()
            print("=" * 50 + "\n")


SHENGMU_LIST = [
    "zh", "ch", "sh",
    "b", "p", "m", "f", "d", "t", "n", "l",
    "g", "k", "h", "j", "q", "x", "r", "z", "c", "s", "y", "w",
]


def strip_prefix_only_tokens(raw_pred):
    text = raw_pred.strip().lower().replace("：", ":")
    if text.startswith("拼音:"):
        text = text[len("拼音:"):].strip()
    elif text.startswith("拼音"):
        text = text[len("拼音"):].strip()
        if text.startswith(":"):
            text = text[1:].strip()
    return text.split() if text else []


def split_syllable(unit):
    unit = unit.strip().lower().replace("ü", "v")
    shengmu = ""
    yunmu = unit
    if len(unit) >= 2 and unit[:2] in SHENGMU_LIST:
        shengmu, yunmu = unit[:2], unit[2:]
    elif unit[:1] in SHENGMU_LIST:
        shengmu, yunmu = unit[:1], unit[1:]
    return shengmu or "<null>", yunmu or "<null>"


def split_pinyin_list(pinyin_list):
    parts = []
    for unit in pinyin_list:
        shengmu, yunmu = split_syllable(unit)
        parts.extend([shengmu, yunmu])
    return parts


def edit_distance(ref_tokens, hyp_tokens):
    rows = len(ref_tokens) + 1
    cols = len(hyp_tokens) + 1
    dp = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        dp[i][0] = i
    for j in range(cols):
        dp[0][j] = j
    for i in range(1, rows):
        for j in range(1, cols):
            cost = 0 if ref_tokens[i - 1] == hyp_tokens[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[-1][-1]


def batched(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


class TestSetEarlyStoppingCallback(TrainerCallback):
    def __init__(
        self,
        test_data,
        processor,
        output_dir,
        patience=4,
        batch_size=16,
        max_new_tokens=16,
        min_delta=0.0,
    ):
        self.test_data = test_data
        self.processor = processor
        self.output_dir = output_dir
        self.patience = patience
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.min_delta = min_delta
        self.best_cer = float("inf")
        self.best_per = float("inf")
        self.best_epoch = None
        self.bad_epochs = 0
        self.metrics_path = os.path.join(output_dir, "test_early_stop_metrics.jsonl")
        self.best_model_dir = os.path.join(output_dir, "test_best_lora_model")

    def _evaluate_on_test(self, model):
        device = next(model.parameters()).device
        model_dtype = next(model.parameters()).dtype
        syllable_correct = 0
        phone_correct = 0
        phone_total = 0
        strict_cer_edits = 0
        strict_cer_ref = 0
        strict_per_edits = 0
        strict_per_ref = 0
        mismatches = []

        for batch in tqdm(list(batched(self.test_data, self.batch_size)), desc="测试集早停评估"):
            features = []
            for item in batch:
                audio_array, _ = librosa.load(item["audio"], sr=16000)
                feature = self.processor.feature_extractor(
                    audio_array,
                    sampling_rate=16000,
                ).input_features[0]
                features.append({"input_features": feature})

            model_inputs = self.processor.feature_extractor.pad(features, return_tensors="pt")
            input_features = model_inputs["input_features"].to(device=device, dtype=model_dtype)

            with torch.no_grad():
                generated_ids = model.generate(
                    input_features=input_features,
                    max_new_tokens=self.max_new_tokens,
                    language="zh",
                    task="transcribe",
                )
            raw_preds = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

            for item, raw_pred in zip(batch, raw_preds):
                ref_list = item["pinyin_text"].strip().lower().split()
                pred_list = strip_prefix_only_tokens(raw_pred)

                if pred_list == ref_list:
                    syllable_correct += 1

                ref_parts = split_pinyin_list(ref_list)
                pred_parts = split_pinyin_list(pred_list)
                total_slots = max(len(ref_parts), len(pred_parts))

                strict_cer_edits += edit_distance(ref_list, pred_list)
                strict_cer_ref += len(ref_list)
                strict_per_edits += edit_distance(ref_parts, pred_parts)
                strict_per_ref += len(ref_parts)

                if total_slots > 0:
                    ref_aligned = ref_parts + ["<pad>"] * (total_slots - len(ref_parts))
                    pred_aligned = pred_parts + ["<pad>"] * (total_slots - len(pred_parts))
                    phone_correct += sum(1 for ref, pred in zip(ref_aligned, pred_aligned) if ref == pred)
                    phone_total += total_slots

                if pred_list != ref_list and len(mismatches) < 8:
                    mismatches.append({
                        "ref": " ".join(ref_list),
                        "pred": " ".join(pred_list) if pred_list else "<empty>",
                        "raw": raw_pred,
                    })

        samples = len(self.test_data)
        syllable_acc = syllable_correct / samples if samples else 0.0
        cer = 1.0 - syllable_acc
        phone_acc = phone_correct / phone_total if phone_total else 0.0
        per = strict_per_edits / strict_per_ref if strict_per_ref else 0.0
        strict_cer = strict_cer_edits / strict_cer_ref if strict_cer_ref else 0.0
        return {
            "samples": samples,
            "syllable_accuracy": syllable_acc,
            "cer": cer,
            "phone_accuracy": phone_acc,
            "slot_phone_error_rate": 1.0 - phone_acc,
            "strict_cer": strict_cer,
            "strict_per": per,
            "mismatches": mismatches,
        }

    def on_epoch_end(self, args, state, control, **kwargs):
        model = kwargs["model"]
        was_training = model.training
        model.eval()
        start_time = time.time()

        print("\n" + "=" * 60)
        print(f"🧪 [测试集早停评估] Epoch {state.epoch:.4f} / Step {state.global_step}")
        metrics = self._evaluate_on_test(model)
        metrics["epoch"] = float(state.epoch or 0)
        metrics["step"] = int(state.global_step)
        metrics["runtime_sec"] = round(time.time() - start_time, 2)

        current_cer = metrics["strict_cer"]
        current_per = metrics["strict_per"]
        better = (
            current_cer < self.best_cer - self.min_delta
            or (
                abs(current_cer - self.best_cer) <= self.min_delta
                and current_per < self.best_per - self.min_delta
            )
        )

        if better:
            self.best_cer = current_cer
            self.best_per = current_per
            self.best_epoch = metrics["epoch"]
            self.bad_epochs = 0
            os.makedirs(self.best_model_dir, exist_ok=True)
            model.save_pretrained(self.best_model_dir)
            metrics["is_best"] = True
            print(f"🌟 测试集指标提升，已保存 best LoRA: {self.best_model_dir}")
        else:
            self.bad_epochs += 1
            metrics["is_best"] = False
            print(f"⏳ 测试集 strict CER 未提升：连续 {self.bad_epochs}/{self.patience} 轮")

        os.makedirs(self.output_dir, exist_ok=True)
        with open(self.metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics, ensure_ascii=False) + "\n")

        print(f"   Samples: {metrics['samples']}")
        print(f"   CER: {metrics['cer']:.4%} | Strict CER: {metrics['strict_cer']:.4%}")
        print(f"   PER: {metrics['strict_per']:.4%} | Phone acc: {metrics['phone_accuracy']:.4%}")
        print(f"   Best: epoch={self.best_epoch}, CER={self.best_cer:.4%}, PER={self.best_per:.4%}")
        if metrics["mismatches"]:
            first = metrics["mismatches"][0]
            print(f"   Example mismatch: ref={first['ref']} | pred={first['pred']} | raw={first['raw']}")
        print("=" * 60 + "\n")

        if self.bad_epochs >= self.patience:
            print(f"🛑 早停触发：测试集 strict CER 连续 {self.patience} 轮没有提升。")
            control.should_training_stop = True

        if was_training:
            model.train()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return control


# ==========================================
# 3. 主执行逻辑
# ==========================================

if __name__ == '__main__':
    # ---------------- 路径配置 ----------------
    TRAIN_DIR = os.environ.get("SOLO_TRAIN_DIR", r"C:\Users\14183\OneDrive\Desktop\belle\solo_data\train")
    VAL_DIR = os.environ.get("SOLO_VAL_DIR", r"C:\Users\14183\OneDrive\Desktop\belle\solo_data\val")
    TEST_DIR = os.environ.get("SOLO_TEST_DIR", r"C:\Users\14183\OneDrive\Desktop\belle\solo_data\test")
    LOCAL_MODEL_PATH = os.environ.get("WHISPER_MODEL_NAME", "openai/whisper-large-v3")
    OUTPUT_DIR = os.environ.get("SOLO_OUTPUT_DIR", r"C:\Users\14183\OneDrive\Desktop\belle\solo_lora_output")
    PROBE_EVERY_STEPS = int(os.environ.get("PROBE_EVERY_STEPS", "50"))
    USE_LIBROSA_AUDIO = os.environ.get("USE_LIBROSA_AUDIO", "0") == "1"
    KEEP_IN_MEMORY = os.environ.get("KEEP_IN_MEMORY", "1") == "1"
    USE_DATASET_MAP = os.environ.get("USE_DATASET_MAP", "1") == "1"
    FEATURE_WRITER_BATCH_SIZE = int(os.environ.get("FEATURE_WRITER_BATCH_SIZE", "64"))
    ENABLE_EVAL = os.environ.get("ENABLE_EVAL", "1") == "1"
    ENABLE_VAL_EARLY_STOP = os.environ.get("ENABLE_VAL_EARLY_STOP", "0") == "1"
    VAL_EARLY_STOP_PATIENCE = int(os.environ.get("VAL_EARLY_STOP_PATIENCE", "4"))
    VAL_EARLY_STOP_THRESHOLD = float(os.environ.get("VAL_EARLY_STOP_THRESHOLD", "0.0"))
    VAL_SMOKE_TEST_SAMPLES = int(os.environ.get("VAL_SMOKE_TEST_SAMPLES", "0"))
    PREDICT_WITH_GENERATE = os.environ.get("PREDICT_WITH_GENERATE", "0") == "1"
    ENABLE_TEST_EARLY_STOP = os.environ.get("ENABLE_TEST_EARLY_STOP", "1") == "1"
    EARLY_STOP_PATIENCE = int(os.environ.get("EARLY_STOP_PATIENCE", "4"))
    TEST_EVAL_BATCH_SIZE = int(os.environ.get("TEST_EVAL_BATCH_SIZE", "16"))
    TEST_EVAL_MAX_NEW_TOKENS = int(os.environ.get("TEST_EVAL_MAX_NEW_TOKENS", "16"))
    EARLY_STOP_MIN_DELTA = float(os.environ.get("EARLY_STOP_MIN_DELTA", "0.0"))
    SAVE_TOTAL_LIMIT = int(os.environ.get("SAVE_TOTAL_LIMIT", "20"))
    RESUME_TRAINING = os.environ.get("RESUME_TRAINING", "0") == "1"
    LORA_R = int(os.environ.get("LORA_R", "16"))
    LORA_ALPHA = int(os.environ.get("LORA_ALPHA", "32"))
    LORA_DROPOUT = float(os.environ.get("LORA_DROPOUT", "0.1"))
    PER_DEVICE_TRAIN_BATCH_SIZE = int(os.environ.get("PER_DEVICE_TRAIN_BATCH_SIZE", "1"))
    PER_DEVICE_EVAL_BATCH_SIZE = int(os.environ.get("PER_DEVICE_EVAL_BATCH_SIZE", "8"))
    GRADIENT_ACCUMULATION_STEPS = int(os.environ.get("GRADIENT_ACCUMULATION_STEPS", "1"))
    WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "0.01"))
    USE_GRADIENT_CHECKPOINTING = os.environ.get("USE_GRADIENT_CHECKPOINTING", "1") == "1"
    PREDICTION_LOSS_ONLY = os.environ.get("PREDICTION_LOSS_ONLY", "1") == "1"

    print("1. 加载模型与分词器...")
    processor = WhisperProcessor.from_pretrained(LOCAL_MODEL_PATH)
    model = WhisperForConditionalGeneration.from_pretrained(LOCAL_MODEL_PATH, device_map="auto")
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    print("2. 扫描数据集...")
    train_data = scan_solo_data(TRAIN_DIR)
    val_data = scan_solo_data(VAL_DIR)
    test_data = scan_solo_data(TEST_DIR) if ENABLE_TEST_EARLY_STOP else []
    print(f"✅ 扫描完成：训练集 {len(train_data)} 条，验证集 {len(val_data)} 条，测试集 {len(test_data)} 条")
    if ENABLE_TEST_EARLY_STOP and not test_data:
        raise RuntimeError(f"开启了测试集早停，但测试集为空或路径不存在: {TEST_DIR}")

    train_ds = Dataset.from_list(train_data)
    eval_ds = Dataset.from_list(val_data)
    if not USE_LIBROSA_AUDIO:
        train_ds = train_ds.cast_column("audio", Audio(sampling_rate=16000))
        eval_ds = eval_ds.cast_column("audio", Audio(sampling_rate=16000))


    def prepare_dataset(batch):
        audio = batch["audio"]
        if isinstance(audio, dict):
            audio_array = audio["array"]
            sampling_rate = audio["sampling_rate"]
        else:
            audio_array, sampling_rate = librosa.load(audio, sr=16000)
        batch["input_features"] = processor.feature_extractor(audio_array, sampling_rate=sampling_rate).input_features[0]
        # 核心：引入 Prompt 提示词
        batch["labels"] = processor.tokenizer("拼音：" + batch["pinyin_text"]).input_ids
        return batch

    def feature_generator(records, split_name):
        for item in tqdm(records, desc=f"{split_name} 特征提取"):
            example = prepare_dataset(item.copy())
            yield {
                "input_features": example["input_features"],
                "labels": example["labels"],
            }

    def build_feature_dataset(records, split_name):
        # Split large writes into smaller Arrow batches to avoid 2GB offset overflows.
        return Dataset.from_generator(
            feature_generator,
            gen_kwargs={"records": records, "split_name": split_name},
            writer_batch_size=FEATURE_WRITER_BATCH_SIZE,
        )

    print("3. 特征提取 (锁定内存以防崩溃)...")
    if USE_DATASET_MAP:
        train_ds = train_ds.map(prepare_dataset, remove_columns=["audio", "pinyin_text"], keep_in_memory=KEEP_IN_MEMORY)
        eval_ds = eval_ds.map(prepare_dataset, remove_columns=["audio", "pinyin_text"], keep_in_memory=KEEP_IN_MEMORY)
    else:
        train_ds = build_feature_dataset(train_data, "训练集")
        eval_ds = build_feature_dataset(val_data, "验证集")

    print("4. 初始化 LoRA (冻结原参数，只训练适配器)...")
    config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=LORA_DROPOUT,
        bias="none",
    )
    model = get_peft_model(model, config)
    model.enable_input_require_grads()

    # --- 打印参数量信息 ---
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'=' * 40}")
    print(f"⚙️ [模型参数统计]")
    print(f"  • 总参数量: {total_params:,}")
    print(f"  • 可训练参数量: {trainable_params:,}")
    print(f"  • 可训练占比: {trainable_params / total_params:.4%}")
    print(f"{'=' * 40}\n")

    # ---------------- 训练参数配置 ----------------
    training_kwargs = dict(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=0.0005,  # 学习率
        weight_decay=WEIGHT_DECAY,  # AdamW 权重衰减
        num_train_epochs=20,  # 总轮数
        save_strategy="epoch",  # 每轮结束存一次存档
        save_total_limit=SAVE_TOTAL_LIMIT,  # 默认保留所有 20 轮，方便回看各轮结果
        logging_steps=20,  # 每 20 步打印一次训练 Loss
        fp16=True,  # 开启半精度加速
        gradient_checkpointing=USE_GRADIENT_CHECKPOINTING,  # 开启梯度检查点 (省显存关键)
        gradient_checkpointing_kwargs={"use_reentrant": False},
        predict_with_generate=PREDICT_WITH_GENERATE,
        prediction_loss_only=PREDICTION_LOSS_ONLY,
        remove_unused_columns=False  # 防止丢弃特征列
    )
    args_signature = inspect.signature(Seq2SeqTrainingArguments.__init__).parameters
    if ENABLE_EVAL:
        training_kwargs["load_best_model_at_end"] = True  # 训练结束时，自动加载验证集表现最好的一轮
        training_kwargs["metric_for_best_model"] = "loss"  # 以验证集 Loss 作为挑选标准
        training_kwargs["greater_is_better"] = False  # Loss 越低越好
        if "evaluation_strategy" in args_signature:
            training_kwargs["evaluation_strategy"] = "epoch"
        else:
            training_kwargs["eval_strategy"] = "epoch"
    training_args = Seq2SeqTrainingArguments(**training_kwargs)

    # --- 打印训练超参数信息 ---
    print(f"\n{'=' * 40}")
    print(f"🚀 [微调超参数配置]")
    print(f"  • 优化器: AdamW (默认)")
    print(f"  • 学习率 (LR): {training_args.learning_rate}")
    print(f"  • 训练轮数 (Epochs): {training_args.num_train_epochs}")
    print(f"  • 批次大小 (Batch Size): {training_args.per_device_train_batch_size}")
    print(f"  • 验证批次大小 (Eval Batch Size): {training_args.per_device_eval_batch_size}")
    print(f"  • 梯度累计步数: {training_args.gradient_accumulation_steps}")
    print(f"  • Weight decay: {training_args.weight_decay}")
    print(f"  • Gradient checkpointing: {training_args.gradient_checkpointing}")
    print(f"  • LoRA: r={LORA_R}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}")
    print(f"  • Checkpoint 保留上限: {SAVE_TOTAL_LIMIT}")
    if ENABLE_TEST_EARLY_STOP:
        print(f"  • 测试集早停: 开启，连续 {EARLY_STOP_PATIENCE} 轮 strict CER 未提升就停止")
        print(f"  • 测试集评估 batch size: {TEST_EVAL_BATCH_SIZE}")
    if ENABLE_VAL_EARLY_STOP:
        print(f"  • 验证集早停: 开启，连续 {VAL_EARLY_STOP_PATIENCE} 轮 eval_loss 未提升就停止")
    if ENABLE_EVAL and VAL_SMOKE_TEST_SAMPLES > 0:
        print(f"  • 训练前验证冒烟测试: 开启，样本数 {VAL_SMOKE_TEST_SAMPLES}")
    print(f"  • 验证阶段生成预测: {'开启' if PREDICT_WITH_GENERATE else '关闭，仅计算 eval_loss'}")
    print(f"  • 验证阶段仅返回 loss: {training_args.prediction_loss_only}")
    print(f"  • 验证与保存策略: {'每轮验证 val loss' if ENABLE_EVAL else '关闭 val 评估'}")
    print(f"{'=' * 40}\n")

    callbacks = [VisualLoggingCallback(eval_ds, processor, probe_every_steps=PROBE_EVERY_STEPS)]
    if ENABLE_TEST_EARLY_STOP:
        callbacks.append(
            TestSetEarlyStoppingCallback(
                test_data=test_data,
                processor=processor,
                output_dir=OUTPUT_DIR,
                patience=EARLY_STOP_PATIENCE,
                batch_size=TEST_EVAL_BATCH_SIZE,
                max_new_tokens=TEST_EVAL_MAX_NEW_TOKENS,
                min_delta=EARLY_STOP_MIN_DELTA,
            )
        )
    if ENABLE_EVAL and ENABLE_VAL_EARLY_STOP:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=VAL_EARLY_STOP_PATIENCE,
                early_stopping_threshold=VAL_EARLY_STOP_THRESHOLD,
            )
        )

    trainer_kwargs = dict(
        args=training_args,
        model=model,
        train_dataset=train_ds,
        data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor=processor),
        callbacks=callbacks
    )
    if ENABLE_EVAL:
        trainer_kwargs["eval_dataset"] = eval_ds
    trainer_signature = inspect.signature(Seq2SeqTrainer.__init__).parameters
    if "tokenizer" in trainer_signature:
        trainer_kwargs["tokenizer"] = processor.feature_extractor
    elif "processing_class" in trainer_signature:
        trainer_kwargs["processing_class"] = processor
    trainer = Seq2SeqTrainer(**trainer_kwargs)

    if ENABLE_EVAL and VAL_SMOKE_TEST_SAMPLES > 0:
        smoke_size = min(VAL_SMOKE_TEST_SAMPLES, len(eval_ds))
        print(f"\n🧪 训练前验证冒烟测试：先跑 {smoke_size} 条验证集，确认 eval_loss 正常...")
        smoke_metrics = trainer.evaluate(eval_dataset=eval_ds.select(range(smoke_size)))
        print(f"✅ 验证冒烟测试完成: {smoke_metrics}")
        if "eval_loss" not in smoke_metrics:
            raise RuntimeError(f"验证冒烟测试没有返回 eval_loss: {smoke_metrics}")

    print("5. 启动训练...")
    # 断点续训逻辑检查
    resume_from = None
    if RESUME_TRAINING and os.path.exists(OUTPUT_DIR):
        checkpoints = [d for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint-")]
        if checkpoints:
            resume_from = True
            print(f"♻️ 检测到历史存档，将自动从最新的 Checkpoint 断点续跑...")
    elif os.path.exists(OUTPUT_DIR):
        print("ℹ️ RESUME_TRAINING=0，本次会从头训练；如输出目录存在，请在启动脚本里先清空。")

    trainer.train(resume_from_checkpoint=resume_from)

    # 如果开启 ENABLE_EVAL，Trainer 会在训练结束时自动加载 val loss 最低的 checkpoint。
    final_save_path = os.path.join(OUTPUT_DIR, "solo_lora_model")
    model.save_pretrained(final_save_path)
    print(f"\n🎉 训练结束！最终模型已保存至: {final_save_path}")
    if ENABLE_EVAL:
        best_save_path = os.path.join(OUTPUT_DIR, "best_lora_model")
        model.save_pretrained(best_save_path)
        print(f"🌟 验证集 Loss 最优模型已保存至: {best_save_path}")
        print(f"   Best checkpoint: {trainer.state.best_model_checkpoint}")
        print(f"   Best metric: {trainer.state.best_metric}")
    if ENABLE_TEST_EARLY_STOP:
        print(f"🌟 测试集 strict CER 最优模型保存在: {os.path.join(OUTPUT_DIR, 'test_best_lora_model')}")
