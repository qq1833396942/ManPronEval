import os
import torch
import torch.nn as nn
import torch.optim as optim
from contextlib import nullcontext
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, f1_score

# 导入适配模块
from dataset_mtl_large import MTLDataset, MTLCollateFn
from model_mtl_large import MultiTaskWhisperLarge
from utils_mtl import load_or_create_vocab, pinyin_error_rate, safe_pearsonr


# ==========================================
# 🛑 早停机制类
# ==========================================
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'早停计数: {self.counter} / {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'验证集 Loss 降低 ({self.val_loss_min:.6f} --> {val_loss:.6f}). 保存模型...')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss


def train_and_validate_mtl_large():
    # ==========================================
    # ⚙️ 1. 配置区
    # ==========================================
    project_dir = Path(__file__).resolve().parent
    local_model_path = os.environ.get(
        "WHISPER_LARGE_MODEL_PATH",
        os.environ.get("WHISPER_MODEL_PATH", "openai/whisper-large-v3"),
    )
    base_dir = Path(os.environ.get("MTL_BASE_DIR", project_dir)).expanduser()
    audio_root = os.environ.get("MTL_AUDIO_ROOT")

    vocab_path = base_dir / 'syllable_vocab.json'
    train_json_path = base_dir / 'metadata_train_mtl.json'
    val_json_path = base_dir / 'metadata_val_mtl.json'
    test_json_path = base_dir / 'metadata_test_mtl.json'

    epochs = 30
    batch_size = 4
    accum_steps = 4
    learning_rate = 1e-4
    num_workers = int(os.environ.get("MTL_NUM_WORKERS", "4"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"🔥 计算设备: {device} | Whisper-Large MTL | AMP: {'on' if use_amp else 'off'}")
    print(f"📁 数据目录: {base_dir}")
    print(f"🤗 模型路径/名称: {local_model_path}")

    # ==========================================
    # 📚 2. 加载数据
    # ==========================================
    pinyin2id = load_or_create_vocab(vocab_path, [train_json_path, val_json_path, test_json_path])
    id2pinyin = {v: k for k, v in pinyin2id.items()}

    train_dataset = MTLDataset(train_json_path, pinyin2id, audio_root=audio_root)
    val_dataset = MTLDataset(val_json_path, pinyin2id, audio_root=audio_root)

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError("训练集或验证集为空，请检查 metadata 文件。")

    whisper_collate = MTLCollateFn(model_path=local_model_path)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=whisper_collate, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=whisper_collate, num_workers=num_workers)

    # ==========================================
    # 🚀 3. 模型与优化器
    # ==========================================
    model = MultiTaskWhisperLarge(model_path=local_model_path, vocab_size=len(pinyin2id)).to(device)

    # 优化器参数保持一致
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    def amp_context():
        return torch.amp.autocast(device_type='cuda') if use_amp else nullcontext()

    def optimizer_step():
        if use_amp:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()

    criterion_apa = nn.MSELoss()
    criterion_mdd = nn.CrossEntropyLoss()
    criterion_asr = nn.CrossEntropyLoss()

    # 初始化早停
    model_save_path = base_dir / "whisper_large_mtl_best.pth"
    early_stopping = EarlyStopping(patience=7, verbose=True)

    # ==========================================
    # 🏃 4. 训练大循环
    # ==========================================
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        pbar_train = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]")

        optimizer.zero_grad()

        for i, batch in enumerate(pbar_train):
            input_features = batch["input_features"].to(device)
            target_pinyin_ids = batch["target_pinyin_ids"].to(device)
            actual_pinyin_ids = batch["actual_pinyin_ids"].to(device)
            mdd_labels = batch["mdd_labels"].to(device)
            targets_scores = batch["scores"].to(device)

            with amp_context():
                apa_scores, mdd_logits, asr_logits = model(input_features, target_pinyin_ids)

                loss_apa = criterion_apa(apa_scores, targets_scores)
                loss_mdd = criterion_mdd(mdd_logits, mdd_labels)
                loss_asr = criterion_asr(asr_logits, actual_pinyin_ids)

                loss = (loss_apa + loss_mdd + loss_asr) / accum_steps

            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (i + 1) % accum_steps == 0:
                optimizer_step()

            train_loss += loss.item() * accum_steps
            pbar_train.set_postfix({'L': f"{loss.item() * accum_steps:.2f}"})

        if len(train_loader) % accum_steps != 0:
            optimizer_step()

        # ==========================================
        # 🎯 5. 验证阶段
        # ==========================================
        model.eval()
        val_loss = 0
        apa_true, apa_pred = [], []
        mdd_true, mdd_pred = [], []
        asr_true_ids, asr_pred_ids = [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} [Val]", leave=False):
                input_features = batch["input_features"].to(device)
                target_pinyin_ids = batch["target_pinyin_ids"].to(device)
                actual_pinyin_ids = batch["actual_pinyin_ids"].to(device)
                mdd_labels = batch["mdd_labels"].to(device)
                targets_scores = batch["scores"].to(device)

                with amp_context():
                    apa_scores, mdd_logits, asr_logits = model(input_features, target_pinyin_ids)

                    v_loss = criterion_apa(apa_scores, targets_scores) + \
                             criterion_mdd(mdd_logits, mdd_labels) + \
                             criterion_asr(asr_logits, actual_pinyin_ids)

                val_loss += v_loss.item()

                apa_true.extend(targets_scores[:, 0].cpu().numpy())
                apa_pred.extend(apa_scores[:, 0].float().cpu().numpy())
                mdd_true.extend(mdd_labels.cpu().numpy())
                mdd_pred.extend(torch.argmax(mdd_logits, dim=1).cpu().numpy())
                asr_true_ids.extend(actual_pinyin_ids.cpu().numpy())
                asr_pred_ids.extend(torch.argmax(asr_logits, dim=1).cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)

        # 指标计算
        val_pcc = safe_pearsonr(pearsonr, apa_true, apa_pred)
        val_mdd_f1 = f1_score(mdd_true, mdd_pred, average='macro', zero_division=0)
        val_asr_acc = accuracy_score(asr_true_ids, asr_pred_ids)
        val_asr_per = pinyin_error_rate(asr_true_ids, asr_pred_ids, id2pinyin)

        print(f"\n✅ [Epoch {epoch + 1} 总结]")
        print(f"   📉 Train Loss: {train_loss / len(train_loader):.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"   🎯 APA PCC: {val_pcc:.4f} | MDD F1: {val_mdd_f1:.4f} | ASR Acc: {val_asr_acc:.4f} | ASR PER: {val_asr_per:.4f}")

        # 检查早停
        early_stopping(avg_val_loss, model, model_save_path)
        if early_stopping.early_stop:
            print("🛑 早停触发，模型性能已不再提升。")
            break


if __name__ == "__main__":
    train_and_validate_mtl_large()
