import os
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, f1_score

from dataset_mtl_base import MTLDataset, MTLCollateFn
from model_mtl_base import MultiTaskWhisper
from utils_mtl import load_or_create_vocab, pinyin_error_rate, safe_pearsonr

def train_and_validate_mtl():
    # ==========================================
    # ⚙️ 1. 配置区
    # ==========================================
    project_dir = Path(__file__).resolve().parent
    local_model_path = os.environ.get(
        "WHISPER_BASE_MODEL_PATH",
        os.environ.get("WHISPER_MODEL_PATH", "openai/whisper-base"),
    )
    base_dir = Path(os.environ.get("MTL_BASE_DIR", project_dir)).expanduser()
    audio_root = os.environ.get("MTL_AUDIO_ROOT")

    vocab_path = base_dir / 'syllable_vocab.json'
    train_json_path = base_dir / 'metadata_train_mtl.json'
    val_json_path = base_dir / 'metadata_val_mtl.json'
    test_json_path = base_dir / 'metadata_test_mtl.json'

    epochs = 30
    batch_size = 16
    num_workers = int(os.environ.get("MTL_NUM_WORKERS", "4"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 计算设备: {device} | Whisper-Base MTL | lr=1e-4")
    print(f"📁 数据目录: {base_dir}")
    print(f"🤗 模型路径/名称: {local_model_path}")

    # ==========================================
    # 📚 2. 加载数据
    # ==========================================
    pinyin2id = load_or_create_vocab(vocab_path, [train_json_path, val_json_path, test_json_path])
    vocab_size = len(pinyin2id)

    # 🌟 建立 id 到 pinyin 的反向映射，用于验证时还原音素计算 PER
    id2pinyin = {v: k for k, v in pinyin2id.items()}

    train_dataset = MTLDataset(train_json_path, pinyin2id, audio_root=audio_root)
    val_dataset = MTLDataset(val_json_path, pinyin2id, audio_root=audio_root)

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError("训练集或验证集为空，请检查 metadata 文件。")

    whisper_collate = MTLCollateFn(model_path=local_model_path)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=whisper_collate,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=whisper_collate,
        num_workers=num_workers,
    )

    # ==========================================
    # 🚀 3. 模型与优化器
    # ==========================================
    model = MultiTaskWhisper(model_path=local_model_path, vocab_size=vocab_size).to(device)

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    criterion_apa = nn.MSELoss()
    criterion_mdd = nn.CrossEntropyLoss()
    criterion_asr = nn.CrossEntropyLoss()

    w_apa, w_mdd, w_asr = 1.0, 1.0, 1.0

    best_val_loss = float('inf')

    # ==========================================
    # 🏃 4. 训练大循环
    # ==========================================
    for epoch in range(epochs):
        model.train()
        train_loss, train_apa_loss, train_mdd_loss, train_asr_loss = 0, 0, 0, 0

        pbar_train = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch in pbar_train:
            input_features = batch["input_features"].to(device)
            target_pinyin_ids = batch["target_pinyin_ids"].to(device)
            actual_pinyin_ids = batch["actual_pinyin_ids"].to(device)
            mdd_labels = batch["mdd_labels"].to(device)
            targets_scores = batch["scores"].to(device)

            optimizer.zero_grad()
            apa_scores, mdd_logits, asr_logits = model(input_features, target_pinyin_ids)

            loss_apa = criterion_apa(apa_scores, targets_scores)
            loss_mdd = criterion_mdd(mdd_logits, mdd_labels)
            loss_asr = criterion_asr(asr_logits, actual_pinyin_ids)

            loss = (w_apa * loss_apa) + (w_mdd * loss_mdd) + (w_asr * loss_asr)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            train_apa_loss += loss_apa.item()
            train_mdd_loss += loss_mdd.item()
            train_asr_loss += loss_asr.item()

            pbar_train.set_postfix({'L': f"{loss.item():.2f}"})

        num_train_batches = len(train_loader)
        avg_train_loss = train_loss / num_train_batches

        # ------------------------------------------
        # 验证阶段
        # ------------------------------------------
        model.eval()
        val_loss = 0
        apa_true_total, apa_pred_total = [], []
        mdd_true, mdd_pred = [], []
        asr_true, asr_pred = [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]  ", leave=False):
                input_features = batch["input_features"].to(device)
                target_pinyin_ids = batch["target_pinyin_ids"].to(device)
                actual_pinyin_ids = batch["actual_pinyin_ids"].to(device)
                mdd_labels = batch["mdd_labels"].to(device)
                targets_scores = batch["scores"].to(device)

                apa_scores, mdd_logits, asr_logits = model(input_features, target_pinyin_ids)

                loss_apa = criterion_apa(apa_scores, targets_scores)
                loss_mdd = criterion_mdd(mdd_logits, mdd_labels)
                loss_asr = criterion_asr(asr_logits, actual_pinyin_ids)

                v_loss = (w_apa * loss_apa) + (w_mdd * loss_mdd) + (w_asr * loss_asr)
                val_loss += v_loss.item()

                apa_true_total.extend(targets_scores[:, 0].cpu().numpy())
                apa_pred_total.extend(apa_scores[:, 0].cpu().numpy())

                mdd_true.extend(mdd_labels.cpu().numpy())
                mdd_pred.extend(torch.argmax(mdd_logits, dim=1).cpu().numpy())

                asr_true.extend(actual_pinyin_ids.cpu().numpy())
                asr_pred.extend(torch.argmax(asr_logits, dim=1).cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)

        val_pcc_total = safe_pearsonr(pearsonr, apa_true_total, apa_pred_total)

        val_mdd_f1 = f1_score(mdd_true, mdd_pred, average='macro', zero_division=0)
        val_asr_acc = accuracy_score(asr_true, asr_pred)
        val_asr_per = pinyin_error_rate(asr_true, asr_pred, id2pinyin)

        print(f"\n✅ [Epoch {epoch+1} 总结]")
        print(f"   📉 Train Loss: {avg_train_loss:.4f} | APA: {train_apa_loss/num_train_batches:.2f} | MDD: {train_mdd_loss/num_train_batches:.2f} | ASR: {train_asr_loss/num_train_batches:.2f}")
        print(f"   📈 Val   Loss: {avg_val_loss:.4f}  | 🎯 APA PCC: {val_pcc_total:.4f} | 🎯 MDD F1: {val_mdd_f1:.4f} | 🎯 ASR Acc: {val_asr_acc:.4f} | 🎯 ASR PER: {val_asr_per:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = base_dir / "whisper_base_mtl_best.pth"
            torch.save(model.state_dict(), best_model_path)
            print(f"   🌟 发现最佳模型！权重已保存至: {best_model_path}\n")
        else:
            print("\n")

if __name__ == "__main__":
    train_and_validate_mtl()
