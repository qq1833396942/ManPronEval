import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, f1_score
from jiwer import cer

from dataset_mtl_wavlm import MTLDataset, mtl_collate_fn
from model_mtl_wavlm_large import MultiTaskWavLMLarge


def train_and_validate_mtl_large():
    CODE_DIR = os.path.dirname(os.path.abspath(__file__))
    SAVE_DIR = CODE_DIR
    VOCAB_PATH = os.path.join(CODE_DIR, "syllable_vocab.json")
    TRAIN_JSON_PATH = os.path.join(CODE_DIR, "metadata_train_mtl.json")
    VAL_JSON_PATH = os.path.join(CODE_DIR, "metadata_val_mtl.json")
    LOCAL_MODEL_PATH = r""

    # 与 HuBERT-Large 训练脚本严格对齐
    epochs = 30
    batch_size = 4
    accum_steps = 4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 计算设备: {device} | 模式: 2/3 冻结 + 后 1/3 LoRA")

    with open(VOCAB_PATH, 'r', encoding='utf-8') as f:
        pinyin2id = json.load(f)
    vocab_size = len(pinyin2id)

    train_dataset = MTLDataset(TRAIN_JSON_PATH, pinyin2id)
    val_dataset = MTLDataset(VAL_JSON_PATH, pinyin2id)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=mtl_collate_fn, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=mtl_collate_fn, num_workers=4)

    model = MultiTaskWavLMLarge(model_path=LOCAL_MODEL_PATH, vocab_size=vocab_size).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler()

    criterion_apa = nn.MSELoss()
    criterion_mdd = nn.CrossEntropyLoss()
    criterion_asr = nn.CrossEntropyLoss()

    w_apa, w_mdd, w_asr = 1.0, 1.0, 1.0
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        pbar_train = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")

        for i, batch in enumerate(pbar_train):
            input_values = batch["input_values"].to(device)
            target_pinyin_ids = batch["target_pinyin_ids"].to(device)
            actual_pinyin_ids = batch["actual_pinyin_ids"].to(device)
            mdd_labels = batch["mdd_labels"].to(device)
            targets_scores = batch["scores"].to(device)

            with torch.autocast(device_type='cuda'):
                apa_scores, mdd_logits, asr_logits = model(input_values, target_pinyin_ids)

                loss_apa = criterion_apa(apa_scores, targets_scores)
                loss_mdd = criterion_mdd(mdd_logits, mdd_labels)
                loss_asr = criterion_asr(asr_logits, actual_pinyin_ids)

                loss = ((w_apa * loss_apa) + (w_mdd * loss_mdd) + (w_asr * loss_asr)) / accum_steps

            scaler.scale(loss).backward()

            if (i + 1) % accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss += loss.item() * accum_steps
            pbar_train.set_postfix({'L': f"{loss.item() * accum_steps:.2f}"})

        model.eval()
        val_loss = 0
        apa_true, apa_pred = [], []
        mdd_true, mdd_pred = [], []
        asr_true_ids, asr_pred_ids = [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]  ", leave=False):
                input_values = batch["input_values"].to(device)
                target_pinyin_ids = batch["target_pinyin_ids"].to(device)
                actual_pinyin_ids = batch["actual_pinyin_ids"].to(device)
                mdd_labels = batch["mdd_labels"].to(device)
                targets_scores = batch["scores"].to(device)

                with torch.autocast(device_type='cuda'):
                    apa_scores, mdd_logits, asr_logits = model(input_values, target_pinyin_ids)
                    v_loss = (w_apa * criterion_apa(apa_scores, targets_scores)) +                              (w_mdd * criterion_mdd(mdd_logits, mdd_labels)) +                              (w_asr * criterion_asr(asr_logits, actual_pinyin_ids))

                val_loss += v_loss.item()

                apa_true.extend(targets_scores[:, 0].cpu().numpy())
                apa_pred.extend(apa_scores[:, 0].cpu().numpy())
                mdd_true.extend(mdd_labels.cpu().numpy())
                mdd_pred.extend(torch.argmax(mdd_logits, dim=1).cpu().numpy())

                a_true = actual_pinyin_ids.cpu().numpy()
                a_pred = torch.argmax(asr_logits, dim=1).cpu().numpy()
                asr_true_ids.extend(a_true)
                asr_pred_ids.extend(a_pred)

        avg_val_loss = val_loss / len(val_loader)

        try:
            val_pcc, _ = pearsonr(apa_true, apa_pred)
        except Exception:
            val_pcc = 0.0

        val_mdd_f1 = f1_score(mdd_true, mdd_pred, average='macro', zero_division=0)
        val_asr_acc = accuracy_score(asr_true_ids, asr_pred_ids)

        str_true = [str(i) for i in asr_true_ids]
        str_pred = [str(i) for i in asr_pred_ids]
        val_asr_cer = cer(str_true, str_pred)

        print(f"\n✅ [Epoch {epoch+1} 总结]")
        print(f"   📉 Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"   🎯 APA PCC: {val_pcc:.4f} | MDD F1: {val_mdd_f1:.4f}")
        print(f"   🎯 ASR Acc: {val_asr_acc:.4f} | ASR CER: {val_asr_cer:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "mtl_wavlm_large_best.pth"))
            print("   🌟 发现最佳模型！权重已保存至: mtl_wavlm_large_best.pth\n")


if __name__ == "__main__":
    train_and_validate_mtl_large()
