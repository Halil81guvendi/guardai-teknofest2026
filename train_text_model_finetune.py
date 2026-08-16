"""
GuardAI - train_text_model_finetune.py
=========================================
TEKNOFEST 2026 Sosyal İnovasyon Yarışması — FINE-TUNE TURU (2. Aşama)

BERTurk'ü BİRDEN FAZLA bağımsız Türkçe toksisite/nefret söylemi veri setini
birleştirerek yeniden eğitir. Amaç: tek bir veri setinin etiketleme
üslubuna/önyargısına aşırı uyum sağlamayı önlemek, genelleme kabiliyetini
artırmak.

Kullanılan veri setleri:
    1. Overfit-GM/turkish-toxic-language   (~77.8K)
    2. Toygar/turkish-offensive-language-detection  (~53K)
    3. coltekin/offenseval2020_tr           (~31.7K, OffensEval-TR resmi seti)

Her veri setinin kolon isimleri/etiket formatı farklı olabileceğinden,
otomatik kolon algılama ve esnek etiket eşleştirme kullanılır. Bir veri
seti yüklenemez/ayrıştırılamazsa, script DURMAZ — sadece o veri setini
atlar ve diğerleriyle devam eder.

İyileştirmeler (v1'e göre):
    - Çoklu veri seti birleşimi (genelleme kabiliyeti)
    - Artırılmış dropout (hidden_dropout_prob, attention_probs_dropout_prob)
    - Label smoothing
    - En iyi model seçim metriği: F1 (dengeli ikili sınıflandırma için uygun)

Çıktı: ./trained_models/guardai-text-turkish-toxic-finetune/
       (v1 modeli üzerine YAZILMAZ, ayrı klasörde tutulur — karşılaştırma
       için ikisi de saklanır)

Kullanım:
    python train_text_model_finetune.py
    python train_text_model_finetune.py --epochs 3 --dropout 0.2
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="[GuardAI-TextFinetune] %(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("GuardAI-TextFinetune")

MODEL_CHECKPOINT = "dbmdz/bert-base-turkish-cased"
OUTPUT_DIR = "./trained_models/guardai-text-turkish-toxic-finetune"
REPORTS_DIR = "./reports"
MAX_LENGTH = 128
SEED = 42

DATASET_SOURCES = [
    "Overfit-GM/turkish-toxic-language",
    "Toygar/turkish-offensive-language-detection",
    "coltekin/offenseval2020_tr",
]

TEXT_COL_CANDIDATES = ["text", "tweet", "Tweet", "comment", "content", "sentence"]
LABEL_COL_CANDIDATES = [
    "is_toxic", "label", "is_offensive", "offensive", "target",
    "class", "subtask_a", "Label",
]

# Etiket değerini ikili (0=toksik değil, 1=toksik/offensive) hale getirmek
# için kullanılan anahtar kelime eşleştirmesi (küçük harfe çevrilip aranır)
POSITIVE_MARKERS = ["off", "offensive", "toxic", "hate", "1", "true", "insult"]
NEGATIVE_MARKERS = ["not", "non", "none", "clean", "0", "false", "neutral"]


def parse_args():
    parser = argparse.ArgumentParser(description="GuardAI Metin Modeli Fine-Tune (Çoklu Veri Seti)")
    parser.add_argument("--epochs", type=int, default=4, help="Eğitim epoch sayısı")
    parser.add_argument("--batch-size", type=int, default=16, help="Eğitim batch boyutu")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Öğrenme oranı")
    parser.add_argument("--dropout", type=float, default=0.2, help="Hidden/attention dropout oranı")
    parser.add_argument("--label-smoothing", type=float, default=0.05, help="Label smoothing faktörü")
    parser.add_argument("--sample-size", type=int, default=None, help="Hızlı test için toplam örnek sınırı")
    return parser.parse_args()


def set_seed(seed: int = SEED):
    import random
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_label_value(value) -> int:
    """Farklı formatlardaki etiket değerlerini (0/1, 'OFF'/'NOT', True/False vb.)
    ikili (0/1) forma çevirir. Belirsiz durumda None döner (satır atlanır)."""
    if pd.isna(value):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        return int(value) if int(value) in (0, 1) else None
    if isinstance(value, bool):
        return int(value)

    text_val = str(value).strip().lower()
    for marker in POSITIVE_MARKERS:
        if text_val == marker or text_val.startswith(marker):
            return 1
    for marker in NEGATIVE_MARKERS:
        if text_val == marker or text_val.startswith(marker):
            return 0
    return None


def load_single_dataset(name: str) -> pd.DataFrame:
    """Tek bir veri setini indirir, kolonları otomatik algılar, (text, is_toxic)
    şemasına normalize eder. Başarısız olursa boş DataFrame döner (script durmaz)."""
    from datasets import load_dataset

    try:
        logger.info(f"Veri seti deneniyor: {name}")
        raw = load_dataset(name)

        # Tüm split'leri birleştir (kendi train/val/test bölmemizi kendimiz yapacağız)
        frames = []
        for split_name in raw.keys():
            frames.append(raw[split_name].to_pandas())
        df = pd.concat(frames, ignore_index=True)

        text_col = next((c for c in TEXT_COL_CANDIDATES if c in df.columns), None)
        label_col = next((c for c in LABEL_COL_CANDIDATES if c in df.columns), None)

        if text_col is None or label_col is None:
            logger.warning(
                f"'{name}' için text/label kolonu bulunamadı "
                f"(mevcut kolonlar: {list(df.columns)}). Bu veri seti atlanıyor."
            )
            return pd.DataFrame(columns=["text", "is_toxic"])

        out = pd.DataFrame()
        out["text"] = df[text_col].astype(str)
        out["is_toxic"] = df[label_col].apply(normalize_label_value)
        out = out.dropna(subset=["is_toxic"])
        out["is_toxic"] = out["is_toxic"].astype(int)
        out = out[out["text"].str.len() >= 2]

        logger.info(f"'{name}' başarıyla yüklendi ve normalize edildi: {len(out)} satır")
        return out
    except Exception as e:
        logger.warning(f"'{name}' yüklenemedi ({type(e).__name__}: {e}). Bu veri seti atlanıyor.")
        return pd.DataFrame(columns=["text", "is_toxic"])


def load_and_prepare_combined_dataset(sample_size: int = None):
    from sklearn.model_selection import train_test_split

    all_frames = [load_single_dataset(name) for name in DATASET_SOURCES]
    combined = pd.concat(all_frames, ignore_index=True)
    combined = combined.dropna(subset=["text", "is_toxic"])
    combined = combined.drop_duplicates(subset=["text"]).reset_index(drop=True)

    logger.info(f"BİRLEŞTİRİLMİŞ veri seti: {len(combined)} benzersiz satır "
                f"({len(DATASET_SOURCES)} kaynaktan)")
    logger.info(
        f"Sınıf dağılımı -> Toksik: {int(combined['is_toxic'].sum())} | "
        f"Toksik Değil: {int((combined['is_toxic'] == 0).sum())}"
    )

    if sample_size and sample_size < len(combined):
        combined, _ = train_test_split(
            combined, train_size=sample_size, stratify=combined["is_toxic"], random_state=SEED
        )
        logger.info(f"Hızlı-test modu: {sample_size} satıra örneklendi.")

    train_df, temp_df = train_test_split(
        combined, test_size=0.15, stratify=combined["is_toxic"], random_state=SEED
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df["is_toxic"], random_state=SEED
    )

    logger.info(f"Bölünme -> Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    return (
        train_df[["text", "is_toxic"]].reset_index(drop=True),
        val_df[["text", "is_toxic"]].reset_index(drop=True),
        test_df[["text", "is_toxic"]].reset_index(drop=True),
    )


def tokenize_datasets(train_df, val_df, test_df, tokenizer):
    from datasets import Dataset

    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH, padding="max_length")

    train_ds = Dataset.from_pandas(train_df.rename(columns={"is_toxic": "labels"}))
    val_ds = Dataset.from_pandas(val_df.rename(columns={"is_toxic": "labels"}))
    test_ds = Dataset.from_pandas(test_df.rename(columns={"is_toxic": "labels"}))

    train_ds = train_ds.map(tokenize_fn, batched=True)
    val_ds = val_ds.map(tokenize_fn, batched=True)
    test_ds = test_ds.map(tokenize_fn, batched=True)

    keep_cols = ["input_ids", "attention_mask", "token_type_ids", "labels"]
    train_ds.set_format(type="torch", columns=[c for c in keep_cols if c in train_ds.column_names])
    val_ds.set_format(type="torch", columns=[c for c in keep_cols if c in val_ds.column_names])
    test_ds.set_format(type="torch", columns=[c for c in keep_cols if c in test_ds.column_names])

    return train_ds, val_ds, test_ds


def compute_metrics_fn(eval_pred):
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    probs = np.exp(logits[:, 1]) / np.sum(np.exp(logits), axis=-1)
    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
    }
    try:
        metrics["roc_auc"] = roc_auc_score(labels, probs)
    except ValueError:
        metrics["roc_auc"] = 0.0
    return metrics


def plot_confusion_matrix(y_true, y_pred, save_path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Toksik Değil", "Toksik"])
    ax.set_yticklabels(["Toksik Değil", "Toksik"])
    ax.set_xlabel("Tahmin Edilen"); ax.set_ylabel("Gerçek Etiket")
    ax.set_title("GuardAI Metin Modeli (Fine-tune) - Karışıklık Matrisi")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"Karışıklık matrisi kaydedildi: {save_path}")


def main():
    args = parse_args()
    set_seed()

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(REPORTS_DIR).mkdir(parents=True, exist_ok=True)

    import torch
    from transformers import (
        AutoConfig, AutoModelForSequenceClassification, AutoTokenizer,
        Trainer, TrainingArguments, EarlyStoppingCallback,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Kullanılan cihaz: {device.upper()}"
                + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    train_df, val_df, test_df = load_and_prepare_combined_dataset(sample_size=args.sample_size)

    logger.info(f"Tokenizer ve model yükleniyor: {MODEL_CHECKPOINT} (dropout={args.dropout})")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

    config = AutoConfig.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=2,
        id2label={0: "NOT_TOXIC", 1: "TOXIC"},
        label2id={"NOT_TOXIC": 0, "TOXIC": 1},
        hidden_dropout_prob=args.dropout,
        attention_probs_dropout_prob=args.dropout,
    )
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_CHECKPOINT, config=config)

    train_ds, val_ds, test_ds = tokenize_datasets(train_df, val_df, test_df, tokenizer)

    training_args = TrainingArguments(
        output_dir="./trained_models/_checkpoints_text_finetune",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        warmup_ratio=0.1,
        label_smoothing_factor=args.label_smoothing,
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="f1",   # Metin/ikili sınıflandırma -> F1 seçim metriği
        greater_is_better=True,
        fp16=(device == "cuda"),
        dataloader_num_workers=0,     # Windows uyumluluğu
        report_to=[],
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    logger.info("=" * 60)
    logger.info(f"FINE-TUNE BAŞLIYOR — {args.epochs} epoch, dropout={args.dropout}, "
                f"label_smoothing={args.label_smoothing}")
    logger.info(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    logger.info("=" * 60)

    start_time = time.time()
    trainer.train()
    elapsed = time.time() - start_time
    logger.info(f"Eğitim tamamlandı. Süre: {elapsed / 60:.1f} dakika")

    log_history = pd.DataFrame(trainer.state.log_history)
    log_history.to_csv(f"{REPORTS_DIR}/text_finetune_training_log.csv", index=False)

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    logger.info(f"Fine-tune edilmiş model kaydedildi: {OUTPUT_DIR}")

    # Nihai test değerlendirmesi
    predictions = trainer.predict(test_ds)
    preds = np.argmax(predictions.predictions, axis=-1)
    labels = predictions.label_ids

    from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score

    probs = np.exp(predictions.predictions[:, 1]) / np.sum(np.exp(predictions.predictions), axis=-1)
    report_dict = classification_report(
        labels, preds, target_names=["NOT_TOXIC", "TOXIC"], output_dict=True, zero_division=0
    )
    report_text = classification_report(labels, preds, target_names=["NOT_TOXIC", "TOXIC"], zero_division=0)
    test_auc = roc_auc_score(labels, probs)

    logger.info("\n" + report_text)
    logger.info(f"Test Accuracy: {accuracy_score(labels, preds):.4f} | "
                f"Test F1: {f1_score(labels, preds):.4f} | Test ROC-AUC: {test_auc:.4f}")

    plot_confusion_matrix(labels, preds, f"{REPORTS_DIR}/text_finetune_confusion_matrix.png")

    final_report = {
        "model_checkpoint": MODEL_CHECKPOINT,
        "datasets_used": DATASET_SOURCES,
        "dropout": args.dropout,
        "label_smoothing": args.label_smoothing,
        "train_size": len(train_df), "val_size": len(val_df), "test_size": len(test_df),
        "selection_metric": "f1",
        "test_metrics": report_dict,
        "test_accuracy": accuracy_score(labels, preds),
        "test_f1": f1_score(labels, preds),
        "test_roc_auc": test_auc,
        "output_dir": OUTPUT_DIR,
    }
    with open(f"{REPORTS_DIR}/text_finetune_evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(final_report, f, ensure_ascii=False, indent=2)
    logger.info(f"Değerlendirme raporu kaydedildi: {REPORTS_DIR}/text_finetune_evaluation_report.json")

    logger.info("=" * 60)
    logger.info(f"TAMAMLANDI. Fine-tune edilmiş model: {OUTPUT_DIR}")
    logger.info("models.py otomatik olarak önce bu 'finetune' klasörünü arayacak.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()