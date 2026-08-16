"""
GuardAI - train_text_model.py
================================
TEKNOFEST 2026 Sosyal İnovasyon Yarışması

BERTurk (dbmdz/bert-base-turkish-cased) modelini, Türkçe toksik dil /
nefret söylemi / siber zorbalık tespiti için gerçek, açık kaynaklı bir
veri seti üzerinde FINE-TUNE eder.

Veri Seti: Overfit-GM/turkish-toxic-language (HuggingFace Datasets)
    - 77.800 satır, Türkçe
    - Jigsaw Multilingual Toxic Comments + Turkish Offensive Language
      Detection Dataset + Turkish Cyberbullying Dataset birleşimi
    - Kolonlar: text, target (OTHER/PROFANITY/INSULT/RACIST/SEXIST), is_toxic (0/1)

Kullanım:
    python train_text_model.py
    python train_text_model.py --epochs 4 --batch-size 16 --sample-size 20000

Çıktılar:
    ./trained_models/guardai-text-turkish-toxic/   -> Eğitilmiş model + tokenizer
    ./reports/text_model_evaluation_report.json    -> Detaylı metrikler
    ./reports/text_model_confusion_matrix.png      -> Karışıklık matrisi görseli
    ./reports/text_model_training_log.csv          -> Epoch bazlı loss/metric geçmişi
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="[GuardAI-TextTrain] %(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("GuardAI-TextTrain")

MODEL_CHECKPOINT = "dbmdz/bert-base-turkish-cased"
DATASET_NAME = "Overfit-GM/turkish-toxic-language"
OUTPUT_DIR = "./trained_models/guardai-text-turkish-toxic"
REPORTS_DIR = "./reports"
MAX_LENGTH = 128
SEED = 42


def parse_args():
    parser = argparse.ArgumentParser(description="GuardAI Metin Toksisite Modeli Eğitimi")
    parser.add_argument("--epochs", type=int, default=3, help="Eğitim epoch sayısı")
    parser.add_argument("--batch-size", type=int, default=16, help="Eğitim batch boyutu")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Öğrenme oranı")
    parser.add_argument(
        "--sample-size", type=int, default=None,
        help="Hızlı test için veri setinden alınacak örnek sayısı (varsayılan: tümü, 77.8k)",
    )
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Sadece mevcut eğitilmiş modeli değerlendir, yeniden eğitme",
    )
    return parser.parse_args()


def set_seed(seed: int = SEED):
    import random
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_and_prepare_dataset(sample_size: int = None):
    """HuggingFace Hub'dan veri setini indirir, temizler ve train/val/test olarak böler."""
    from datasets import load_dataset
    from sklearn.model_selection import train_test_split

    logger.info(f"Veri seti indiriliyor: {DATASET_NAME} (ilk çalıştırmada ~18MB indirilecek)")
    raw = load_dataset(DATASET_NAME, split="train")
    df = raw.to_pandas()
    logger.info(f"Ham veri seti yüklendi: {len(df)} satır")

    # Temizlik: boş/çok kısa metinleri at
    df = df.dropna(subset=["text", "is_toxic"])
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() >= 2].reset_index(drop=True)
    df["is_toxic"] = df["is_toxic"].astype(int)

    if sample_size and sample_size < len(df):
        logger.info(f"Hızlı-test modu: veri seti {sample_size} satıra örneklendi.")
        df, _ = train_test_split(
            df, train_size=sample_size, stratify=df["is_toxic"], random_state=SEED
        )

    logger.info(f"Sınıf dağılımı -> Toksik: {int(df['is_toxic'].sum())} | "
                f"Toksik Değil: {int((df['is_toxic'] == 0).sum())}")

    # %80 train, %10 val, %10 test (stratified)
    train_df, temp_df = train_test_split(
        df, test_size=0.2, stratify=df["is_toxic"], random_state=SEED
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
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
    }


def plot_confusion_matrix(y_true, y_pred, save_path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Toksik Değil", "Toksik"])
    ax.set_yticklabels(["Toksik Değil", "Toksik"])
    ax.set_xlabel("Tahmin Edilen")
    ax.set_ylabel("Gerçek Etiket")
    ax.set_title("GuardAI Metin Modeli - Karışıklık Matrisi")
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
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        EarlyStoppingCallback,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Kullanılan cihaz: {device.upper()}"
                + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    # -----------------------------------------------------------
    # 1) VERİ SETİ
    # -----------------------------------------------------------
    train_df, val_df, test_df = load_and_prepare_dataset(sample_size=args.sample_size)

    # -----------------------------------------------------------
    # 2) TOKENIZER & MODEL
    # -----------------------------------------------------------
    logger.info(f"Tokenizer ve model yükleniyor: {MODEL_CHECKPOINT}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=2,
        id2label={0: "NOT_TOXIC", 1: "TOXIC"},
        label2id={"NOT_TOXIC": 0, "TOXIC": 1},
    )

    train_ds, val_ds, test_ds = tokenize_datasets(train_df, val_df, test_df, tokenizer)

    # -----------------------------------------------------------
    # 3) EĞİTİM
    # -----------------------------------------------------------
    training_args = TrainingArguments(
        output_dir="./trained_models/_checkpoints_text",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_dir="./reports/text_model_tb_logs",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=(device == "cuda"),
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

    if not args.eval_only:
        logger.info("=" * 60)
        logger.info(f"EĞİTİM BAŞLIYOR — {args.epochs} epoch, batch={args.batch_size}, lr={args.learning_rate}")
        logger.info(f"Train örnek sayısı: {len(train_ds)} | Val örnek sayısı: {len(val_ds)}")
        logger.info("=" * 60)

        start_time = time.time()
        train_result = trainer.train()
        elapsed = time.time() - start_time
        logger.info(f"Eğitim tamamlandı. Süre: {elapsed / 60:.1f} dakika")

        # Epoch bazlı log geçmişini kaydet
        log_history = pd.DataFrame(trainer.state.log_history)
        log_history.to_csv(f"{REPORTS_DIR}/text_model_training_log.csv", index=False)

        # Modeli ve tokenizer'ı nihai konuma kaydet
        trainer.save_model(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        logger.info(f"Model kaydedildi: {OUTPUT_DIR}")

    # -----------------------------------------------------------
    # 4) TEST SETİ ÜZERİNDE NİHAİ DEĞERLENDİRME
    # -----------------------------------------------------------
    logger.info("Test seti üzerinde nihai değerlendirme yapılıyor...")
    predictions = trainer.predict(test_ds)
    preds = np.argmax(predictions.predictions, axis=-1)
    labels = predictions.label_ids

    from sklearn.metrics import classification_report, accuracy_score, f1_score

    report_dict = classification_report(
        labels, preds, target_names=["NOT_TOXIC", "TOXIC"], output_dict=True, zero_division=0
    )
    report_text = classification_report(
        labels, preds, target_names=["NOT_TOXIC", "TOXIC"], zero_division=0
    )

    logger.info("\n" + report_text)
    logger.info(f"Test Accuracy: {accuracy_score(labels, preds):.4f} | "
                f"Test F1: {f1_score(labels, preds):.4f}")

    plot_confusion_matrix(labels, preds, f"{REPORTS_DIR}/text_model_confusion_matrix.png")

    final_report = {
        "model_checkpoint": MODEL_CHECKPOINT,
        "dataset": DATASET_NAME,
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "test_metrics": report_dict,
        "test_accuracy": accuracy_score(labels, preds),
        "test_f1": f1_score(labels, preds),
        "output_dir": OUTPUT_DIR,
    }
    with open(f"{REPORTS_DIR}/text_model_evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(final_report, f, ensure_ascii=False, indent=2)
    logger.info(f"Değerlendirme raporu kaydedildi: {REPORTS_DIR}/text_model_evaluation_report.json")

    # -----------------------------------------------------------
    # 5) HIZLI ÖRNEK ÇIKARIM TESTİ (jüriye gösterilebilir sanity-check)
    # -----------------------------------------------------------
    logger.info("=" * 60)
    logger.info("ÖRNEK ÇIKARIM TESTİ")
    logger.info("=" * 60)
    model.eval()
    model.to(device)
    test_sentences = [
        "Bugün hava çok güzel, dışarı çıkacağım.",
        "Sen tam bir aptalsın, kimse seni sevmiyor.",
        "Bu maçta hakem kararları çok tartışmalıydı.",
        "Siktir git buradan seni pislik.",
    ]
    for sentence in test_sentences:
        inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
        pred_label = "TOXIC" if probs[1] > probs[0] else "NOT_TOXIC"
        logger.info(f"  '{sentence}' -> {pred_label} (skor: {probs[1].item():.4f})")

    logger.info("=" * 60)
    logger.info(f"TAMAMLANDI. Eğitilmiş model burada: {OUTPUT_DIR}")
    logger.info("Bu modeli models.py içindeki TextModerationEngine'e bağlamak için:")
    logger.info(f'    TextModerationEngine(model_name="{OUTPUT_DIR}")')
    logger.info("=" * 60)


if __name__ == "__main__":
    main()