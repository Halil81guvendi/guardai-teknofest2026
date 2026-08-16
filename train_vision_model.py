"""
GuardAI - train_vision_model.py
==================================
TEKNOFEST 2026 Sosyal İnovasyon Yarışması

Vision Transformer (google/vit-base-patch16-224-in21k) modelini gerçek/
deepfake yüz görseli veri seti üzerinde FINE-TUNE eder.

Veri Seti: pujanpaudel/deepfake_face_classification (HuggingFace Datasets)
    - 32.134 görsel (16.060 gerçek + 16.060 sahte, dengeli)
    - DF40 veri setinden türetilmiş (40 farklı state-of-the-art deepfake
      tekniğini kapsayan akademik referans veri seti)

Kullanım:
    python train_vision_model.py
    python train_vision_model.py --epochs 3 --batch-size 8 --sample-size 4000

Çıktılar:
    ./trained_models/guardai-vision-deepfake/       -> Eğitilmiş model + processor
    ./reports/vision_model_evaluation_report.json   -> Detaylı metrikler
    ./reports/vision_model_confusion_matrix.png      -> Karışıklık matrisi görseli
    ./reports/vision_model_training_log.csv          -> Epoch bazlı loss/metric geçmişi
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
    format="[GuardAI-VisionTrain] %(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("GuardAI-VisionTrain")

MODEL_CHECKPOINT = "google/vit-base-patch16-224-in21k"

# Birincil veri seti bozuk çıktı (bkz. HF sayfasındaki resmi format hatası).
# Bu yüzden otomatik olarak sağlam alternatiflere geçen bir zincir kullanıyoruz.
DATASET_CANDIDATES = [
    "prithivMLmods/Deepfake-vs-Real-v2",      # Temiz binary (Deepfake/Real), standart parquet
    "Hemg/deepfake-and-real-images",           # Yedek 1
    "pujanpaudel/deepfake_face_classification",  # Orijinal tercih (bozuk olabilir, en son denenir)
]
OUTPUT_DIR = "./trained_models/guardai-vision-deepfake"
REPORTS_DIR = "./reports"
SEED = 42


def parse_args():
    parser = argparse.ArgumentParser(description="GuardAI Görsel Deepfake Tespit Modeli Eğitimi")
    parser.add_argument("--epochs", type=int, default=3, help="Eğitim epoch sayısı")
    parser.add_argument("--batch-size", type=int, default=8,
                         help="Eğitim batch boyutu (düşük VRAM için 8 önerilir)")
    parser.add_argument("--grad-accum", type=int, default=2,
                         help="Gradient accumulation adımı (etkin batch = batch_size * grad_accum)")
    parser.add_argument("--learning-rate", type=float, default=3e-5, help="Öğrenme oranı")
    parser.add_argument(
        "--sample-size", type=int, default=None,
        help="Hızlı test için veri setinden alınacak toplam örnek sayısı (varsayılan: tümü, 32.1k)",
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


def resolve_label_names(dataset):
    """Veri setindeki ClassLabel isimlerini (real/fake) otomatik tespit eder."""
    features = dataset.features
    label_col = "label" if "label" in features else list(features.keys())[-1]
    class_names = getattr(features[label_col], "names", None)
    if class_names is None:
        class_names = ["real", "fake"]
    logger.info(f"Etiket kolonu: '{label_col}' | Sınıflar: {class_names}")
    return label_col, class_names


def load_and_prepare_dataset(sample_size: int = None):
    """
    HuggingFace Hub'dan DATASET_CANDIDATES listesindeki veri setlerini sırayla
    dener, ilk başarılı olanı kullanır. Bu, tekil bir deponun bozuk/eksik
    yapılandırılmış olması durumunda (örn. format algılama hatası) sürecin
    durmasını engeller.
    """
    from datasets import load_dataset

    raw = None
    used_dataset_name = None
    last_error = None

    for candidate in DATASET_CANDIDATES:
        try:
            logger.info(f"Veri seti deneniyor: {candidate}")
            raw = load_dataset(candidate)
            used_dataset_name = candidate
            logger.info(f"Başarılı! Kullanılacak veri seti: {candidate}")
            break
        except Exception as e:
            logger.warning(f"'{candidate}' yüklenemedi ({type(e).__name__}), bir sonraki aday deneniyor...")
            last_error = e
            continue

    if raw is None:
        raise RuntimeError(
            f"DATASET_CANDIDATES listesindeki hiçbir veri seti yüklenemedi. Son hata: {last_error}"
        )

    logger.info(f"Yüklenen split'ler: {list(raw.keys())}")

    has_ready_splits = "train" in raw and any(k in raw for k in ("validation", "val", "test"))

    if has_ready_splits:
        train_ds = raw["train"]
        val_ds = raw.get("validation", raw.get("val"))
        test_ds = raw.get("test")
        if test_ds is None:
            split = val_ds.train_test_split(test_size=0.5, seed=SEED)
            val_ds, test_ds = split["train"], split["test"]
        elif val_ds is None:
            split = train_ds.train_test_split(test_size=0.1, seed=SEED)
            train_ds, val_ds = split["train"], split["test"]

        label_col, class_names = resolve_label_names(train_ds)

        if sample_size:
            total = len(train_ds) + len(val_ds) + len(test_ds)
            ratio = min(1.0, sample_size / total)
            train_ds = train_ds.shuffle(seed=SEED).select(range(max(1, int(len(train_ds) * ratio))))
            val_ds = val_ds.shuffle(seed=SEED).select(range(max(1, int(len(val_ds) * ratio))))
            test_ds = test_ds.shuffle(seed=SEED).select(range(max(1, int(len(test_ds) * ratio))))
            logger.info("Hızlı-test modu: veri seti oranlanarak küçültüldü.")
    else:
        split_name = list(raw.keys())[0]
        ds = raw[split_name]
        label_col, class_names = resolve_label_names(ds)

        if sample_size and sample_size < len(ds):
            ds = ds.shuffle(seed=SEED).select(range(sample_size))

        split1 = ds.train_test_split(test_size=0.2, seed=SEED)
        split2 = split1["test"].train_test_split(test_size=0.5, seed=SEED)
        train_ds, val_ds, test_ds = split1["train"], split2["train"], split2["test"]

    logger.info(f"Bölünme -> Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    return train_ds, val_ds, test_ds, label_col, class_names, used_dataset_name


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


def plot_confusion_matrix(y_true, y_pred, class_names, save_path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Purples")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Tahmin Edilen")
    ax.set_ylabel("Gerçek Etiket")
    ax.set_title("GuardAI Görsel Modeli - Karışıklık Matrisi")
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
        AutoImageProcessor,
        ViTForImageClassification,
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
    train_ds, val_ds, test_ds, label_col, class_names, used_dataset_name = load_and_prepare_dataset(
        sample_size=args.sample_size
    )

    # id2label / label2id — "fake" veya "1" gibi farklı formatları normalize et
    id2label = {i: name for i, name in enumerate(class_names)}
    label2id = {name: i for i, name in enumerate(class_names)}

    # -----------------------------------------------------------
    # 2) IMAGE PROCESSOR & MODEL
    # -----------------------------------------------------------
    logger.info(f"Image processor ve model yükleniyor: {MODEL_CHECKPOINT}")
    processor = AutoImageProcessor.from_pretrained(MODEL_CHECKPOINT, use_fast=True)
    model = ViTForImageClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=len(class_names),
        id2label=id2label,
        label2id=label2id,
    )

    def transform_batch(examples):
        images = [img.convert("RGB") for img in examples["image"]]
        processed = processor(images=images, return_tensors="pt")
        examples["pixel_values"] = list(processed["pixel_values"])
        examples["labels"] = examples[label_col]
        return examples

    train_ds.set_transform(transform_batch)
    val_ds.set_transform(transform_batch)
    test_ds.set_transform(transform_batch)

    def collate_fn(batch):
        pixel_values = torch.stack([torch.as_tensor(item["pixel_values"]) for item in batch])
        labels = torch.tensor([int(item["labels"]) for item in batch])
        return {"pixel_values": pixel_values, "labels": labels}

    # -----------------------------------------------------------
    # 3) EĞİTİM
    # -----------------------------------------------------------
    training_args = TrainingArguments(
        output_dir="./trained_models/_checkpoints_vision",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=(device == "cuda"),
        dataloader_num_workers=0,  # Windows'ta local fonksiyonların pickle sorunu nedeniyle 0
        report_to=[],
        seed=SEED,
        remove_unused_columns=False,  # set_transform ile özel kolonları korumak için şart
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics_fn,
        data_collator=collate_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    if not args.eval_only:
        logger.info("=" * 60)
        logger.info(f"EĞİTİM BAŞLIYOR — {args.epochs} epoch, batch={args.batch_size} "
                    f"(x{args.grad_accum} accum = etkin {args.batch_size * args.grad_accum}), "
                    f"lr={args.learning_rate}")
        logger.info(f"Train örnek sayısı: {len(train_ds)} | Val örnek sayısı: {len(val_ds)}")
        logger.info("VRAM sınırlıysa ve 'CUDA out of memory' hatası alırsan: "
                    "--batch-size 4 --grad-accum 4 ile tekrar dene.")
        logger.info("=" * 60)

        start_time = time.time()
        trainer.train()
        elapsed = time.time() - start_time
        logger.info(f"Eğitim tamamlandı. Süre: {elapsed / 60:.1f} dakika")

        log_history = pd.DataFrame(trainer.state.log_history)
        log_history.to_csv(f"{REPORTS_DIR}/vision_model_training_log.csv", index=False)

        trainer.save_model(OUTPUT_DIR)
        processor.save_pretrained(OUTPUT_DIR)
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
        labels, preds, target_names=class_names, output_dict=True, zero_division=0
    )
    report_text = classification_report(labels, preds, target_names=class_names, zero_division=0)

    logger.info("\n" + report_text)
    logger.info(f"Test Accuracy: {accuracy_score(labels, preds):.4f} | "
                f"Test F1: {f1_score(labels, preds):.4f}")

    plot_confusion_matrix(labels, preds, class_names, f"{REPORTS_DIR}/vision_model_confusion_matrix.png")

    final_report = {
        "model_checkpoint": MODEL_CHECKPOINT,
        "dataset": used_dataset_name,
        "class_names": class_names,
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "test_size": len(test_ds),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "learning_rate": args.learning_rate,
        "test_metrics": report_dict,
        "test_accuracy": accuracy_score(labels, preds),
        "test_f1": f1_score(labels, preds),
        "output_dir": OUTPUT_DIR,
    }
    with open(f"{REPORTS_DIR}/vision_model_evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(final_report, f, ensure_ascii=False, indent=2)
    logger.info(f"Değerlendirme raporu kaydedildi: {REPORTS_DIR}/vision_model_evaluation_report.json")

    logger.info("=" * 60)
    logger.info(f"TAMAMLANDI. Eğitilmiş model burada: {OUTPUT_DIR}")
    logger.info("Bu modeli models.py içindeki VisualForensicsEngine'e bağlamak için:")
    logger.info(f'    VisualForensicsEngine(model_name="{OUTPUT_DIR}")')
    logger.info("(models.py zaten bu klasörü otomatik arayacak şekilde güncellenecek.)")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()