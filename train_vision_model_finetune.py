"""
GuardAI - train_vision_model_finetune.py
===========================================
TEKNOFEST 2026 Sosyal İnovasyon Yarışması — FINE-TUNE TURU (2. Aşama)

ViT'yi İKİ FARKLI sahtecilik "ailesinden" gelen veri setlerini birleştirerek
yeniden eğitir:
    1. Hemg/deepfake-and-real-images       (~190K, GAN ile TAMAMEN SENTEZLENMİŞ yüzler)
    2. RohanRamesh/ff-images-dataset        (~224K, FaceForensics++ — YÜZ DEĞİŞTİRME/
                                              YENİDEN CANLANDIRMA teknikleri: Deepfakes,
                                              Face2Face, FaceSwap, NeuralTextures)

Bu iki veri seti kökten farklı sahtecilik yöntemlerini temsil eder — biri
sıfırdan yüz üretir (StyleGAN benzeri), diğeri var olan bir yüzü değiştirir/
manipüle eder. İkisini birleştirmek, modelin tek bir sahtecilik türüne
"ezberlenmesini" önler ve gerçek dünyada karşılaşabileceği çeşitliliğe karşı
genelleme kabiliyetini ölçülebilir şekilde artırır (literatürde bu
"cross-dataset generalization" olarak adlandırılır).

İyileştirmeler (v1'e göre):
    - Çoklu veri seti birleşimi (2 farklı manipülasyon ailesi)
    - Artırılmış dropout (ViT config: hidden_dropout_prob)
    - Label smoothing
    - En iyi model seçim metriği: F1

Çıktı: ./trained_models/guardai-vision-deepfake-finetune/
       (v1 modeli korunur, üzerine yazılmaz)

Kullanım:
    python train_vision_model_finetune.py
    python train_vision_model_finetune.py --epochs 2 --sample-size 150000
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
    format="[GuardAI-VisionFinetune] %(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("GuardAI-VisionFinetune")

MODEL_CHECKPOINT = "google/vit-base-patch16-224-in21k"
OUTPUT_DIR = "./trained_models/guardai-vision-deepfake-finetune"
REPORTS_DIR = "./reports"
SEED = 42

# Her iki veri seti de AYNI kodlamayı kullanıyor: 0=Fake/FAKE, 1=Real/REAL
# (Hemg: ClassLabel(['Fake','Real']) — RohanRamesh: 0=FAKE, 1=REAL) — bu yüzden
# ekstra bir etiket eşleştirmesi gerekmez, doğrudan birleştirilebilir.
DATASET_SOURCES = [
    "Hemg/deepfake-and-real-images",
    "RohanRamesh/ff-images-dataset",
]
CLASS_NAMES = ["Fake", "Real"]


def parse_args():
    parser = argparse.ArgumentParser(description="GuardAI Görsel Modeli Fine-Tune (Çoklu Veri Seti)")
    parser.add_argument("--epochs", type=int, default=2, help="Eğitim epoch sayısı")
    parser.add_argument("--batch-size", type=int, default=16, help="Eğitim batch boyutu")
    parser.add_argument("--grad-accum", type=int, default=2, help="Gradient accumulation adımı")
    parser.add_argument("--learning-rate", type=float, default=3e-5, help="Öğrenme oranı")
    parser.add_argument("--dropout", type=float, default=0.15, help="Hidden dropout oranı")
    parser.add_argument("--label-smoothing", type=float, default=0.05, help="Label smoothing faktörü")
    parser.add_argument(
        "--sample-size", type=int, default=None,
        help="Toplam görsel sayısını sınırlar (varsayılan: tümü, ~414K — GPU'da uzun sürer, "
             "hızlı test için örn. 100000 önerilir)",
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


def load_single_dataset(name: str):
    """Tek bir görsel veri setini yükler, (image, label) şemasına normalize eder.
    Başarısız olursa None döner (script durmaz, o kaynak atlanır)."""
    from datasets import load_dataset, concatenate_datasets

    try:
        logger.info(f"Veri seti deneniyor: {name}")
        raw = load_dataset(name)

        # Tüm split'leri tek havuzda birleştir (kendi bölmemizi kendimiz yapacağız)
        parts = [raw[split] for split in raw.keys()]
        ds = concatenate_datasets(parts) if len(parts) > 1 else parts[0]

        if "image" not in ds.column_names or "label" not in ds.column_names:
            logger.warning(
                f"'{name}' için 'image'/'label' kolonu bulunamadı "
                f"(mevcut kolonlar: {ds.column_names}). Bu veri seti atlanıyor."
            )
            return None

        # Fazladan kolonları at (kaynaklar arası şema tutarlılığı için)
        extra_cols = [c for c in ds.column_names if c not in ("image", "label")]
        if extra_cols:
            ds = ds.remove_columns(extra_cols)

        logger.info(f"'{name}' başarıyla yüklendi: {len(ds)} görsel")
        return ds
    except Exception as e:
        logger.warning(f"'{name}' yüklenemedi ({type(e).__name__}: {e}). Bu veri seti atlanıyor.")
        return None


def load_and_prepare_combined_dataset(sample_size: int = None):
    from datasets import concatenate_datasets

    loaded = [load_single_dataset(name) for name in DATASET_SOURCES]
    loaded = [d for d in loaded if d is not None]

    if not loaded:
        raise RuntimeError("DATASET_SOURCES listesindeki hiçbir veri seti yüklenemedi.")

    combined = concatenate_datasets(loaded) if len(loaded) > 1 else loaded[0]
    logger.info(f"BİRLEŞTİRİLMİŞ görsel veri seti: {len(combined)} görsel ({len(loaded)} kaynaktan)")

    combined = combined.shuffle(seed=SEED)
    if sample_size and sample_size < len(combined):
        combined = combined.select(range(sample_size))
        logger.info(f"Hızlı-mod: veri seti {sample_size} görsele sınırlandı.")

    split1 = combined.train_test_split(test_size=0.15, seed=SEED)
    split2 = split1["test"].train_test_split(test_size=0.5, seed=SEED)
    train_ds, val_ds, test_ds = split1["train"], split2["train"], split2["test"]

    logger.info(f"Bölünme -> Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
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
    im = ax.imshow(cm, cmap="Purples")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(CLASS_NAMES); ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Tahmin Edilen"); ax.set_ylabel("Gerçek Etiket")
    ax.set_title("GuardAI Görsel Modeli (Fine-tune) - Karışıklık Matrisi")
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
        AutoImageProcessor, ViTConfig, ViTForImageClassification,
        Trainer, TrainingArguments, EarlyStoppingCallback,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Kullanılan cihaz: {device.upper()}"
                + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    train_ds, val_ds, test_ds = load_and_prepare_combined_dataset(sample_size=args.sample_size)

    id2label = {0: CLASS_NAMES[0], 1: CLASS_NAMES[1]}
    label2id = {name: i for i, name in enumerate(CLASS_NAMES)}

    logger.info(f"Image processor ve model yükleniyor: {MODEL_CHECKPOINT} (dropout={args.dropout})")
    processor = AutoImageProcessor.from_pretrained(MODEL_CHECKPOINT, use_fast=True)

    config = ViTConfig.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=len(CLASS_NAMES),
        id2label=id2label,
        label2id=label2id,
        hidden_dropout_prob=args.dropout,
    )
    model = ViTForImageClassification.from_pretrained(MODEL_CHECKPOINT, config=config)

    def transform_batch(examples):
        images = [img.convert("RGB") for img in examples["image"]]
        processed = processor(images=images, return_tensors="pt")
        examples["pixel_values"] = list(processed["pixel_values"])
        examples["labels"] = examples["label"]
        return examples

    train_ds.set_transform(transform_batch)
    val_ds.set_transform(transform_batch)
    test_ds.set_transform(transform_batch)

    def collate_fn(batch):
        pixel_values = torch.stack([torch.as_tensor(item["pixel_values"]) for item in batch])
        labels = torch.tensor([int(item["labels"]) for item in batch])
        return {"pixel_values": pixel_values, "labels": labels}

    training_args = TrainingArguments(
        output_dir="./trained_models/_checkpoints_vision_finetune",
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
        label_smoothing_factor=args.label_smoothing,
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=(device == "cuda"),
        dataloader_num_workers=0,
        report_to=[],
        seed=SEED,
        remove_unused_columns=False,
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

    logger.info("=" * 60)
    logger.info(f"FINE-TUNE BAŞLIYOR — {args.epochs} epoch, batch={args.batch_size} "
                f"(x{args.grad_accum} accum), dropout={args.dropout}")
    logger.info(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    logger.info("VRAM sınırlıysa: --batch-size 8 --grad-accum 4 ile tekrar dene.")
    logger.info("=" * 60)

    start_time = time.time()
    trainer.train()
    elapsed = time.time() - start_time
    logger.info(f"Eğitim tamamlandı. Süre: {elapsed / 60:.1f} dakika")

    log_history = pd.DataFrame(trainer.state.log_history)
    log_history.to_csv(f"{REPORTS_DIR}/vision_finetune_training_log.csv", index=False)

    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    logger.info(f"Fine-tune edilmiş model kaydedildi: {OUTPUT_DIR}")

    predictions = trainer.predict(test_ds)
    preds = np.argmax(predictions.predictions, axis=-1)
    labels = predictions.label_ids

    from sklearn.metrics import classification_report, accuracy_score, f1_score

    report_dict = classification_report(
        labels, preds, target_names=CLASS_NAMES, output_dict=True, zero_division=0
    )
    report_text = classification_report(labels, preds, target_names=CLASS_NAMES, zero_division=0)

    logger.info("\n" + report_text)
    logger.info(f"Test Accuracy: {accuracy_score(labels, preds):.4f} | "
                f"Test F1: {f1_score(labels, preds):.4f}")

    plot_confusion_matrix(labels, preds, f"{REPORTS_DIR}/vision_finetune_confusion_matrix.png")

    final_report = {
        "model_checkpoint": MODEL_CHECKPOINT,
        "datasets_used": DATASET_SOURCES,
        "dropout": args.dropout,
        "label_smoothing": args.label_smoothing,
        "train_size": len(train_ds), "val_size": len(val_ds), "test_size": len(test_ds),
        "selection_metric": "f1",
        "test_metrics": report_dict,
        "test_accuracy": accuracy_score(labels, preds),
        "test_f1": f1_score(labels, preds),
        "output_dir": OUTPUT_DIR,
    }
    with open(f"{REPORTS_DIR}/vision_finetune_evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(final_report, f, ensure_ascii=False, indent=2)
    logger.info(f"Değerlendirme raporu kaydedildi: {REPORTS_DIR}/vision_finetune_evaluation_report.json")

    logger.info("=" * 60)
    logger.info(f"TAMAMLANDI. Fine-tune edilmiş model: {OUTPUT_DIR}")
    logger.info("models.py otomatik olarak önce bu 'finetune' klasörünü arayacak.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()