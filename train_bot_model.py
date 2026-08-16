"""
GuardAI - train_bot_model.py
==============================
TEKNOFEST 2026 Sosyal İnovasyon Yarışması

RandomForestClassifier modelini GERÇEK Twitter/X hesap verisi üzerinde eğitir.

Veri Seti: airt-ml/twitter-human-bots (HuggingFace Datasets)
    - 37.438 gerçek Twitter hesabı (bot / human etiketli)
    - Gerçek hesap meta verisi: takipçi sayısı, hesap yaşı, günlük tweet
      frekansı, profil doluluk durumu vb.

Kullanım:
    python train_bot_model.py
    python train_bot_model.py --n-estimators 300 --max-depth 12

Çıktılar:
    ./trained_models/guardai-bot-detector/model.joblib        -> Eğitilmiş RandomForest
    ./trained_models/guardai-bot-detector/scaler.joblib       -> StandardScaler
    ./trained_models/guardai-bot-detector/feature_names.json  -> Özellik sırası/isimleri
    ./reports/bot_model_evaluation_report.json                -> Detaylı metrikler
    ./reports/bot_model_confusion_matrix.png                  -> Karışıklık matrisi
    ./reports/bot_model_feature_importance.png                -> Özellik önem grafiği
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="[GuardAI-BotTrain] %(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("GuardAI-BotTrain")

DATASET_NAME = "airt-ml/twitter-human-bots"
OUTPUT_DIR = "./trained_models/guardai-bot-detector"
REPORTS_DIR = "./reports"
SEED = 42

# Gerçek Twitter API alanlarından türetilen, model tarafından kullanılacak
# nihai özellik seti (sırası önemli - inference'ta da aynı sırada olmalı)
FEATURE_NAMES = [
    "favourites_count",
    "followers_count",
    "friends_count",
    "followers_friends_ratio",
    "statuses_count",
    "average_tweets_per_day",
    "account_age_days",
    "verified",
    "geo_enabled",
    "default_profile",
    "default_profile_image",
    "description_length",
    "has_location",
    "screen_name_digit_ratio",
]


def parse_args():
    parser = argparse.ArgumentParser(description="GuardAI Bot Tespit Modeli Eğitimi (Gerçek Veri)")
    parser.add_argument("--n-estimators", type=int, default=300, help="RandomForest ağaç sayısı")
    parser.add_argument("--max-depth", type=int, default=12, help="Maksimum ağaç derinliği")
    parser.add_argument("--min-samples-leaf", type=int, default=3, help="Yaprak başına min örnek")
    return parser.parse_args()


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ham Twitter API alanlarından model özelliklerini türetir."""
    out = pd.DataFrame()

    out["favourites_count"] = df["favourites_count"].fillna(0)
    out["followers_count"] = df["followers_count"].fillna(0)
    out["friends_count"] = df["friends_count"].fillna(0)
    out["followers_friends_ratio"] = out["followers_count"] / (out["friends_count"] + 1)
    out["statuses_count"] = df["statuses_count"].fillna(0)
    out["average_tweets_per_day"] = df["average_tweets_per_day"].fillna(0)
    out["account_age_days"] = df["account_age_days"].fillna(df["account_age_days"].median())

    out["verified"] = df["verified"].astype(int)
    out["geo_enabled"] = df["geo_enabled"].astype(int)
    out["default_profile"] = df["default_profile"].astype(int)
    out["default_profile_image"] = df["default_profile_image"].astype(int)

    out["description_length"] = df["description"].fillna("").astype(str).str.len()
    out["has_location"] = (
        df["location"].fillna("unknown").astype(str).str.lower().ne("unknown")
    ).astype(int)

    def digit_ratio(name):
        name = str(name)
        if len(name) == 0:
            return 0.0
        return sum(c.isdigit() for c in name) / len(name)

    out["screen_name_digit_ratio"] = df["screen_name"].fillna("").apply(digit_ratio)

    return out


def load_and_prepare_dataset():
    from datasets import load_dataset
    from sklearn.model_selection import train_test_split

    logger.info(f"Veri seti indiriliyor: {DATASET_NAME}")
    raw = load_dataset(DATASET_NAME, split="train")
    df = raw.to_pandas()
    logger.info(f"Ham veri seti yüklendi: {len(df)} hesap")

    df = df.dropna(subset=["account_type"])
    df["label"] = (df["account_type"].astype(str).str.lower() == "bot").astype(int)

    logger.info(
        f"Sınıf dağılımı -> Bot: {int(df['label'].sum())} | "
        f"Human: {int((df['label'] == 0).sum())}"
    )

    X = engineer_features(df)
    y = df["label"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp
    )

    logger.info(
        f"Bölünme -> Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}"
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def plot_confusion_matrix(y_true, y_pred, save_path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Greens")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Human", "Bot"])
    ax.set_yticklabels(["Human", "Bot"])
    ax.set_xlabel("Tahmin Edilen")
    ax.set_ylabel("Gerçek Etiket")
    ax.set_title("GuardAI Bot Tespit Modeli - Karışıklık Matrisi")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"Karışıklık matrisi kaydedildi: {save_path}")


def plot_feature_importance(model, feature_names, save_path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(
        [feature_names[i] for i in order][::-1],
        [importances[i] for i in order][::-1],
        color="#6366f1",
    )
    ax.set_xlabel("Önem Skoru")
    ax.set_title("GuardAI Bot Tespit Modeli - Özellik Önem Sıralaması")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"Özellik önem grafiği kaydedildi: {save_path}")


def main():
    args = parse_args()

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(REPORTS_DIR).mkdir(parents=True, exist_ok=True)

    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score,
        classification_report, roc_auc_score,
    )

    X_train, X_val, X_test, y_train, y_val, y_test = load_and_prepare_dataset()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train[FEATURE_NAMES])
    X_val_scaled = scaler.transform(X_val[FEATURE_NAMES])
    X_test_scaled = scaler.transform(X_test[FEATURE_NAMES])

    logger.info("=" * 60)
    logger.info(
        f"EĞİTİM BAŞLIYOR — RandomForest (n_estimators={args.n_estimators}, "
        f"max_depth={args.max_depth}, min_samples_leaf={args.min_samples_leaf})"
    )
    logger.info(f"Train örnek sayısı: {len(X_train)}")
    logger.info("=" * 60)

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        random_state=SEED,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(X_train_scaled, y_train)

    val_preds = model.predict(X_val_scaled)
    logger.info(
        f"Validation -> Accuracy: {accuracy_score(y_val, val_preds):.4f} | "
        f"F1: {f1_score(y_val, val_preds):.4f}"
    )

    # -----------------------------------------------------------
    # Nihai test değerlendirmesi
    # -----------------------------------------------------------
    test_preds = model.predict(X_test_scaled)
    test_probs = model.predict_proba(X_test_scaled)[:, 1]

    report_dict = classification_report(
        y_test, test_preds, target_names=["human", "bot"], output_dict=True, zero_division=0
    )
    report_text = classification_report(
        y_test, test_preds, target_names=["human", "bot"], zero_division=0
    )
    auc = roc_auc_score(y_test, test_probs)

    logger.info("\n" + report_text)
    logger.info(
        f"Test Accuracy: {accuracy_score(y_test, test_preds):.4f} | "
        f"Test F1: {f1_score(y_test, test_preds):.4f} | "
        f"Test ROC-AUC: {auc:.4f}"
    )

    plot_confusion_matrix(y_test, test_preds, f"{REPORTS_DIR}/bot_model_confusion_matrix.png")
    plot_feature_importance(model, FEATURE_NAMES, f"{REPORTS_DIR}/bot_model_feature_importance.png")

    # -----------------------------------------------------------
    # Kaydet
    # -----------------------------------------------------------
    joblib.dump(model, f"{OUTPUT_DIR}/model.joblib")
    joblib.dump(scaler, f"{OUTPUT_DIR}/scaler.joblib")
    with open(f"{OUTPUT_DIR}/feature_names.json", "w", encoding="utf-8") as f:
        json.dump(FEATURE_NAMES, f, ensure_ascii=False, indent=2)
    logger.info(f"Model, scaler ve özellik listesi kaydedildi: {OUTPUT_DIR}")

    # -----------------------------------------------------------
    # Jüri demosu için: test setinden (modelin hiç eğitimde görmediği,
    # GERÇEK hesaplardan) gizlilik-güvenli bir örnek havuzu kaydet.
    # Sadece sayısal/mühendislik özellikleri + gerçek etiket saklanır;
    # kullanıcı adı, biyografi metni gibi kimlik belirten alanlar HİÇ
    # kaydedilmez.
    # -----------------------------------------------------------
    n_samples = min(300, len(X_test))
    sample_idx_for_demo = np.random.RandomState(SEED).choice(len(X_test), size=n_samples, replace=False)
    sample_accounts = []
    for idx in sample_idx_for_demo:
        row = X_test.iloc[idx]
        sample_accounts.append({
            "features": {name: float(row[name]) for name in FEATURE_NAMES},
            "true_label": "bot" if y_test.iloc[idx] == 1 else "human",
        })
    with open(f"{OUTPUT_DIR}/sample_test_accounts.json", "w", encoding="utf-8") as f:
        json.dump(sample_accounts, f, ensure_ascii=False, indent=2)
    logger.info(
        f"Jüri demosu için {n_samples} gerçek test hesabı (gizlilik-güvenli, "
        f"sadece sayısal özellikler + gerçek etiket) kaydedildi: "
        f"{OUTPUT_DIR}/sample_test_accounts.json"
    )

    final_report = {
        "model_type": "RandomForestClassifier",
        "dataset": DATASET_NAME,
        "feature_names": FEATURE_NAMES,
        "train_size": len(X_train),
        "val_size": len(X_val),
        "test_size": len(X_test),
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "min_samples_leaf": args.min_samples_leaf,
        "test_metrics": report_dict,
        "test_accuracy": accuracy_score(y_test, test_preds),
        "test_f1": f1_score(y_test, test_preds),
        "test_roc_auc": auc,
        "output_dir": OUTPUT_DIR,
    }
    with open(f"{REPORTS_DIR}/bot_model_evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(final_report, f, ensure_ascii=False, indent=2)
    logger.info(f"Değerlendirme raporu kaydedildi: {REPORTS_DIR}/bot_model_evaluation_report.json")

    # -----------------------------------------------------------
    # Örnek çıkarım testi
    # -----------------------------------------------------------
    logger.info("=" * 60)
    logger.info("ÖRNEK ÇIKARIM TESTİ")
    logger.info("=" * 60)
    sample_idx = np.random.RandomState(SEED).choice(len(X_test), size=5, replace=False)
    for idx in sample_idx:
        row = X_test.iloc[[idx]]
        true_label = "BOT" if y_test.iloc[idx] == 1 else "HUMAN"
        pred_prob = model.predict_proba(scaler.transform(row[FEATURE_NAMES]))[0][1]
        pred_label = "BOT" if pred_prob >= 0.5 else "HUMAN"
        match = "✓" if pred_label == true_label else "✗"
        logger.info(
            f"  Gerçek: {true_label:6s} | Tahmin: {pred_label:6s} "
            f"(p={pred_prob:.3f}) {match}"
        )

    logger.info("=" * 60)
    logger.info(f"TAMAMLANDI. Eğitilmiş model burada: {OUTPUT_DIR}")
    logger.info("models.py zaten bu klasörü otomatik arayacak şekilde güncellenecek.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()