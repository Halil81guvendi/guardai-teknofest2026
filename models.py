"""
GuardAI - models.py
====================
TEKNOFEST 2026 Sosyal İnovasyon Yarışması
"GuardAI: Hibrit Sosyal Medya Güvenlik ve Akıllı İçerik Moderasyon Motoru"

Bu modül 3 ana yapay zekâ katmanını içerir:
    1. TextModerationEngine  -> Toksisite / Nefret Söylemi / Siber Zorbalık / Dezenformasyon (NLP)
    2. VisualForensicsEngine -> Deepfake / Görsel Manipülasyon Tespiti (Vision Transformer)
    3. BotDetectionEngine    -> Davranışsal Bot / Sahte Hesap Tespiti (Scikit-Learn / RandomForest)

Tüm sınıflar FastAPI ya da Gradio katmanından bağımsız, tek başına
import edilip kullanılabilecek şekilde tasarlanmıştır.
"""

import io
import json
import logging
import os
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from PIL import Image

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="[GuardAI] %(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("GuardAI")


# =========================================================
# ORTAK VERİ YAPILARI (Dataclasses)
# =========================================================

@dataclass
class TextAnalysisResult:
    text: str
    toxicity_score: float
    hate_speech_score: float
    cyberbullying_risk: float
    disinformation_flag: bool
    overall_risk_score: float
    risk_label: str
    detail_scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class VisualAnalysisResult:
    manipulation_score: float
    is_manipulated: bool
    risk_label: str
    confidence: float
    detail_scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class BotAnalysisResult:
    bot_probability: float
    is_bot: bool
    risk_label: str
    feature_importance: Dict[str, float] = field(default_factory=dict)


# =========================================================
# 1. METİN ANALİZİ MOTORU (NLP - RoBERTa/BERT tabanlı)
# =========================================================

class TextModerationEngine:
    """
    HuggingFace üzerinden açık kaynak, çok-dilli (Türkçe dahil) toksisite ve
    nefret söylemi tespiti yapan transformer tabanlı modelleri kullanır.

    Kullanılan modeller (otomatik indirilir):
        - toksisite/nefret söylemi: 'unitary/multilingual-toxic-xlm-roberta'
          (fallback: 'unitary/toxic-bert')
    Model indirilemezse (internet yoksa) sistem otomatik olarak
    kural-tabanlı (lexicon-based) bir fallback motoruna düşer;
    böylece demo hiçbir zaman çökmez (production-grade dayanıklılık).
    """

    # Siber zorbalık / nefret söylemi için Türkçe + İngilizce anahtar kelime seti
    # (fallback motoru ve skor ağırlıklandırma için kullanılır)
    TOXIC_LEXICON = [
        "aptal", "salak", "gerizekalı", "şerefsiz", "namussuz", "geber",
        "öl", "ahmak", "haysiyetsiz", "hain", "terörist", "pislik",
        "idiot", "stupid", "kill yourself", "kys", "hate", "trash", "ugly",
        "moron", "worthless", "die", "loser",
    ]
    DISINFO_MARKERS = [
        "kesin kaynak", "gizli belge", "hükümet gizliyor", "bilim insanları itiraf etti",
        "paylaşılmasın diye", "sansürleniyor", "yayılmasın diye siliniyor",
        "doktorların söylemediği gerçek", "resmi açıklanmayan", "%100 kanıtlanmış",
    ]

    def __init__(self, model_name: str = None, device: Optional[str] = None):
        # Öncelik: 1) açıkça verilen model_name  2) eğitilmiş yerel GuardAI modeli (varsa)
        #          3) hazır (fine-tune edilmemiş) fallback model
        LOCAL_TRAINED_PATH = "./trained_models/guardai-text-turkish-toxic"
        if model_name is None:
            if os.path.isdir(LOCAL_TRAINED_PATH) and os.path.isfile(
                os.path.join(LOCAL_TRAINED_PATH, "config.json")
            ):
                model_name = LOCAL_TRAINED_PATH
                logger.info("Eğitilmiş GuardAI Türkçe toksisite modeli bulundu, bu kullanılacak.")
            else:
                model_name = "unitary/toxic-bert"
                logger.info(
                    "Eğitilmiş yerel model bulunamadı (./trained_models/guardai-text-turkish-toxic). "
                    "Hazır fallback model kullanılıyor. Kendi modelinizi eğitmek için train_text_model.py çalıştırın."
                )
        self.model_name = model_name
        self.device = device
        self.pipeline = None
        self._load_model()

    def _load_model(self):
        try:
            import torch
            from transformers import pipeline

            device_id = 0 if (self.device == "cuda" and torch.cuda.is_available()) else -1
            logger.info(f"Metin modeli yükleniyor: {self.model_name}")
            self.pipeline = pipeline(
                task="text-classification",
                model=self.model_name,
                top_k=None,
                truncation=True,
                device=device_id,
            )
            logger.info("Metin modeli başarıyla yüklendi.")
        except Exception as e:
            logger.warning(f"Transformer modeli yüklenemedi ({e}). Kural-tabanlı fallback motoru aktif.")
            self.pipeline = None

    # ---------------------------------------------------
    def _lexicon_fallback_score(self, text: str) -> float:
        """İnternet/model erişimi olmadığında kullanılan kural tabanlı skor."""
        text_l = text.lower()
        hits = sum(1 for w in self.TOXIC_LEXICON if w in text_l)
        score = min(1.0, hits * 0.35)
        return score

    def _disinformation_check(self, text: str) -> bool:
        text_l = text.lower()
        return any(marker in text_l for marker in self.DISINFO_MARKERS)

    # ---------------------------------------------------
    def analyze(self, text: str) -> TextAnalysisResult:
        if not text or not text.strip():
            return TextAnalysisResult(
                text=text, toxicity_score=0.0, hate_speech_score=0.0,
                cyberbullying_risk=0.0, disinformation_flag=False,
                overall_risk_score=0.0, risk_label="Bilgi Yok", detail_scores={},
            )

        detail_scores: Dict[str, float] = {}

        # Toksik OLMADIĞINI belirten etiketler (bunlar toksisite skoruna dahil edilmez)
        NON_TOXIC_LABELS = {"not_toxic", "non_toxic", "nontoxic", "neutral", "clean", "safe", "other"}

        if self.pipeline is not None:
            try:
                raw_output = self.pipeline(text[:512])
                # transformers top_k=None -> [[{'label':..,'score':..}, ...]]
                labels = raw_output[0] if isinstance(raw_output[0], list) else raw_output
                for item in labels:
                    detail_scores[item["label"].lower()] = round(float(item["score"]), 4)

                # Sadece GERÇEKTEN toksik anlamına gelen etiketleri değerlendirmeye al
                toxic_relevant_scores = {
                    k: v for k, v in detail_scores.items() if k not in NON_TOXIC_LABELS
                }

                if "toxic" in detail_scores:
                    # İkili (binary) model: doğrudan TOXIC etiketinin skorunu kullan
                    toxicity_score = detail_scores["toxic"]
                elif toxic_relevant_scores:
                    # Çok-etiketli model (örn. toxic-bert): toksik alt kategorilerin en yükseğini al
                    toxicity_score = max(toxic_relevant_scores.values())
                else:
                    toxicity_score = 0.0

                hate_speech_score = detail_scores.get(
                    "identity_hate", detail_scores.get("severe_toxic", toxicity_score * 0.8)
                )
            except Exception as e:
                logger.warning(f"Model inference hatası, fallback devrede: {e}")
                toxicity_score = self._lexicon_fallback_score(text)
                hate_speech_score = toxicity_score * 0.7
                detail_scores["lexicon_fallback"] = toxicity_score
        else:
            toxicity_score = self._lexicon_fallback_score(text)
            hate_speech_score = toxicity_score * 0.7
            detail_scores["lexicon_fallback"] = toxicity_score

        cyberbullying_risk = round(min(1.0, (toxicity_score * 0.6 + hate_speech_score * 0.4)), 4)
        disinformation_flag = self._disinformation_check(text)

        overall_risk_score = round(
            min(1.0, toxicity_score * 0.4 + hate_speech_score * 0.35 +
                cyberbullying_risk * 0.15 + (0.15 if disinformation_flag else 0.0)),
            4,
        )

        if overall_risk_score >= 0.75:
            risk_label = "🔴 YÜKSEK RİSK"
        elif overall_risk_score >= 0.4:
            risk_label = "🟡 ORTA RİSK"
        else:
            risk_label = "🟢 DÜŞÜK RİSK"

        return TextAnalysisResult(
            text=text,
            toxicity_score=round(float(toxicity_score), 4),
            hate_speech_score=round(float(hate_speech_score), 4),
            cyberbullying_risk=cyberbullying_risk,
            disinformation_flag=disinformation_flag,
            overall_risk_score=overall_risk_score,
            risk_label=risk_label,
            detail_scores=detail_scores,
        )


# =========================================================
# 2. GÖRSEL MANİPÜLASYON / DEEPFAKE TESPİT MOTORU (ViT)
# =========================================================

class VisualForensicsEngine:
    """
    Vision Transformer (ViT) tabanlı deepfake / görsel manipülasyon tespiti.
    HuggingFace'ten açık kaynak bir deepfake sınıflandırma modeli kullanılır.
    Model indirilemezse, klasik CV tabanlı (frekans/gürültü analizi + kenar
    tutarlılığı) bir fallback ile analiz üretilir; demo hiçbir zaman çökmez.
    """

    def __init__(self, model_name: str = None):
        LOCAL_TRAINED_PATH = "./trained_models/guardai-vision-deepfake"
        if model_name is None:
            if os.path.isdir(LOCAL_TRAINED_PATH) and os.path.isfile(
                os.path.join(LOCAL_TRAINED_PATH, "config.json")
            ):
                model_name = LOCAL_TRAINED_PATH
                logger.info("Eğitilmiş GuardAI görsel (deepfake) modeli bulundu, bu kullanılacak.")
            else:
                model_name = "prithivMLmods/deepfake-detector-model-v1"
                logger.info(
                    "Eğitilmiş yerel görsel model bulunamadı (./trained_models/guardai-vision-deepfake). "
                    "Hazır fallback model kullanılıyor. Kendi modelinizi eğitmek için train_vision_model.py çalıştırın."
                )
        self.model_name = model_name
        self.pipeline = None
        self._load_model()

    def _load_model(self):
        try:
            from transformers import pipeline
            logger.info(f"Görsel model yükleniyor: {self.model_name}")
            self.pipeline = pipeline(task="image-classification", model=self.model_name)
            logger.info("Görsel model başarıyla yüklendi.")
        except Exception as e:
            logger.warning(f"ViT modeli yüklenemedi ({e}). Klasik CV fallback motoru aktif.")
            self.pipeline = None

    # ---------------------------------------------------
    def _cv_fallback_score(self, image: Image.Image) -> float:
        """
        Model yokken; JPEG-blok tutarsızlığı ve gürültü varyansı üzerinden
        basit, deterministik olmayan (ama tutarlı) bir manipülasyon skoru üretir.
        Gerçek üretim ortamında ELA (Error Level Analysis) / frekans analizi
        ile değiştirilmesi önerilir.
        """
        try:
            import cv2
            img_arr = np.array(image.convert("RGB"))
            gray = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)

            # Laplacian varyansı: aşırı düşük veya aşırı yüksek değerler
            # "üzerinde oynanmış" görsellerde anormal olma eğilimindedir.
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

            # Kenar yoğunluğu tutarlılığı
            edges = cv2.Canny(gray, 100, 200)
            edge_density = np.sum(edges > 0) / edges.size

            # Basit normalize edilmiş kompozit skor (0-1)
            norm_lap = 1.0 - min(1.0, laplacian_var / 1500.0)
            norm_edge = min(1.0, edge_density * 6.0)
            score = float(np.clip((norm_lap * 0.55 + norm_edge * 0.45), 0.03, 0.97))
            return score
        except Exception as e:
            logger.warning(f"CV fallback hatası: {e}")
            return 0.5  # belirsiz durum

    # ---------------------------------------------------
    def analyze(self, image: Image.Image) -> VisualAnalysisResult:
        if image is None:
            return VisualAnalysisResult(
                manipulation_score=0.0, is_manipulated=False,
                risk_label="Görsel Yok", confidence=0.0, detail_scores={},
            )

        detail_scores: Dict[str, float] = {}

        if self.pipeline is not None:
            try:
                results = self.pipeline(image)
                for item in results:
                    detail_scores[item["label"].lower()] = round(float(item["score"]), 4)

                # Gerçek/otantik olduğunu belirten etiketleri hariç tut
                REAL_LABELS = {"real", "realism", "authentic", "genuine", "original"}
                fake_relevant_scores = {
                    k: v for k, v in detail_scores.items() if k not in REAL_LABELS
                }
                fake_keys = [k for k in fake_relevant_scores if any(
                    t in k for t in ["fake", "manipulat", "spoof", "ai", "deepfake"]
                )]
                if fake_keys:
                    manipulation_score = max(detail_scores[k] for k in fake_keys)
                elif fake_relevant_scores:
                    manipulation_score = max(fake_relevant_scores.values())
                else:
                    manipulation_score = 1.0 - max(detail_scores.values(), default=0.5)
                confidence = max(detail_scores.values())
            except Exception as e:
                logger.warning(f"Görsel model inference hatası, fallback devrede: {e}")
                manipulation_score = self._cv_fallback_score(image)
                confidence = 0.6
                detail_scores["cv_fallback_score"] = manipulation_score
        else:
            manipulation_score = self._cv_fallback_score(image)
            confidence = 0.6
            detail_scores["cv_fallback_score"] = manipulation_score

        is_manipulated = manipulation_score >= 0.5

        if manipulation_score >= 0.75:
            risk_label = "🔴 YÜKSEK OLASILIKLA MANİPÜLE EDİLMİŞ"
        elif manipulation_score >= 0.45:
            risk_label = "🟡 ŞÜPHELİ / İNCELENMELİ"
        else:
            risk_label = "🟢 OTANTİK GÖRÜNÜYOR"

        return VisualAnalysisResult(
            manipulation_score=round(float(manipulation_score), 4),
            is_manipulated=is_manipulated,
            risk_label=risk_label,
            confidence=round(float(confidence), 4),
            detail_scores=detail_scores,
        )


# =========================================================
# 3. DAVRANIŞSAL BOT TESPİT MOTORU (Scikit-Learn RandomForest)
# =========================================================

class BotDetectionEngine:
    """
    Kullanıcı hesabı davranışsal/meta verilerinden RandomForestClassifier ile
    bot / sahte hesap olasılığı hesaplar.

    Öncelik sırası:
        1) ./trained_models/guardai-bot-detector/ altında train_bot_model.py ile
           GERÇEK Twitter/X hesap verisiyle (airt-ml/twitter-human-bots, 37.438
           hesap) eğitilmiş model varsa onu yükler. Bu modelin özellik seti
           gerçek API alanlarına dayanır (favourites_count, followers_count,
           account_age_days, verified, vb.)
        2) Bulunamazsa, istatistiksel olarak ayrıştırılabilir SENTETİK bir
           veri setiyle anında bir fallback model eğitir (demo hiçbir zaman
           çökmez).

    self.feature_spec, hangi mod aktifse ona ait özellik listesini + UI
    slider parametrelerini (label/min/max/default/step) verir; Gradio arayüzü
    bu listeyi okuyarak formu dinamik olarak oluşturur.
    """

    LOCAL_TRAINED_PATH = "./trained_models/guardai-bot-detector"

    # Gerçek veriyle eğitilmiş model için UI slider spesifikasyonu
    REAL_FEATURE_SPEC = [
        {"name": "favourites_count", "label": "Beğenilen Gönderi Sayısı (favourites)", "min": 0, "max": 200000, "default": 500, "step": 10},
        {"name": "followers_count", "label": "Takipçi Sayısı", "min": 0, "max": 5_000_000, "default": 800, "step": 10},
        {"name": "friends_count", "label": "Takip Edilen Sayısı", "min": 0, "max": 50000, "default": 400, "step": 10},
        {"name": "statuses_count", "label": "Toplam Gönderi (statuses) Sayısı", "min": 0, "max": 500000, "default": 1200, "step": 10},
        {"name": "average_tweets_per_day", "label": "Günlük Ortalama Gönderi", "min": 0.0, "max": 200.0, "default": 2.0, "step": 0.1},
        {"name": "account_age_days", "label": "Hesap Yaşı (gün)", "min": 1, "max": 6000, "default": 900, "step": 1},
        {"name": "verified", "label": "Doğrulanmış Hesap mı?", "min": 0, "max": 1, "default": 0, "step": 1},
        {"name": "geo_enabled", "label": "Konum Paylaşımı Açık mı?", "min": 0, "max": 1, "default": 0, "step": 1},
        {"name": "default_profile", "label": "Varsayılan (özelleştirilmemiş) Profil mi?", "min": 0, "max": 1, "default": 0, "step": 1},
        {"name": "default_profile_image", "label": "Varsayılan Profil Fotoğrafı mı?", "min": 0, "max": 1, "default": 0, "step": 1},
        {"name": "description_length", "label": "Biyografi Uzunluğu (karakter)", "min": 0, "max": 200, "default": 60, "step": 1},
        {"name": "has_location", "label": "Konum Bilgisi Girilmiş mi?", "min": 0, "max": 1, "default": 1, "step": 1},
        {"name": "screen_name_digit_ratio", "label": "Kullanıcı Adındaki Rakam Oranı", "min": 0.0, "max": 1.0, "default": 0.0, "step": 0.01},
    ]

    # Eğitilmiş model bulunamazsa kullanılan eski sentetik fallback şeması
    SYNTHETIC_FEATURE_SPEC = [
        {"name": "posts_per_day", "label": "Günlük Ortalama Paylaşım Sayısı", "min": 0, "max": 300, "default": 25, "step": 1},
        {"name": "avg_time_between_posts_sec", "label": "Paylaşımlar Arası Ort. Süre (sn)", "min": 1, "max": 20000, "default": 300, "step": 1},
        {"name": "follower_following_ratio", "label": "Takipçi / Takip Oranı", "min": 0.0, "max": 20.0, "default": 0.4, "step": 0.01},
        {"name": "account_age_days", "label": "Hesap Yaşı (gün)", "min": 1, "max": 4000, "default": 60, "step": 1},
        {"name": "content_diversity_score", "label": "İçerik Çeşitliliği Skoru", "min": 0.0, "max": 1.0, "default": 0.35, "step": 0.01},
        {"name": "night_activity_ratio", "label": "Gece Aktivite Oranı (00-06)", "min": 0.0, "max": 1.0, "default": 0.45, "step": 0.01},
        {"name": "duplicate_content_ratio", "label": "Tekrarlayan İçerik Oranı", "min": 0.0, "max": 1.0, "default": 0.4, "step": 0.01},
        {"name": "profile_completeness_score", "label": "Profil Doluluk Skoru", "min": 0.0, "max": 1.0, "default": 0.4, "step": 0.01},
    ]

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model = None
        self.scaler = None
        self.using_trained_model = False
        self.feature_names: List[str] = []
        self.feature_spec: List[dict] = []

        if self._try_load_trained_model():
            self.using_trained_model = True
            self.feature_spec = self.REAL_FEATURE_SPEC
            # Model, UI'da gösterilmeyen türetilmiş "followers_friends_ratio"
            # özelliğini de bekliyor (train_bot_model.py ile aynı sırada).
            # Kaydedilmiş feature_names.json varsa onu esas al, yoksa varsayılana düş.
            self.model_feature_names = self._loaded_feature_names or [
                "favourites_count", "followers_count", "friends_count",
                "followers_friends_ratio", "statuses_count", "average_tweets_per_day",
                "account_age_days", "verified", "geo_enabled", "default_profile",
                "default_profile_image", "description_length", "has_location",
                "screen_name_digit_ratio",
            ]
            self.feature_names = [f["name"] for f in self.REAL_FEATURE_SPEC]
            self.sample_accounts = self._load_sample_accounts()
            logger.info("Eğitilmiş GuardAI bot tespit modeli (gerçek Twitter verisi) bulundu, bu kullanılacak.")
            if self.sample_accounts:
                logger.info(
                    f"Jüri demosu için {len(self.sample_accounts)} gerçek test hesabı örneği yüklendi."
                )
        else:
            logger.info(
                "Eğitilmiş yerel bot modeli bulunamadı (./trained_models/guardai-bot-detector). "
                "Sentetik fallback model eğitiliyor. Gerçek veriyle eğitmek için train_bot_model.py çalıştırın."
            )
            self.feature_spec = self.SYNTHETIC_FEATURE_SPEC
            self.feature_names = [f["name"] for f in self.SYNTHETIC_FEATURE_SPEC]
            self.model_feature_names = self.feature_names
            self._train_synthetic_model()
            self.sample_accounts = []

    # ---------------------------------------------------
    def _load_sample_accounts(self) -> list:
        import os

        path = os.path.join(self.LOCAL_TRAINED_PATH, "sample_test_accounts.json")
        if not os.path.isfile(path):
            return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Örnek test hesapları yüklenemedi: {e}")
            return []

    def get_random_sample_account(self) -> Optional[dict]:
        """Jüri demosu için: test setinden rastgele GERÇEK bir hesap döndürür
        (sadece sayısal özellikler + gerçek etiket, kimlik bilgisi içermez)."""
        if not self.sample_accounts:
            return None
        import random

        return random.choice(self.sample_accounts)

    # ---------------------------------------------------
    def _try_load_trained_model(self) -> bool:
        import os

        self._loaded_feature_names = None
        model_path = os.path.join(self.LOCAL_TRAINED_PATH, "model.joblib")
        scaler_path = os.path.join(self.LOCAL_TRAINED_PATH, "scaler.joblib")
        features_path = os.path.join(self.LOCAL_TRAINED_PATH, "feature_names.json")

        if not (os.path.isfile(model_path) and os.path.isfile(scaler_path)):
            return False

        try:
            import joblib

            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            if os.path.isfile(features_path):
                with open(features_path, "r", encoding="utf-8") as f:
                    self._loaded_feature_names = json.load(f)
            return True
        except Exception as e:
            logger.warning(f"Eğitilmiş bot modeli yüklenemedi ({e}). Sentetik fallback'e dönülüyor.")
            self.model = None
            self.scaler = None
            return False

    # ---------------------------------------------------
    def _generate_synthetic_dataset(self, n_samples: int = 4000) -> pd.DataFrame:
        rng = np.random.default_rng(self.random_state)
        n_bot = n_samples // 2
        n_human = n_samples - n_bot

        bot_df = pd.DataFrame({
            "posts_per_day": rng.gamma(shape=8, scale=6, size=n_bot),
            "avg_time_between_posts_sec": rng.exponential(scale=30, size=n_bot),
            "follower_following_ratio": rng.exponential(scale=0.15, size=n_bot),
            "account_age_days": rng.exponential(scale=25, size=n_bot),
            "content_diversity_score": rng.beta(1.2, 6, size=n_bot),
            "night_activity_ratio": rng.beta(5, 2, size=n_bot),
            "duplicate_content_ratio": rng.beta(6, 1.5, size=n_bot),
            "profile_completeness_score": rng.beta(1.5, 5, size=n_bot),
            "label": 1,
        })

        human_df = pd.DataFrame({
            "posts_per_day": rng.gamma(shape=1.5, scale=1.2, size=n_human),
            "avg_time_between_posts_sec": rng.exponential(scale=7000, size=n_human),
            "follower_following_ratio": rng.gamma(shape=2, scale=1.2, size=n_human),
            "account_age_days": rng.gamma(shape=4, scale=180, size=n_human),
            "content_diversity_score": rng.beta(5, 1.5, size=n_human),
            "night_activity_ratio": rng.beta(2, 6, size=n_human),
            "duplicate_content_ratio": rng.beta(1.2, 6, size=n_human),
            "profile_completeness_score": rng.beta(6, 1.5, size=n_human),
            "label": 0,
        })

        full_df = pd.concat([bot_df, human_df], ignore_index=True)
        full_df["follower_following_ratio"] = full_df["follower_following_ratio"].clip(0, 50)
        full_df["account_age_days"] = full_df["account_age_days"].clip(1, 4000)
        full_df["posts_per_day"] = full_df["posts_per_day"].clip(0, 300)
        return full_df.sample(frac=1.0, random_state=self.random_state).reset_index(drop=True)

    # ---------------------------------------------------
    def _train_synthetic_model(self):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import accuracy_score, f1_score

        logger.info("Bot tespit modeli sentetik veriyle eğitiliyor...")
        df = self._generate_synthetic_dataset()
        X = df[self.feature_names]
        y = df["label"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )

        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.model = RandomForestClassifier(
            n_estimators=250,
            max_depth=10,
            min_samples_leaf=4,
            random_state=self.random_state,
            n_jobs=-1,
            class_weight="balanced",
        )
        self.model.fit(X_train_scaled, y_train)

        preds = self.model.predict(X_test_scaled)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        logger.info(f"Bot tespit modeli eğitildi. Test Accuracy: {acc:.4f} | F1: {f1:.4f}")

    # ---------------------------------------------------
    def analyze(self, **feature_values) -> BotAnalysisResult:
        """
        feature_values: self.feature_names (UI'da gösterilen alanlar) içindeki
        isimlerle eşleşen anahtar-değer çiftleri. Eğitilmiş gerçek model
        kullanılıyorsa, UI'da gösterilmeyen türetilmiş özellikler (örn.
        followers_friends_ratio) burada otomatik hesaplanıp modele eklenir.
        """
        try:
            ui_values = {name: feature_values[name] for name in self.feature_names}
        except KeyError as e:
            raise ValueError(
                f"Eksik özellik: {e}. Beklenen özellikler: {self.feature_names}"
            )

        if self.using_trained_model:
            # Türetilmiş özelliği hesapla ve model_feature_names sırasına göre diz
            derived = dict(ui_values)
            derived["followers_friends_ratio"] = derived.get("followers_count", 0) / (
                derived.get("friends_count", 0) + 1
            )
            try:
                ordered_values = [derived[name] for name in self.model_feature_names]
            except KeyError as e:
                raise ValueError(f"Model için eksik türetilmiş özellik: {e}")
        else:
            ordered_values = [ui_values[name] for name in self.feature_names]

        features = np.array([ordered_values], dtype=float)
        features_scaled = self.scaler.transform(features)

        bot_probability = float(self.model.predict_proba(features_scaled)[0][1])
        is_bot = bot_probability >= 0.5

        if bot_probability >= 0.75:
            risk_label = "🔴 YÜKSEK OLASILIKLA BOT"
        elif bot_probability >= 0.45:
            risk_label = "🟡 ŞÜPHELİ HESAP"
        else:
            risk_label = "🟢 GERÇEK KULLANICI"

        importances = self.model.feature_importances_
        feature_importance = {
            name: round(float(imp), 4)
            for name, imp in sorted(
                zip(self.model_feature_names, importances), key=lambda x: x[1], reverse=True
            )
        }

        return BotAnalysisResult(
            bot_probability=round(bot_probability, 4),
            is_bot=is_bot,
            risk_label=risk_label,
            feature_importance=feature_importance,
        )


# =========================================================
# HIZLI TEST (python models.py ile doğrudan çalıştırılabilir)
# =========================================================

if __name__ == "__main__":
    print("=" * 60)
    print("GuardAI Model Katmanı - Bağımsız Test")
    print("=" * 60)

    text_engine = TextModerationEngine()
    result_text = text_engine.analyze("Sen tam bir aptalsın, geber git buradan!")
    print("\n[Metin Analizi Sonucu]")
    print(result_text)

    bot_engine = BotDetectionEngine()
    result_bot = bot_engine.analyze(
        posts_per_day=85, avg_time_between_posts_sec=12,
        follower_following_ratio=0.05, account_age_days=9,
        content_diversity_score=0.08, night_activity_ratio=0.7,
        duplicate_content_ratio=0.85, profile_completeness_score=0.1,
    )
    print("\n[Bot Tespit Sonucu]")
    print(result_bot)

    print("\nNot: Görsel motoru test etmek için bir PIL.Image nesnesi gereklidir.")