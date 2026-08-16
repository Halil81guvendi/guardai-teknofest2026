# 🛡️ GuardAI

**Hibrit Sosyal Medya Güvenlik ve Akıllı İçerik Moderasyon Motoru**

TEKNOFEST 2026 · NSosyal İnovasyon Yarışması · İnovasyon Dikeyi: *Sosyal Yapay Zekâ*

---

## Proje Hakkında

GuardAI, sosyal medya platformlarının karşı karşıya olduğu üç temel güvenlik tehdidine — **toksik dil/nefret söylemi ve dezenformasyon**, **deepfake/görsel manipülasyon** ve **bot ağları/sahte hesaplar** — eş zamanlı çözüm üreten hibrit bir yapay zekâ motorudur.

Sistem, birbirinden bağımsız fakat ortak bir karar motorunda birleşen üç yapay zekâ katmanından oluşur:

| Katman | Model | Veri Seti | Test Sonucu |
|---|---|---|---|
| **Metin Analizi** | BERTurk (fine-tuned) | 2 birleştirilmiş Türkçe veri seti, 124.920 satır | F1: %96.21, ROC-AUC: %99.24 |
| **Görsel Adli Analiz** | Vision Transformer (ViT) | 190.335 görsel (GAN-üretimi deepfake + gerçek) | F1: %93.87 |
| **Bot Tespiti** | RandomForestClassifier | 37.438 gerçek Twitter/X hesabı | ROC-AUC: %92.95, F1: %80.16 |

Tüm modeller **gerçek, açık kaynaklı veri setleriyle uçtan uca eğitilmiş**, bağımsız test setlerinde doğrulanmış ve internetten alınan hiç görülmemiş içeriklerle canlı olarak sınanmıştır.

## Mimari

```
guardai/
├── models.py                       # AI motor katmanı (3 model sınıfı)
├── app.py                          # Gradio demo arayüzü (3 sekme)
├── requirements.txt                # Bağımlılıklar
├── train_text_model.py             # Metin modeli - ilk eğitim
├── train_text_model_finetune.py    # Metin modeli - çoklu veri seti fine-tune
├── train_vision_model.py           # Görsel model - ilk eğitim
├── train_vision_model_finetune.py  # Görsel model - fine-tune denemesi
└── train_bot_model.py              # Bot tespit modeli eğitimi
```

`models.py`, üretime hazır (production-ready) modüler bir yapıda tasarlanmıştır: her model sınıfı (`TextModerationEngine`, `VisualForensicsEngine`, `BotDetectionEngine`) önce yerel olarak eğitilmiş bir model arar (önce fine-tune edilmiş versiyon, sonra v1, sonra hazır fallback model), bu da sistemin FastAPI tabanlı bir REST servisine kolayca dönüştürülebilmesini sağlar.

## Kurulum ve Çalıştırma

```bash
pip install -r requirements.txt

# Modelleri eğitmek için (opsiyonel, önceden eğitilmiş modeller yoksa
# sistem otomatik olarak hazır fallback modellere düşer):
python train_text_model.py
python train_vision_model.py
python train_bot_model.py

# Demo arayüzünü başlatmak için:
python app.py
```

Arayüz `http://localhost:7860` üzerinde açılır.

## Teknoloji Yığını

- **PyTorch** + **HuggingFace Transformers** — dil ve görüntü modelleri
- **scikit-learn** — RandomForest sınıflandırıcısı
- **Gradio** — etkileşimli demo arayüzü
- **HuggingFace Datasets** — veri toplama/işleme boru hattı
- **joblib / safetensors** — model kalıcılığı

## Metodolojik Notlar

- Overfitting önlemleri: dropout artırımı, label smoothing, erken durdurma (early stopping) + en iyi checkpoint seçimi
- Metin modeli, tek bir veri setinin etiketleme önyargısına aşırı uyum sağlamaması için 2 bağımsız Türkçe veri setinin birleştirilmesiyle fine-tune edilmiştir
- Gizlilik-güvenli demo özelliği: bot tespit modülü, test setinden kimlik bilgisi içermeyen gerçek hesap örnekleriyle canlı doğrulama yapılabilmesini sağlar

Detaylı metodoloji, veri kaynakları ve sınırlamalar için proje teknik raporuna bakınız.

## Lisans

MIT License — bkz. [LICENSE](./LICENSE)

## Takım

TEKNOFEST 2026 NSosyal İnovasyon Yarışması kapsamında geliştirilmiştir.
