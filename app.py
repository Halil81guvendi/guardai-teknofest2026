"""
GuardAI - app.py
==================
TEKNOFEST 2026 Sosyal İnovasyon Yarışması
"GuardAI: Hibrit Sosyal Medya Güvenlik ve Akıllı İçerik Moderasyon Motoru"

Jüri sunumuna uygun, 3 sekmeli, canlı çalışan Gradio arayüzü:
    Tab 1 -> Metin Moderasyonu   (NLP)
    Tab 2 -> Görsel Moderasyon   (Computer Vision / ViT)
    Tab 3 -> Bot & Akış Analizi  (Davranışsal ML)

Çalıştırmak için:
    pip install -r requirements.txt
    python app.py

FastAPI ile üretime taşımak için bu dosyadaki `analyze_*` fonksiyonları
doğrudan FastAPI endpoint'lerine (örn. api.py -> /api/v1/text-analyze)
bağlanabilecek şekilde saf Python fonksiyonları olarak tasarlanmıştır.
"""

import logging

import gradio as gr
import pandas as pd
from PIL import Image

from models import BotDetectionEngine, TextModerationEngine, VisualForensicsEngine

logging.basicConfig(level=logging.INFO, format="[GuardAI-UI] %(asctime)s - %(message)s")
logger = logging.getLogger("GuardAI-UI")

# =========================================================
# MODELLERİ TEK SEFER YÜKLE (Singleton - uygulama açılışında)
# =========================================================
logger.info("GuardAI motorları başlatılıyor, lütfen bekleyin...")
text_engine = TextModerationEngine()
visual_engine = VisualForensicsEngine()
bot_engine = BotDetectionEngine()
logger.info("Tüm motorlar hazır. Arayüz başlatılıyor...")


# =========================================================
# TAB 1 — METİN MODERASYONU İŞLEV KATMANI
# =========================================================

def analyze_text(text: str):
    if not text or not text.strip():
        return (
            "⚪ Lütfen analiz edilecek bir metin girin.",
            0, 0, 0,
            "-",
            {},
        )

    result = text_engine.analyze(text)

    verdict = (
        f"### {result.risk_label}\n"
        f"**Genel Risk Skoru:** `{result.overall_risk_score * 100:.1f} / 100`\n\n"
        f"{'⚠️ **Dezenformasyon işareti tespit edildi.**' if result.disinformation_flag else '✅ Dezenformasyon işareti bulunamadı.'}"
    )

    detail_df = pd.DataFrame({
        "Metrik": list(result.detail_scores.keys()) or ["skor yok"],
        "Değer": [round(v, 4) for v in result.detail_scores.values()] or [0],
    })

    return (
        verdict,
        round(result.toxicity_score * 100, 1),
        round(result.hate_speech_score * 100, 1),
        round(result.cyberbullying_risk * 100, 1),
        "EVET ⚠️" if result.disinformation_flag else "HAYIR ✅",
        detail_df,
    )


TEXT_EXAMPLES = [
    ["Bugün hava çok güzel, parkta güzel bir yürüyüş yaptım."],
    ["Sen tam bir aptalsın, kimse seni sevmiyor, geber git."],
    ["DİKKAT: Hükümet gizliyor! Bu gizli belge sansürlenmeden önce paylaşın, doktorların söylemediği gerçek bu!"],
    ["Bu maçta hakem kararları çok tartışmalıydı, taraftarlar tepkiliydi."],
]


# =========================================================
# TAB 2 — GÖRSEL MODERASYON İŞLEV KATMANI
# =========================================================

def analyze_image(image: Image.Image):
    if image is None:
        return "⚪ Lütfen bir görsel yükleyin.", 0, "-", {}

    result = visual_engine.analyze(image)

    verdict = (
        f"### {result.risk_label}\n"
        f"**Manipülasyon Skoru:** `{result.manipulation_score * 100:.1f} / 100`\n"
        f"**Model Güven Skoru:** `{result.confidence * 100:.1f} / 100`"
    )

    detail_df = pd.DataFrame({
        "Etiket": list(result.detail_scores.keys()) or ["skor yok"],
        "Olasılık": [round(v, 4) for v in result.detail_scores.values()] or [0],
    })

    return (
        verdict,
        round(result.manipulation_score * 100, 1),
        "MANİPÜLE EDİLMİŞ OLABİLİR ⚠️" if result.is_manipulated else "OTANTİK GÖRÜNÜYOR ✅",
        detail_df,
    )


# =========================================================
# TAB 3 — BOT & AKIŞ ANALİZİ İŞLEV KATMANI
# =========================================================

def analyze_bot(*values):
    """
    values, bot_engine.feature_names ile aynı sırada gelir (Gradio slider
    çıktıları). Hangi model aktifse (gerçek eğitilmiş / sentetik fallback)
    o modun beklediği özellik isimleriyle eşleştirilir.
    """
    feature_dict = dict(zip(bot_engine.feature_names, values))
    result = bot_engine.analyze(**feature_dict)

    verdict = (
        f"### {result.risk_label}\n"
        f"**Bot Olasılığı:** `{result.bot_probability * 100:.1f} / 100`"
    )

    importance_df = pd.DataFrame({
        "Özellik": list(result.feature_importance.keys()),
        "Önem Skoru": [round(v, 4) for v in result.feature_importance.values()],
    })

    return verdict, round(result.bot_probability * 100, 1), importance_df


# Hem gerçek-eğitilmiş hem sentetik-fallback şeması için hazır demo senaryoları.
# Anahtarlar özellik ismi -> değer; aktif olmayan mod için fazladan anahtarlar
# apply_preset() içinde otomatik filtrelenir.
BOT_PRESET_VALUES = {
    "🤖 Tipik Bot Hesabı": {
        # Gerçek model şeması
        "favourites_count": 50, "followers_count": 20, "friends_count": 4800,
        "statuses_count": 45000, "average_tweets_per_day": 65.0, "account_age_days": 90,
        "verified": 0, "geo_enabled": 0, "default_profile": 1, "default_profile_image": 1,
        "description_length": 0, "has_location": 0, "screen_name_digit_ratio": 0.6,
        # Sentetik fallback şeması
        "posts_per_day": 90, "avg_time_between_posts_sec": 8, "follower_following_ratio": 0.03,
        "account_age_days_synth": 5, "content_diversity_score": 0.05, "night_activity_ratio": 0.8,
        "duplicate_content_ratio": 0.9, "profile_completeness_score": 0.05,
    },
    "🙂 Tipik Gerçek Kullanıcı": {
        "favourites_count": 3200, "followers_count": 850, "friends_count": 420,
        "statuses_count": 2100, "average_tweets_per_day": 1.8, "account_age_days": 1600,
        "verified": 0, "geo_enabled": 1, "default_profile": 0, "default_profile_image": 0,
        "description_length": 85, "has_location": 1, "screen_name_digit_ratio": 0.0,
        "posts_per_day": 3, "avg_time_between_posts_sec": 9000, "follower_following_ratio": 1.4,
        "account_age_days_synth": 850, "content_diversity_score": 0.75, "night_activity_ratio": 0.15,
        "duplicate_content_ratio": 0.05, "profile_completeness_score": 0.85,
    },
    "❓ Şüpheli / Belirsiz Hesap": {
        "favourites_count": 400, "followers_count": 150, "friends_count": 1200,
        "statuses_count": 8000, "average_tweets_per_day": 12.0, "account_age_days": 300,
        "verified": 0, "geo_enabled": 0, "default_profile": 0, "default_profile_image": 0,
        "description_length": 20, "has_location": 0, "screen_name_digit_ratio": 0.3,
        "posts_per_day": 25, "avg_time_between_posts_sec": 300, "follower_following_ratio": 0.4,
        "account_age_days_synth": 60, "content_diversity_score": 0.35, "night_activity_ratio": 0.45,
        "duplicate_content_ratio": 0.4, "profile_completeness_score": 0.4,
    },
}


def apply_preset(preset_name):
    preset = BOT_PRESET_VALUES.get(preset_name, BOT_PRESET_VALUES["❓ Şüpheli / Belirsiz Hesap"])
    values = []
    for spec in bot_engine.feature_spec:
        name = spec["name"]
        # account_age_days iki şemada da var ama farklı ölçekte demo değeri istiyoruz;
        # sentetik moddayken özel anahtarı kullan.
        if name == "account_age_days" and not bot_engine.using_trained_model:
            values.append(preset.get("account_age_days_synth", spec["default"]))
        else:
            values.append(preset.get(name, spec["default"]))
    return values


def load_random_real_account():
    """
    Jüri demosu için: eğitimde hiç kullanılmamış (test seti) GERÇEK bir
    Twitter/X hesabının verisini rastgele seçip slider'lara doldurur, gerçek
    etiketini (bot/human) ayrı bir kutuda gösterir. Yalnızca sayısal özellikler
    kullanılır; kullanıcı adı/biyografi gibi kimlik bilgisi hiç yüklenmez.
    """
    if not bot_engine.using_trained_model or not bot_engine.sample_accounts:
        return [gr.update() for _ in bot_engine.feature_spec] + [
            "⚠️ Gerçek test hesabı havuzu bulunamadı (model sentetik modda veya "
            "sample_test_accounts.json eksik)."
        ]

    account = bot_engine.get_random_sample_account()
    features = account["features"]
    true_label = account["true_label"]

    values = [features.get(spec["name"], spec["default"]) for spec in bot_engine.feature_spec]
    label_text = (
        f"📡 **Gerçek test hesabı yüklendi.** Bu hesabın gerçek etiketi: "
        f"{'🤖 **BOT**' if true_label == 'bot' else '🙂 **HUMAN (gerçek kullanıcı)**'} "
        f"— modelin tahminini görmek için 'Hesabı Analiz Et'e bas."
    )
    return values + [label_text]



# =========================================================
# GRADIO ARAYÜZÜ (Jüri Demo Teması)
# =========================================================

CUSTOM_CSS = """
#header_banner {text-align: center; padding: 10px 0 20px 0;}
#header_banner h1 {font-size: 2.1em; margin-bottom: 0;}
#header_banner p {color: #6b7280; font-size: 1.05em; margin-top: 4px;}
.verdict_box {border-radius: 12px; padding: 14px !important;}
footer {display: none !important;}
"""

with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="rose"),
    css=CUSTOM_CSS,
    title="GuardAI | TEKNOFEST 2026",
) as demo:

    gr.HTML(
        """
        <div id="header_banner">
            <h1>🛡️ GuardAI</h1>
            <p>Hibrit Sosyal Medya Güvenlik ve Akıllı İçerik Moderasyon Motoru</p>
            <p style="font-size:0.85em; color:#9ca3af;">TEKNOFEST 2026 · Sosyal İnovasyon Yarışması</p>
        </div>
        """
    )

    with gr.Tabs():

        # -----------------------------------------------
        # TAB 1: METİN MODERASYONU
        # -----------------------------------------------
        with gr.Tab("📝 Metin Moderasyonu"):
            gr.Markdown(
                "Girilen metni; **toksisite, nefret söylemi, siber zorbalık riski** "
                "ve **dezenformasyon işaretleri** açısından RoBERTa/BERT tabanlı NLP modeli ile analiz eder."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    text_input = gr.Textbox(
                        label="Analiz Edilecek Metin",
                        placeholder="Sosyal medya paylaşımını veya yorumu buraya yapıştırın...",
                        lines=6,
                    )
                    gr.Examples(examples=TEXT_EXAMPLES, inputs=[text_input], label="Örnek Metinler")
                    text_btn = gr.Button("🔍 Metni Analiz Et", variant="primary")

                with gr.Column(scale=1):
                    text_verdict = gr.Markdown(elem_classes="verdict_box")
                    with gr.Row():
                        toxicity_gauge = gr.Number(label="Toksisite Skoru (%)")
                        hate_gauge = gr.Number(label="Nefret Söylemi Skoru (%)")
                    with gr.Row():
                        bully_gauge = gr.Number(label="Siber Zorbalık Riski (%)")
                        disinfo_flag = gr.Textbox(label="Dezenformasyon İşareti")
                    text_detail_table = gr.Dataframe(label="Model Detay Skorları", wrap=True)

            text_btn.click(
                fn=analyze_text,
                inputs=[text_input],
                outputs=[text_verdict, toxicity_gauge, hate_gauge, bully_gauge, disinfo_flag, text_detail_table],
            )

        # -----------------------------------------------
        # TAB 2: GÖRSEL MODERASYON
        # -----------------------------------------------
        with gr.Tab("🖼️ Görsel Moderasyon"):
            gr.Markdown(
                "Yüklenen görseli; **deepfake, sahte içerik ve görsel manipülasyon** açısından "
                "Vision Transformer (ViT) tabanlı model ile inceler."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    image_input = gr.Image(label="Görsel Yükle", type="pil", height=320)
                    image_btn = gr.Button("🔍 Görseli Analiz Et", variant="primary")

                with gr.Column(scale=1):
                    image_verdict = gr.Markdown(elem_classes="verdict_box")
                    manipulation_gauge = gr.Number(label="Manipülasyon Skoru (%)")
                    manipulation_flag = gr.Textbox(label="Sonuç")
                    image_detail_table = gr.Dataframe(label="Model Etiket Olasılıkları", wrap=True)

            image_btn.click(
                fn=analyze_image,
                inputs=[image_input],
                outputs=[image_verdict, manipulation_gauge, manipulation_flag, image_detail_table],
            )

        # -----------------------------------------------
        # TAB 3: BOT & AKIŞ ANALİZİ
        # -----------------------------------------------
        with gr.Tab("🤖 Bot & Akış Analizi"):
            model_source_note = (
                "**Gerçek Twitter/X hesap verisiyle eğitilmiş model kullanılıyor** "
                "(airt-ml/twitter-human-bots, 37.438 hesap)."
                if bot_engine.using_trained_model else
                "⚠️ Eğitilmiş yerel model bulunamadı, **sentetik demo verisiyle** eğitilen "
                "fallback model kullanılıyor. Gerçek veriyle eğitmek için `train_bot_model.py` çalıştırın."
            )
            gr.Markdown(
                "Hesabın **etkileşim frekansı ve davranışsal ağ verilerine** göre "
                f"RandomForest tabanlı sınıflandırıcı ile bot/sahte hesap olasılığını hesaplar. {model_source_note}"
            )
            with gr.Row():
                with gr.Column(scale=1):
                    preset_dropdown = gr.Dropdown(
                        choices=list(BOT_PRESET_VALUES.keys()),
                        label="Hızlı Demo Senaryosu Seç (Jüri için)",
                        value="❓ Şüpheli / Belirsiz Hesap",
                    )
                    real_account_btn = gr.Button(
                        "🎲 Gerçek Test Hesabından Rastgele Yükle"
                        + ("" if bot_engine.using_trained_model else " (yalnızca gerçek model modunda)"),
                        variant="secondary",
                    )
                    real_account_label = gr.Markdown()
                    # Slider'lar aktif modele göre (gerçek/sentetik) DİNAMİK oluşturulur
                    bot_sliders = []
                    for spec in bot_engine.feature_spec:
                        slider = gr.Slider(
                            minimum=spec["min"],
                            maximum=spec["max"],
                            value=spec["default"],
                            step=spec["step"],
                            label=spec["label"],
                        )
                        bot_sliders.append(slider)
                    bot_btn = gr.Button("🔍 Hesabı Analiz Et", variant="primary")

                with gr.Column(scale=1):
                    bot_verdict = gr.Markdown(elem_classes="verdict_box")
                    bot_prob_gauge = gr.Number(label="Bot Olasılığı (%)")
                    bot_importance_table = gr.Dataframe(label="Özellik Önem Sıralaması (RandomForest)", wrap=True)

            real_account_btn.click(
                fn=load_random_real_account,
                inputs=[],
                outputs=bot_sliders + [real_account_label],
            )

            preset_dropdown.change(
                fn=apply_preset,
                inputs=[preset_dropdown],
                outputs=bot_sliders,
            )

            bot_btn.click(
                fn=analyze_bot,
                inputs=bot_sliders,
                outputs=[bot_verdict, bot_prob_gauge, bot_importance_table],
            )


    gr.HTML(
        """
        <div style="text-align:center; margin-top:20px; color:#9ca3af; font-size:0.85em;">
            GuardAI &copy; 2026 · Hibrit NLP + Computer Vision + Davranışsal ML Mimarisi ·
            FastAPI backend entegrasyonuna hazır modüler yapı
        </div>
        """
    )


# =========================================================
# UYGULAMAYI BAŞLAT
# =========================================================
if __name__ == "__main__":
    demo.queue()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )