from flask import Flask, render_template, request
import os
import numpy as np
import google.generativeai as genai
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Conv2D
import cv2
import markdown
from tensorflow.keras.applications.efficientnet import preprocess_input
import matplotlib
import base64

matplotlib.use("Agg")
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# ==============================================================================
# Utility functions (Fungsi-fungsi ini sudah sesuai dengan skrip Kaggle Anda)
# ==============================================================================


def get_gemini_explanation(prompt):
    try:
        model = genai.GenerativeModel("gemini-2.5-pro")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Gagal menghasilkan penjelasan: {str(e)}"


LABELS = ["adenocarcinoma", "large.cell.carcinoma", "normal", "squamous.cell.carcinoma"]
IMAGE_SIZE = 460


def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, Conv2D):
            return layer.name
    raise ValueError("Tidak dapat menemukan layer Conv2D di dalam model.")

def validate_with_gemini(image_path):
    try:
        with open(image_path, "rb") as f:
            img_bytes = f.read()
        prompt = (
            "Anda adalah validator untuk aplikasi medis. "
            "Tugas Anda adalah memeriksa apakah gambar berikut adalah citra CT scan dada manusia. "
            "Jika gambar adalah CT scan dada (dengan atau tanpa kelainan), jawab 'VALID'. "
            "Jika bukan (misal foto wajah, hewan, MRI, X-ray, atau objek lain), jawab 'INVALID'. "
            "Jawaban hanya satu kata: VALID atau INVALID."
        )
        model = genai.GenerativeModel("gemini-2.5-pro")
        response = model.generate_content(
            [
                prompt,
                {"mime_type": "image/jpeg", "data": img_bytes}
            ]
        )
        return response.text.strip().upper() == "VALID"
    except Exception as e:
        print("Error Gemini Validation:", e)
        return False

def get_gradcam_heatmap(model, img_array, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output],
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    denom = tf.math.reduce_max(heatmap) + tf.keras.backend.epsilon()
    heatmap = heatmap / denom
    return heatmap.numpy()


def load_selected_model(model_name):
    model_path = ""
    if model_name == "efficientnet":
        model_path = "model/chest_efficientnetB1.keras"
    elif model_name == "resnet":
        model_path = "model/resnet.h5"
    elif model_name == "vgg":
        model_path = "model/vgg.h5"
    elif model_name == "densenet":
        model_path = "model/densenet.h5"
    else:
        raise ValueError("Model tidak dikenal")
    model = load_model(model_path)
    last_conv = find_last_conv_layer(model)
    return model, last_conv


# ==============================================================================
# Flask App Routes
# ==============================================================================

app = Flask(__name__)


@app.route("/")
def dashboard():
    return render_template("dashboard.html")


@app.route("/classify", methods=["GET", "POST"])
def classify():
    prediction = None
    gradcam_filename = None
    gradcam_only_filename = None
    selected_model = None
    explanation_html = None
    input_path = None
    error_message = None

    if request.method == "POST":
        selected_model = request.form["model"]
        img_file = request.files["image"]

        img_filename = img_file.filename
        img_path = os.path.join("static", img_filename)
        img_file.save(img_path)
        input_path = img_filename

        # Input Validation
        if not validate_with_gemini(img_path):
            error_message = "Gambar Anda tidak dikenali sebagai citra CT scan dada. Silakan unggah gambar CT scan dada."
            return render_template(
                "classify.html",
                error_message=error_message,
                prediction=None,
                gradcam_path=None,
                gradcam_only_path=None,
                input_path=input_path,
                selected_model=selected_model,
                explanation=None,
            )

        img = cv2.imread(img_path)
        img_resized = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        img_array = np.expand_dims(img_resized, axis=0)
        img_array_preprocessed = preprocess_input(img_array)

        model, last_conv = load_selected_model(selected_model)

        preds = model.predict(img_array_preprocessed)
        pred_index = np.argmax(preds)
        pred_class = LABELS[pred_index]
        prediction = pred_class

        heatmap = get_gradcam_heatmap(
            model, img_array_preprocessed, last_conv, pred_index=pred_index
        )
        heatmap = np.nan_to_num(heatmap).astype(np.float32)
        if np.max(heatmap) > 0:
            heatmap /= np.max(heatmap)
        heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        gradcam_only_filename = f"heatmap_{img_filename}"
        gradcam_only_path = os.path.join("static", gradcam_only_filename)
        cv2.imwrite(gradcam_only_path, heatmap_color)

        gradcam_filename = f"gradcam_{img_filename}"
        gradcam_path = os.path.join("static", gradcam_filename)
        superimposed_img = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)
        cv2.imwrite(gradcam_path, superimposed_img)

        prompt = (
            f"Anda adalah seorang ahli radiologi AI. Gambar CT scan dada telah diklasifikasikan sebagai '{prediction}'. "
            "Visualisasi Grad-CAM menunjukkan area fokus model (area panas berwarna merah/kuning). "
            "Berdasarkan klasifikasi dan area fokus Grad-CAM tersebut, berikan analisis mendalam. "
            "Jelaskan secara spesifik di mana letak area panas pada gambar (misalnya: paru-paru kanan atas, dekat bronkus, dll.) dan apa signifikansinya terkait dengan klasifikasi. "
            "Berikan penjelasan dalam format Markdown yang terstruktur dengan heading, poin-poin, dan teks tebal untuk istilah penting.\n\n"
            "Struktur Jawaban:\n"
            "### Analisis Hasil Klasifikasi\n"
            f"- **Prediksi Model:** {prediction}\n"
            "- **Penjelasan Singkat:** (Jelaskan secara singkat apa itu {prediction})\n\n"
            "### Interpretasi Visualisasi Grad-CAM\n"
            "- **Lokasi Fokus:** (Deskripsikan di mana area merah/kuning paling intens berada pada gambar paru-paru).\n"
            "- **Signifikansi:** (Jelaskan mengapa model mungkin fokus pada area tersebut dan hubungannya dengan {prediction}).\n\n"
            "### Potensi Implikasi & Langkah Selanjutnya\n"
            "- **Implikasi Klinis:** (Jelaskan apa arti temuan ini secara umum).\n"
            "- **Rekomendasi:** (Sebutkan bahwa hasil ini bukan diagnosis dan perlu divalidasi oleh ahli radiologi dan melalui tes lebih lanjut).\n\n"
            "**Disclaimer:** Analisis ini dihasilkan oleh AI dan bukan merupakan diagnosis medis. Konsultasikan dengan profesional kesehatan untuk evaluasi lebih lanjut."
        )
        classification_info = get_gemini_explanation(prompt)

        if classification_info:
            explanation_html = markdown.markdown(
                classification_info, extensions=["fenced_code", "tables"]
            )

    return render_template(
        "classify.html",
        prediction=prediction,
        gradcam_path=gradcam_filename,
        gradcam_only_path=gradcam_only_filename,
        input_path=input_path,
        selected_model=selected_model,
        explanation=explanation_html,
        error_message=error_message,
    )


if __name__ == "__main__":
    app.run(debug=True)
