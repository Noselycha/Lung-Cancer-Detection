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
import io
from bson import ObjectId
from flask import redirect, url_for

matplotlib.use("Agg")
from dotenv import load_dotenv

load_dotenv()

from pymongo import MongoClient
import datetime

MONGO_URI = os.getenv("MONGO_URI")  
client = MongoClient(MONGO_URI)
db = client.lungcancer
history_collection = db.history

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

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
        model_path = "model/chest_efficientnetB1_RMSprop.keras"
    elif model_name == "resnet":
        model_path = "model/chest_resnet50_RMSprop.keras"
    elif model_name == "vgg":
        model_path = "model/chest_VGG16RMSprop.keras"
    elif model_name == "densenet":
        model_path = "model/chest_densenet121.keras"
    else:
        raise ValueError("Model tidak dikenal")
    model = load_model(model_path)
    last_conv = find_last_conv_layer(model)
    return model, last_conv

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
    input_base64 = None
    gradcam_base64 = None
    overlay_base64 = None

    if request.method == "POST":
        selected_model = request.form["model"]
        nama_pasien = request.form["nama_pasien"]
        umur_pasien = request.form["umur_pasien"]
        gender_pasien = request.form["gender_pasien"]
        file = request.files["image"]
        img_bytes = file.read()
        input_base64 = base64.b64encode(img_bytes).decode("utf-8")

        # Validasi Gemini langsung dari img_bytes (tanpa simpan ke disk)
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
        if response.text.strip().upper() != "VALID":
            error_message = "Gambar yang diupload bukan CT scan dada manusia."
            return render_template("classify.html", error_message=error_message)

        # Validasi langsung dari img_bytes (tanpa save ke disk)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
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

        # GradCAM base64 langsung dari array
        superimposed_img = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)
        _, gradcam_buf = cv2.imencode('.jpg', superimposed_img)
        gradcam_base64 = base64.b64encode(gradcam_buf).decode("utf-8")

        # Overlay base64 langsung dari array
        _, overlay_buf = cv2.imencode('.jpg', heatmap_color)
        overlay_base64 = base64.b64encode(overlay_buf).decode("utf-8")

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


        history_doc = {
            "model": selected_model,
            "prediction": prediction,
            "input_filename": file.filename,
            "image_base64": input_base64,
            "gradcam_base64": gradcam_base64,
            "nama_pasien": nama_pasien, 
        }
        history_collection.insert_one(history_doc)

    return render_template(
        "classify.html",
        prediction=prediction,
        input_base64=input_base64,   
        gradcam_base64=gradcam_base64, 
        overlay_base64=overlay_base64,  
        explanation=explanation_html,
        error_message=error_message,
    )


@app.route("/performance")
def performance():
    # Nama model yang tersedia
    models = ["efficientnet", "densenet", "resnet", "vgg"]

    # Gambar confusion matrix dan loss curve untuk setiap model
    confusion_matrix_imgs = {
        "efficientnet": "EfficientNet_cm.jpg",
        "densenet": "DenseNet_cm.jpg",
        "resnet": "ResNet_cm.jpg",
        "vgg": "VGG_cm.jpg",
    }
    loss_imgs = {
        "efficientnet": "EfficientNet_curve.jpg",
        "densenet": "DenseNet_curve.jpg",
        "resnet": "ResNet_curve.jpg",
        "vgg": "VGG_curve.jpg",
    }

    # Contoh data evaluasi (isi dengan data asli Anda)
    metrics = {
        "efficientnet": {
            "accuracy": 0.93,
            "precision": [0.92, 0.94, 0.95, 0.92],
            "recall": [0.94, 0.92, 0.93, 0.95],
            "f1_score": [0.93, 0.93, 0.94, 0.93],
        },
        "densenet": {
            "accuracy": 0.91,
            "precision": [0.90, 0.92, 0.93, 0.91],
            "recall": [0.92, 0.90, 0.91, 0.93],
            "f1_score": [0.91, 0.91, 0.92, 0.92],
        },
        "resnet": {
            "accuracy": 0.89,
            "precision": [0.88, 0.90, 0.91, 0.89],
            "recall": [0.90, 0.88, 0.89, 0.91],
            "f1_score": [0.89, 0.89, 0.90, 0.90],
        },
        "vgg": {
            "accuracy": 0.87,
            "precision": [0.86, 0.88, 0.89, 0.87],
            "recall": [0.88, 0.86, 0.87, 0.89],
            "f1_score": [0.87, 0.87, 0.88, 0.88],
        },
    }

    labels = LABELS

    return render_template(
        "performance.html",
        models=models,
        confusion_matrix_imgs=confusion_matrix_imgs,
        loss_imgs=loss_imgs,
        metrics=metrics,
        labels=labels,
    )

@app.route("/history")
def history():
    feedback = request.args.get("feedback")
    histories = list(history_collection.find({}, {
        "nama_pasien": 1,
        "timestamp": 1,
        "model": 1,
        "prediction": 1,
        "input_filename": 1,
        "image_base64": 1,
        "gradcam_base64": 1
    }).sort("timestamp", -1))
    for h in histories:
        h["_id"] = str(h["_id"])
    return render_template("history.html", histories=histories, feedback=feedback)

@app.route("/delete_history/<history_id>", methods=["POST"])
def delete_history(history_id):
    history_collection.delete_one({"_id": ObjectId(history_id)})
    return redirect(url_for("history", feedback="Riwayat berhasil dihapus."))

if __name__ == "__main__":
    app.run(debug=True)
