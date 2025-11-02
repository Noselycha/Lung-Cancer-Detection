import os
import io
import cv2
import base64
import datetime
import markdown
import numpy as np
import tensorflow as tf
import google.generativeai as genai
from flask import Flask, render_template, request, redirect, url_for
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input
from pymongo import MongoClient
from bson import ObjectId
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
load_dotenv()

# Setup MongoDB
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client.lungcancer
history_collection = db.history

try:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
except Exception as e:
    print(f"Error: {e}")

LABELS = ["adenocarcinoma", "large.cell.carcinoma", "normal", "squamous.cell.carcinoma"]
IMAGE_SIZE = 224

app = Flask(__name__)

def get_gemini_explanation(prompt, img_bytes=None):
    """Menghasilkan penjelasan dari Gemini, mendukung input multimodal (teks+gambar)."""
    try:
        model = genai.GenerativeModel("gemini-2.5-pro") 
        if img_bytes is not None:
            image_data = img_bytes.tobytes()
            image_part = {"mime_type": "image/jpeg", "data": image_data}
            response = model.generate_content([prompt, image_part])
        else:
            response = model.generate_content(prompt)
            
        return response.text
    except Exception as e:
        print(f"Error saat memanggil Gemini: {str(e)}")
        return f"Gagal menghasilkan penjelasan: {str(e)}"

def validate_image_with_gemini(img_bytes):
    """Memvalidasi apakah gambar adalah CT scan dada menggunakan Gemini."""
    try:
        prompt = (
            "Anda adalah validator untuk aplikasi medis. "
            "Tugas Anda adalah memeriksa apakah gambar berikut adalah citra CT scan dada manusia. "
            "Jika gambar adalah CT scan dada (dengan atau tanpa kelainan), jawab 'VALID'. "
            "Jika bukan (misal foto wajah, hewan, MRI, X-ray, atau objek lain), jawab 'INVALID'. "
            "Jawaban hanya satu kata: VALID atau INVALID."
        )
        model = genai.GenerativeModel("gemini-2.5-pro")
        image_part = {"mime_type": "image/jpeg", "data": img_bytes}
        response = model.generate_content([prompt, image_part])
        return response.text.strip().upper() == "VALID"
    except Exception as e:
        print(f"Error saat validasi Gemini: {str(e)}")
        return True 

def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("Tidak ada layer Conv2D yang ditemukan di model.")

def get_gradcam_heatmap(model, img_array, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
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
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_file = None
    
    if model_name == "efficientnet":
        model_file = "lungcancer_efficientnetB3_RMSprop.h5"
    else:
        raise ValueError("Model tidak dikenal atau tidak didukung. Hanya 'efficientnet' yang tersedia.")

    model_path = os.path.join(base_dir, "model", model_file)
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}. Letakkan file .h5 di dalam folder 'model'."
        )

    class CastLayer(tf.keras.layers.Layer):
        def __init__(self, dtype='float32', **kwargs):
            self._dtype = dtype
            super().__init__(**kwargs)

        @property
        def dtype(self):
            return self._dtype

        def call(self, inputs):
            return tf.cast(inputs, dtype=self.dtype)

        def get_config(self):
            config = super().get_config()
            config.update({"dtype": self._dtype})
            return config

        @classmethod
        def from_config(cls, config):
            return cls(**config)

    custom_objects = {
        'Cast': CastLayer
    }
    
    try:
        with tf.keras.utils.custom_object_scope(custom_objects):
            model = tf.keras.models.load_model(model_path, compile=False)
    except Exception as e:
        raise ValueError(f"Error loading model: {str(e)}. Pastikan file model tidak korup.")

    last_conv = find_last_conv_layer(model)
    return model, last_conv

LOADED_MODELS = {}
LOADED_LAST_CONV = {}

def load_all_models():
    available_models = {
        "efficientnet": "lungcancer_efficientnetB3_RMSprop.h5"
    }
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, "model")
    
    print("Mengecek model di:", model_dir)
    if not os.path.exists(model_dir):
        print("Folder 'model' tidak ditemukan! Membuat folder...")
        os.makedirs(model_dir)
        
    print("File yang tersedia:", os.listdir(model_dir))
    
    for model_name, model_file in available_models.items():
        try:
            print(f"Memuat model {model_name}...")
            model, last_conv = load_selected_model(model_name)
            LOADED_MODELS[model_name] = model
            LOADED_LAST_CONV[model_name] = last_conv
            
            try:
                dummy = np.zeros((1, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)
                _ = model.predict(dummy)
            except Exception as warm_err:
                print(f"Peringatan: Pemanasan model gagal untuk {model_name}: {warm_err}")
                
            print(f"Berhasil memuat {model_name}")
        except Exception as e:
            print(f"Error memuat {model_name}: {str(e)}")
            
    if not LOADED_MODELS:
        print("PERINGATAN: Tidak ada model yang berhasil dimuat!")
    else:
        print(f"Berhasil memuat {len(LOADED_MODELS)} model: {list(LOADED_MODELS.keys())}")

load_all_models()

def process_image(img_bytes, selected_model):
    """Memproses gambar: prediksi, dan buat heatmap."""
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_resized = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img_array = np.expand_dims(img_resized, axis=0)
    img_array_preprocessed = preprocess_input(img_array.astype(np.float32))
    
    model = LOADED_MODELS.get(selected_model)
    if not model:
        raise ValueError(f"Model {selected_model} tidak dimuat")
    
    last_conv = LOADED_LAST_CONV.get(selected_model)
    
    preds = model.predict(img_array_preprocessed)[0]
    
    pred_index = np.argmax(preds)
    pred_class = LABELS[pred_index]
    
    heatmap = get_gradcam_heatmap(
        model, img_array_preprocessed, last_conv, pred_index=pred_index
    )
    
    return img, pred_class, pred_index, heatmap

@app.route("/")
def dashboard():
    return render_template("dashboard.html")

@app.route("/classify", methods=["GET", "POST"])
def classify():
    prediction = None
    explanation_html = None
    error_message = None
    input_base64 = None
    gradcam_base64 = None
    overlay_base64 = None

    if request.method == "POST":
        try:
            selected_model = "efficientnet" 
            nama_pasien = request.form["nama_pasien"]
            umur_pasien = request.form["umur_pasien"]
            gender_pasien = request.form["gender_pasien"]
            
            if "image" not in request.files or not request.files["image"].filename:
                raise ValueError("Tidak ada file gambar yang diupload.")

            file = request.files["image"]
            img_bytes = file.read() 
            input_base64 = base64.b64encode(img_bytes).decode("utf-8")
            
            if not validate_image_with_gemini(img_bytes):
                error_message = "Validasi Gagal: Gambar yang diupload bukan CT scan dada manusia."
                return render_template("classify.html", error_message=error_message)
            
            try:
                img, prediction, pred_index, heatmap = process_image(img_bytes, selected_model)
            except Exception as e:
                print(f"Error selama process_image: {e}")
                raise ValueError(f"Gagal memproses gambar dengan model: {str(e)}")

            with ThreadPoolExecutor() as executor:
                heatmap = np.nan_to_num(heatmap).astype(np.float32)
                if np.max(heatmap) > 0:
                    heatmap /= np.max(heatmap)
                
                heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
                heatmap_uint8 = np.uint8(255 * heatmap_resized)
                heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
                
                superimposed_img = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)
                
                future_gradcam = executor.submit(cv2.imencode, '.jpg', superimposed_img)
                future_overlay = executor.submit(cv2.imencode, '.jpg', heatmap_color)

                success_gradcam, gradcam_buf_array = future_gradcam.result()
                if not success_gradcam:
                    raise ValueError("Gagal meng-encode gambar Grad-CAM")
                prompt = (
                         f"Anda adalah seorang ahli radiologi AI. Gambar CT scan dada telah diklasifikasikan sebagai '{prediction}'. "
                         "Saya juga melampirkan gambar Grad-CAM (visualisasi overlay) yang menunjukkan area fokus model (area panas berwarna merah/kuning). "
                         "Berdasarkan klasifikasi DAN gambar Grad-CAM yang terlampir, berikan analisis mendalam. "
                         "Jelaskan secara spesifik di mana letak area panas pada gambar (misalnya: paru-paru kanan atas, dekat bronkus, dll.) dan apa signifikansinya terkait dengan klasifikasi. "
                         "Berikan penjelasan dalam format Markdown yang terstruktur dengan heading, poin-poin, dan teks tebal untuk istilah penting.\n\n"
                         "Struktur Jawaban:\n"
                         "### Analisis Hasil Klasifikasi\n"
                         f"- **Prediksi Model:** {prediction}\n"
                         f"- **Penjelasan Singkat:** (Jelaskan secara singkat apa itu {prediction})\n\n"
                         "### Interpretasi Visualisasi Grad-CAM\n"
                         "- **Lokasi Fokus:** (Deskripsikan di mana area merah/kuning paling intens berada pada gambar paru-paru berdasarkan gambar Grad-CAM terlampir).\n"
                         "- **Signifikansi:** (Jelaskan mengapa model mungkin fokus pada area tersebut dan hubungannya dengan {prediction}).\n\n"
                         "### Potensi Implikasi & Langkah Selanjutnya\n"
                         "- **Implikasi Klinis:** (Jelaskan apa arti temuan ini secara umum).\n"
                         "- **Rekomendasi:** (Sebutkan bahwa hasil ini bukan diagnosis dan perlu divalidasi oleh ahli radiologi dan melalui tes lebih lanjut).\n\n"
                         "**Disclaimer:** Analisis ini dihasilkan oleh AI dan bukan merupakan diagnosis medis. Konsultasikan dengan profesional kesehatan untuk evaluasi lebih lanjut."
                )

                future_explanation = executor.submit(get_gemini_explanation, prompt, img_bytes=gradcam_buf_array)
                

                _, overlay_buf_array = future_overlay.result()
                classification_info = future_explanation.result()
                
                gradcam_base64 = base64.b64encode(gradcam_buf_array).decode("utf-8")
                overlay_base64 = base64.b64encode(overlay_buf_array).decode("utf-8")
            
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
                "timestamp": datetime.datetime.utcnow()
            }
            history_collection.insert_one(history_doc)

        except Exception as e:
            error_message = f"Terjadi kesalahan: {str(e)}"
            print(f"Error di /classify: {e}") 
            
    return render_template(
        "classify.html",
        prediction=prediction,
        input_base64=input_base64,
        gradcam_base64=gradcam_base64,
        overlay_base64=overlay_base64,
        explanation=explanation_html, 
        error_message=error_message,
        available_models=list(LOADED_MODELS.keys())
    )

@app.route("/performance")
def performance():
    models = ["efficientnet"] 

    confusion_matrix_imgs = {
        "efficientnet": "EfficientNet_cm.jpg",
    }
    loss_imgs = {
        "efficientnet": "EfficientNet_curve.jpg",
    }
    metrics = {
        "efficientnet": {
            "accuracy": 0.93,
            "precision": [0.92, 0.94, 0.95, 0.92],
            "recall": [0.94, 0.92, 0.93, 0.95],
            "f1_score": [0.93, 0.93, 0.94, 0.93],
        }
    }

    return render_template(
        "performance.html",
        models=models,
        confusion_matrix_imgs=confusion_matrix_imgs,
        loss_imgs=loss_imgs,
        metrics=metrics,
        labels=LABELS,
    )

@app.route("/about")
def about():
    research_team = [
        {
            "name": "Noselycha Soriton",
            "role": "Mahasiswa Peneliti",
            "university": "Universitas Klabat",
            "faculty": "Fakultas Ilmu Komputer",
            "department": "Informatika",
            "year": "2022",
            "email": "noselycha@unklab.ac.id",
            "photo": "assets/lika.jpg",
        },
        {
            "name": "Emily Pangemanan",
            "role": "Mahasiswa Peneliti",
            "university": "Universitas Klabat",
            "faculty": "Fakultas Ilmu Komputer",
            "department": "Informatika",
            "year": "2022",
            "email": "emily@unklab.ac.id",
            "photo": "assets/mili.jpg",
        },
    ]
    advisors = [
        {
            "name": "Green Sandag",
            "role": "Dosen Pembimbing 1",
            "university": "Universitas Klabat",
            "faculty": "Fakultas Ilmu Komputer",
            "email": "gsandag@unklab.ac.id",
            "photo": "assets/sirgreen.jpg",
        },
        {
            "name": "Raissa Maringka",
            "role": "Dosen Pembimbing 2",
            "university": "Universitas Klabat",
            "faculty": "Fakultas Ilmu Komputer",
            "email": "rmaringka@unklab.ac.id",
            "photo": "assets/raissa.jpg",
        },
    ]
    return render_template("about.html", research_team=research_team, advisors=advisors)

@app.route("/history")
def history():
    feedback = request.args.get("feedback")
    try:
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
            
    except Exception as e:
        print(f"Error mengambil riwayat: {e}")
        histories = []
        feedback = "Gagal mengambil data riwayat."
        
    return render_template("history.html", histories=histories, feedback=feedback)

@app.route("/delete_history/<history_id>", methods=["POST"])
def delete_history(history_id):
    try:
        history_collection.delete_one({"_id": ObjectId(history_id)})
        return redirect(url_for("history", feedback="Riwayat berhasil dihapus."))
    except Exception as e:
        print(f"Error menghapus riwayat: {e}")
        return redirect(url_for("history", feedback="Gagal menghapus riwayat."))

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=5000)