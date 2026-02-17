from flask import Flask, render_template, request, jsonify
import librosa
import numpy as np
import joblib
import os

# Flask app (templates & static auto-detect)
app = Flask(__name__)

# Ensure uploads folder exists (Render lo important)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load trained model & scaler
model = joblib.load("final_emotion_model.pkl")
scaler = joblib.load("final_scaler.pkl")

def extract_features(path):
    data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)

    mfcc = np.mean(
        librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=40).T, axis=0
    )
    mel = np.mean(
        librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0
    )
    contrast = np.mean(
        librosa.feature.spectral_contrast(y=data, sr=sample_rate).T, axis=0
    )
    tonnetz = np.mean(
        librosa.feature.tonnetz(
            y=librosa.effects.harmonic(data), sr=sample_rate
        ).T, axis=0
    )

    return np.hstack([mfcc, mel, contrast, tonnetz])

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict_offline", methods=["POST"])
def predict_offline():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    feat = extract_features(path)
    feat = scaler.transform([feat])
    prediction = model.predict(feat)[0]

    return jsonify({"emotion": f"Detected: {prediction}"})

@app.route("/predict_online", methods=["POST"])
def predict_online():
    if "audio" not in request.files:
        return jsonify({"error": "No audio received"}), 400

    audio_data = request.files["audio"]
    path = os.path.join(UPLOAD_FOLDER, "live_audio.wav")
    audio_data.save(path)

    feat = extract_features(path)
    feat = scaler.transform([feat])
    prediction = model.predict(feat)[0]

    return jsonify({"emotion": f"Real Emotion: {prediction}"})

# 🔥 RENDER + LOCAL SUPPORT (MOST IMPORTANT PART)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
