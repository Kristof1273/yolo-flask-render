from flask import Flask, request, jsonify
from ultralytics import YOLO
import numpy as np
import cv2
import os

app = Flask(__name__)
model = YOLO("model.pt")  # Győződj meg róla, hogy a fájl a gyökérkönyvtárban van

@app.route("/")
def home():
    return "🦾 YOLO Flask app is running!"

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    img_bytes = file.read()
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Invalid image"}), 400

    results = model(img)
    return jsonify(results[0].tojson())

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
