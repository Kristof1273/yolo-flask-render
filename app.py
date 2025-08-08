from flask import Flask, request, jsonify
from ultralytics import YOLO
import numpy as np
import cv2
import os

app = Flask(__name__)

try:
    model = YOLO("model.pt")
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Model loading failed: {e}")


@app.route("/")
def home():
    return """
    <html>
        <head>
            <title>YOLO Flask App</title>
        </head>
        <body style="text-align:center; font-family:sans-serif; margin-top:50px;">
            <h1>✅ YOLO Flask app is running!</h1>
            <p>Az API működik, készen áll képek fogadására.</p>
        </body>
    </html>
    """

@app.route("/predict", methods=["POST"])
def predict():
    try:
        print("🔍 Request received")

        if "image" not in request.files:
            print("⚠️ No image found in request")
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files["image"]
        print(f"📦 File received: {file.filename}")

        img_bytes = file.read()
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

        if img is None:
            print("❌ Failed to decode image")
            return jsonify({"error": "Invalid image"}), 400

        print("✅ Image decoded, running model...")
        results = model(img)

        print("📊 Model inference done")
        json_result = results[0].tojson()
        print(f"📄 JSON result: {json_result[:100]}...")  # csak az első 100 karakter

        return json_result, 200, {"Content-Type": "application/json"}

    except Exception as e:
        print(f"🔥 Exception occurred: {e}")
        return jsonify({"error": "Internal server error"}), 500


    if img is None:
        return jsonify({"error": "Invalid image"}), 400

    results = model(img)
    return results[0].tojson(), 200, {"Content-Type": "application/json"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
