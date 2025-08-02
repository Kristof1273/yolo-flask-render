from ultralytics import YOLO
from flask import Flask, request, jsonify
import cv2
import numpy as np


model = YOLO("model.pt")

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    img_bytes = file.read()
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    results = model(img)
    return jsonify(results[0].tojson())
