# app.py
import os
import io
import time
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
import numpy as np
import torch
import easyocr
from ultralytics import YOLO

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)  # Allow frontend to call API

# Load YOLOv8 model
yolo_model = YOLO('best.pt')  # Replace with your custom-trained model
# Load EasyOCR reader
reader = easyocr.Reader(['en'])

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def detect_license_plate(img_bytes):
    results = yolo_model(img_bytes)[0]  # Run YOLOv8 detection
    plates = []

    for det in results.boxes:
        x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
        conf = float(det.conf[0])
        cropped = img.crop((x1, y1, x2, y2))
        text_result = reader.readtext(np.array(cropped), detail=0)
        plate_text = text_result[0] if text_result else ""
        valid = bool(len(plate_text) >= 6)
        plates.append({
            'plate': plate_text,
            'confidence': round(conf * 100, 1),
            'valid': valid,
            'box': [x1, y1, x2, y2]
        })
    return plates

@app.route('/api/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded'}), 400
    file = request.files['file']
    img = Image.open(file.stream).convert('RGB')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes = img_bytes.getvalue()

    try:
        plates = detect_license_plate(img_bytes)
        return jsonify({'success': True, 'plates': plates})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/chatbot', methods=['POST'])
def chatbot():
    data = request.get_json()
    msg = data.get('message', '').lower()

    if 'validate' in msg:
        return jsonify({'reply': 'Indian plates follow pattern: STATE‑RTO‑XXXX format.'})
    elif 'help' in msg:
        return jsonify({'reply': 'Send images or camera frames to detect license plates.'})
    else:
        return jsonify({'reply': "I'm here to help—ask me about license plate detection!"})

@app.route('/', defaults={'path': 'index.html'})
@app.route('/<path:path>')
def serve(path):
    return send_from_directory('.', path)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
