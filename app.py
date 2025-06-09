import os
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import pytesseract
from PIL import Image
import re

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Flask App Setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load YOLOv8 Model
model = YOLO('yolov8n.pt')  # Or your custom-trained license plate model

# Utility Function

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def detect_license_plate(image_path):
    image = cv2.imread(image_path)
    results = model(image)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cropped = image[y1:y2, x1:x2]
        temp_path = os.path.join(UPLOAD_FOLDER, 'temp.jpg')
        cv2.imwrite(temp_path, cropped)

        text = pytesseract.image_to_string(Image.open(temp_path))
        plate_text = re.sub(r'[^A-Z0-9]', '', text.upper())

        if validate_plate(plate_text):
            return plate_text, True
        else:
            return plate_text, False

    return "No plate detected", False

def validate_plate(plate_text):
    # Example: MH12AB1234, DL8CAF5030
    pattern = r'^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$'
    return bool(re.match(pattern, plate_text))

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        plate, valid = detect_license_plate(filepath)
        return jsonify({
            'plate': plate,
            'valid': valid
        })
    else:
        return jsonify({'error': 'File type not allowed'}), 400

if __name__ == '__main__':
    app.run(debug=True)
