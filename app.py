import os
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template
from ultralytics import YOLO
import pytesseract
from PIL import Image
import re
from flask_cors import CORS

# Set tesseract path (modify based on your OS)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Flask app
app = Flask(__name__)
CORS(app)

# Load YOLOv8 model
model = YOLO("best.pt")  # Put your trained YOLOv8 model path here

# License plate regex pattern (India)
plate_pattern = r'^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$'

# Preprocess image for better OCR
def preprocess_for_ocr(plate_img):
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

# Extract text from license plate
def extract_plate_text(cropped_img):
    processed = preprocess_for_ocr(cropped_img)
    pil_img = Image.fromarray(processed)
    text = pytesseract.image_to_string(pil_img, config='--psm 8')
    cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
    return cleaned

# Validate plate format
def is_valid_plate(plate_text):
    return bool(re.match(plate_pattern, plate_text))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/detect', methods=['POST'])
def detect_plate():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    file = request.files['image']
    img_path = os.path.join("uploads", file.filename)
    file.save(img_path)

    # Load image
    img = cv2.imread(img_path)
    results = model(img)

    detection_data = []

    for box in results[0].boxes.xyxy:
        x1, y1, x2, y2 = map(int, box)
        cropped = img[y1:y2, x1:x2]
        plate_text = extract_plate_text(cropped)
        valid = is_valid_plate(plate_text)
        detection_data.append({
            "plate": plate_text,
            "valid": valid,
            "coordinates": [x1, y1, x2, y2]
        })

    return jsonify({"detections": detection_data})

if __name__ == '__main__':
    os.makedirs("uploads", exist_ok=True)
    app.run(debug=True)
