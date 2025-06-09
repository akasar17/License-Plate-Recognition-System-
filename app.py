from flask import Flask, request, jsonify
from flask_cors import CORS
import easyocr
from PIL import Image
import io
import re

app = Flask(__name__)
CORS(app)

reader = easyocr.Reader(['en'])  # EasyOCR reader

# Simple mock vehicle details database
VEHICLE_DB = {
    'MH12AB1234': {'Make': 'Maruti Suzuki', 'Model': 'Swift', 'Year': '2018', 'Color': 'Red'},
    'DL8CAF5032': {'Make': 'Honda', 'Model': 'City', 'Year': '2017', 'Color': 'Black'},
    # Add more sample plates and details here
}

def validate_indian_plate(plate):
    # Simple regex for Indian plates: e.g. MH12AB1234
    pattern = r'^[A-Z]{2}[0-9]{1,2}[A-Z]{1,2}[0-9]{4}$'
    return re.match(pattern, plate) is not None

@app.route('/detect', methods=['POST'])
def detect_plate():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    img_bytes = file.read()
    image = Image.open(io.BytesIO(img_bytes)).convert('RGB')

    # OCR detection
    results = reader.readtext(img_bytes)

    plate_text = None
    for (_, text, prob) in results:
        clean_text = re.sub(r'[^A-Za-z0-9]', '', text.upper())
        if validate_indian_plate(clean_text):
            plate_text = clean_text
            break

    if plate_text:
        valid = True
        details = VEHICLE_DB.get(plate_text, {'Info': 'Details not found in database'})
        return jsonify({'plate': plate_text, 'valid': valid, 'details': details})

    return jsonify({'plate': None, 'valid': False, 'details': 'No valid license plate detected'})

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    message = data.get('message', '').lower()
    # Basic canned responses
    if 'valid' in message:
        reply = "A valid Indian license plate format is like MH12AB1234."
    elif 'help' in message:
        reply = "You can ask me to detect license plates or check vehicle details."
    else:
        reply = "I'm here to help with Indian license plate detection and vehicle info."

    return jsonify({'reply': reply})

if __name__ == '__main__':
    app.run(debug=True)
