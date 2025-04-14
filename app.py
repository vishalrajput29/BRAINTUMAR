from flask import Flask, render_template, request, send_file
from ultralytics import YOLO
import os
from PIL import Image
import io

app = Flask(__name__)

# Load the trained YOLO model
try:
    model = YOLO('best.pt')
    print("✅ YOLO model loaded successfully.")
except Exception as e:
    print(f"❌ Failed to load YOLO model: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return {"error": "No image file found in request"}, 400

    uploaded_file = request.files['image']
    image = Image.open(uploaded_file)

    # Perform inference
    results = model(image)
    result = results[0]
    result_image = result.plot()
    result_image_pil = Image.fromarray(result_image)

    output = io.BytesIO()
    result_image_pil.save(output, format="JPEG")
    output.seek(0)

    return send_file(output, mimetype='image/jpeg')
