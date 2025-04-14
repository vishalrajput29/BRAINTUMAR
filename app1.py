from flask import Flask, render_template, request, send_file
from ultralytics import YOLO
import gdown
import os
from PIL import Image
import io

# Google Drive file info
GDRIVE_FILE_ID = "1nFgDiO15zwI8Z6fKujd4xRNMGh_XliK5"
MODEL_FILENAME = "best.pt"

# Step 1: Download model from Google Drive if not already present
def download_model_from_gdrive():
    if not os.path.exists(MODEL_FILENAME):
        url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
        print("📥 Downloading model from Google Drive...")
        gdown.download(url, MODEL_FILENAME, quiet=False)
        print("✅ Model downloaded successfully.")

# Step 2: Download the model
download_model_from_gdrive()

# Step 3: Load the model
model = YOLO(MODEL_FILENAME)

# Flask setup
app = Flask(__name__)

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
    result_image = result.plot()  # Image with bounding boxes

    # Convert result image to JPEG in-memory
    result_image_pil = Image.fromarray(result_image)
    output = io.BytesIO()
    result_image_pil.save(output, format="JPEG")
    output.seek(0)

    return send_file(output, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
