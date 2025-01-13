from flask import Flask, render_template, request, send_file
from ultralytics import YOLO
import os
from PIL import Image
import io
import boto3

# Initialize the Flask application
app = Flask(__name__)

# AWS S3 Setup: Initialize the S3 client
s3 = boto3.client('s3')


import boto3
import os
from ultralytics import YOLO

# S3 configuration
bucket_name = 'mydeploybusketdetec'  # Your S3 bucket name
object_key = 'best (2).pt'  # Object key for the file in S3
download_path = 'best.pt'  # Path where you want to save the file locally

# Create S3 client
s3 = boto3.client('s3')

# Check if download path has a directory part (in case it's a folder)
download_dir = os.path.dirname(download_path)
if download_dir and not os.path.exists(download_dir):
    os.makedirs(download_dir)  # Create the directory only if it exists

# Download the model file from S3
try:
    s3.download_file(bucket_name, object_key, download_path)
    print(f"Model successfully downloaded to {download_path}")
except Exception as e:
    print(f"Error downloading model from S3: {e}")
    exit(1)  # Exit the program if download fails

# Load the model
try:
    model = YOLO(download_path)  # Load from the local path after downloading
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)


# Function to download the model from S3
def download_model_from_s3():
    try:
        # Download the model file from S3
        s3.download_file(bucket_name, object_key, download_path)
        print(f"Model downloaded to {download_path}")
    except Exception as e:
        print(f"Error downloading model from S3: {str(e)}")

# Download the model before starting the app
download_model_from_s3()

# Load the trained YOLO model (update the path to your model)
model = YOLO(download_path)  # Load from the local path after downloading

# Route to serve the home page
@app.route('/')
def home():
    return render_template('index1.html')  # Make sure the template file is named correctly

# Route to handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Check if an image is included in the request
    if 'image' not in request.files:
        return {"error": "No image file found in request"}, 400

    uploaded_file = request.files['image']
    if uploaded_file.filename == '':
        return {"error": "No selected file"}, 400

    try:
        # Open the uploaded image
        image = Image.open(uploaded_file)

        # Perform inference on the image
        results = model(image)

        # Get the first result (YOLO returns a list of results)
        result = results[0]

        # Convert the result to a numpy array (image with bounding boxes)
        result_image = result.plot()  # This function should plot the bounding boxes

        # Convert the numpy array to a PIL Image
        result_image_pil = Image.fromarray(result_image)

        # Save the result image to a BytesIO object
        output = io.BytesIO()
        result_image_pil.save(output, format="JPEG")
        output.seek(0)

        # Return the processed image with bounding boxes
        return send_file(output, mimetype='image/jpeg')

    except Exception as e:
        return {"error": f"Error processing the image: {str(e)}"}, 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
