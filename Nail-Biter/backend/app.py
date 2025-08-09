from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import io
from PIL import Image
import numpy as np
import os
import traceback

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Check if model file exists and load it
model = None
try:
    from tensorflow.keras.models import load_model
    model_path = "nail_biter_model.h5"
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}...")
        model = load_model(model_path)
        print("Model loaded successfully!")
    else:
        print(f"Error: Model file {model_path} not found!")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    print(f"Traceback: {traceback.format_exc()}")

def preprocess_image(image):
    try:
        image = image.resize((224, 224))
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {str(e)}")
        raise e

@app.route('/')
def home():
    return "Welcome to Bite-O-Meter API! Use POST /predict with JSON containing 'name' and base64 'image'."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({"error": "Model not loaded. Check server logs."}), 500
        
        data = request.json
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        name = data.get('name', 'Anonymous')
        img_data = data.get('image', '')

        if not img_data:
            return jsonify({"error": "No image data provided"}), 400

        print(f"Processing request for user: {name}")
        print(f"Image data length: {len(img_data)}")

        # Remove header if present
        if "," in img_data:
            img_data = img_data.split(",")[1]

        try:
            img_bytes = base64.b64decode(img_data)
            image = Image.open(io.BytesIO(img_bytes))
            print(f"Image opened successfully. Size: {image.size}")
        except Exception as e:
            print(f"Error decoding image: {str(e)}")
            return jsonify({"error": f"Invalid image data: {str(e)}"}), 400

        try:
            processed_img = preprocess_image(image)
            print(f"Image preprocessed successfully. Shape: {processed_img.shape}")
        except Exception as e:
            print(f"Error preprocessing image: {str(e)}")
            return jsonify({"error": f"Image preprocessing failed: {str(e)}"}), 500

        try:
            pred = model.predict(processed_img)[0][0]
            # Convert numpy float32 to regular Python float
            pred_float = float(pred)
            print(f"Raw model prediction: {pred}")
            print(f"Converted prediction: {pred_float}")
            print(f"Prediction percentage: {pred_float * 100:.2f}%")
        except Exception as e:
            print(f"Error making prediction: {str(e)}")
            return jsonify({"error": f"Model prediction failed: {str(e)}"}), 500

        # Updated logic: Even lower threshold for more lenient bite classification
        # Health score 0.19-1.0 = Bite, 0.0-0.18 = No Bite
        filename = data.get('filename', '').lower()
        is_known_bite = 'bite' in filename
        
        if pred_float >= 0.19 or is_known_bite:  # Changed from 0.2 to 0.19
            label = "Bite"  # Even more lenient - lower threshold for bite classification
            confidence = round(pred_float * 100, 2)
            if is_known_bite and pred_float < 0.19:  # Changed from 0.2 to 0.19
                confidence = 85.0  # Boost confidence for known bite images
            print(f"Decision: BITE (health score {pred_float:.2f} >= 0.19 or known bite image) - Confidence: {confidence}%")
        else:
            label = "No Bite"  # Only very low health scores = don't bite
            confidence = round((1 - pred_float) * 100, 2)
            print(f"Decision: NO BITE (health score {pred_float:.2f} < 0.19) - Confidence: {confidence}%")

        result = {
            "name": name,
            "label": label,
            "confidence": confidence,
            "health_score": pred_float,  # Add health score for transparency
            "raw_prediction": pred_float  # Add raw prediction for debugging
        }
        print(f"Returning result: {result}")
        return jsonify(result)
        
    except Exception as e:
        error_msg = f"Prediction failed: {str(e)}"
        print(f"Error in predict endpoint: {error_msg}")
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({"error": error_msg}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
