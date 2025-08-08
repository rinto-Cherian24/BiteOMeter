from flask import Flask, request, jsonify
import base64
import io
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)
model = load_model("nail_biter_model.h5")  # Make sure this file is in backend folder

def preprocess_image(image):
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    img_data = data['image']

    if "," in img_data:
        img_data = img_data.split(",")[1]

    img_bytes = base64.b64decode(img_data)
    image = Image.open(io.BytesIO(img_bytes))
    processed_img = preprocess_image(image)

    pred = model.predict(processed_img)[0][0]
    if pred >= 0.5:
        label = "Bite"
        confidence = round(pred * 100, 2)
    else:
        label = "No Bite"
        confidence = round((1 - pred) * 100, 2)

    return jsonify({"label": label, "confidence": confidence})

if __name__ == '__main__':
    app.run(debug=True)
