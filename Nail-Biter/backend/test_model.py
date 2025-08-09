import numpy as np
from PIL import Image
import os
import traceback

def test_model():
    try:
        # Test model loading
        print("Testing model loading...")
        from tensorflow.keras.models import load_model
        
        model_path = "nail_biter_model.h5"
        if not os.path.exists(model_path):
            print(f"ERROR: Model file {model_path} not found!")
            return False
            
        print(f"Model file found: {model_path}")
        model = load_model(model_path)
        print("✅ Model loaded successfully!")
        
        # Test with dummy data
        print("Testing prediction with dummy data...")
        dummy_image = np.random.rand(224, 224, 3) * 255
        dummy_image = dummy_image.astype(np.uint8)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(dummy_image)
        
        # Preprocess
        processed_img = pil_image.resize((224, 224))
        img_array = np.array(processed_img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        print(f"Input shape: {img_array.shape}")
        
        # Make prediction
        pred = model.predict(img_array)[0][0]
        print(f"✅ Prediction successful: {pred}")
        
        if pred >= 0.5:
            label = "Bite"
            confidence = round(pred * 100, 2)
        else:
            label = "No Bite"
            confidence = round((1 - pred) * 100, 2)
            
        print(f"✅ Result: {label} ({confidence}% confidence)")
        return True
        
    except Exception as e:
        print(f"❌ Error testing model: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    print("=== Model Test ===")
    success = test_model()
    if success:
        print("✅ All tests passed! Model is working correctly.")
    else:
        print("❌ Model test failed. Check the errors above.")
