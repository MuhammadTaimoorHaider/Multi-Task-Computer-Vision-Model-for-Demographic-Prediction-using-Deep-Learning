"""
Facial Demographic Analysis Web Application
Deployed on Railway.app
Author: Muhammad Taimoor Haider
"""

from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import cv2
import numpy as np
import os
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Model configuration
IMAGE_SIZE = (128, 128)
gender_map = {0: 'Male', 1: 'Female'}
ethnicity_map = {0: 'White', 1: 'Black', 2: 'Asian', 3: 'Indian', 4: 'Others'}

# Load model
print("Loading model...")
model = tf.keras.models.load_model('best_model.h5')
print("Model loaded successfully!")

# Load face detector
face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(face_cascade_path)


def preprocess_image(img_array):
    """Preprocess image for model prediction"""
    try:
        # Resize to model input size
        img_resized = cv2.resize(img_array, IMAGE_SIZE)
        # Normalize pixel values
        img_normalized = img_resized.astype('float32') / 255.0
        # Add batch dimension
        img_batch = np.expand_dims(img_normalized, axis=0)
        return img_batch
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return None


def detect_and_predict(image_array):
    """Detect faces and make predictions"""
    try:
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(60, 60)
        )
        
        results = []
        
        if len(faces) == 0:
            return None, "No face detected in the image. Please upload a clear face photo."
        
        for (x, y, w, h) in faces:
            # Extract face region
            face_roi = image_array[y:y+h, x:x+w]
            
            # Preprocess
            preprocessed = preprocess_image(face_roi)
            
            if preprocessed is not None:
                # Make predictions
                predictions = model.predict(preprocessed, verbose=0)
                
                # Extract predictions
                age = int(round(predictions[0][0][0]))
                age = max(1, min(age, 100))  # Clamp between 1-100
                
                gender_idx = np.argmax(predictions[1][0])
                gender = gender_map[gender_idx]
                gender_confidence = float(predictions[1][0][gender_idx] * 100)
                
                ethnicity_idx = np.argmax(predictions[2][0])
                ethnicity = ethnicity_map[ethnicity_idx]
                ethnicity_confidence = float(predictions[2][0][ethnicity_idx] * 100)
                
                results.append({
                    'bbox': {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)},
                    'age': age,
                    'gender': gender,
                    'gender_confidence': round(gender_confidence, 1),
                    'ethnicity': ethnicity,
                    'ethnicity_confidence': round(ethnicity_confidence, 1)
                })
        
        return results, None
    
    except Exception as e:
        return None, f"Error during prediction: {str(e)}"


@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read image
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image file'}), 400
        
        # Detect and predict
        results, error = detect_and_predict(image)
        
        if error:
            return jsonify({'error': error}), 400
        
        return jsonify({
            'success': True,
            'predictions': results,
            'num_faces': len(results)
        })
    
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500


@app.route('/predict_camera', methods=['POST'])
def predict_camera():
    """Handle camera frame prediction"""
    try:
        data = request.get_json()
        
        if 'image' not in data:
            return jsonify({'error': 'No image data'}), 400
        
        # Decode base64 image
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
        image_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Detect and predict
        results, error = detect_and_predict(image_array)
        
        if error:
            return jsonify({'error': error}), 400
        
        return jsonify({
            'success': True,
            'predictions': results,
            'num_faces': len(results)
        })
    
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
