"""
Facial Demographic Analysis Web Application
Deployed on Railway.app
Author: Muhammad Taimoor Haider
"""

from flask import Flask, render_template, request, jsonify
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Reduce TensorFlow logging

import tensorflow as tf
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import sys
import gc

# Force CPU and optimize memory
tf.config.set_visible_devices([], 'GPU')

app = Flask(__name__)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Model configuration
IMAGE_SIZE = (128, 128)
gender_map = {0: 'Male', 1: 'Female'}
ethnicity_map = {0: 'White', 1: 'Black', 2: 'Asian', 3: 'Indian', 4: 'Others'}

# Global model variable
model = None
model_loading_error = None

def load_model_at_startup():
    """Load model at startup with memory optimization"""
    global model, model_loading_error
    try:
        print("=" * 60, file=sys.stderr)
        print("STARTING MODEL LOAD", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        
        model_path = 'best_model.h5'
        
        if not os.path.exists(model_path):
            error_msg = f"Model file not found: {model_path}"
            print(f"ERROR: {error_msg}", file=sys.stderr)
            print(f"Files in current directory: {os.listdir('.')}", file=sys.stderr)
            model_loading_error = error_msg
            return
        
        file_size = os.path.getsize(model_path) / (1024 * 1024)
        print(f"Model file size: {file_size:.2f} MB", file=sys.stderr)
        
        # Load with minimal memory footprint
        print("Loading model (this may take 60-90 seconds)...", file=sys.stderr)
        model = tf.keras.models.load_model(model_path, compile=False)
        
        # Clear memory
        gc.collect()
        
        print("✓ Model loaded successfully!", file=sys.stderr)
        print(f"Model inputs: {model.input_shape}", file=sys.stderr)
        print(f"Model outputs: {len(model.outputs)} outputs", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        
    except Exception as e:
        error_msg = f"Failed to load model: {str(e)}"
        print(f"ERROR: {error_msg}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        model_loading_error = error_msg

# Load face detector
try:
    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    if face_cascade.empty():
        print("WARNING: Face cascade failed to load", file=sys.stderr)
except Exception as e:
    print(f"ERROR loading face cascade: {e}", file=sys.stderr)


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
    global model, model_loading_error
    
    try:
        # Check if model is loaded
        if model is None:
            if model_loading_error:
                return None, f"Model loading failed: {model_loading_error}"
            else:
                return None, "Model is still loading. Please wait 30 seconds and try again."
        
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
    global model, model_loading_error
    
    return jsonify({
        'status': 'healthy' if model is not None else 'loading',
        'model_loaded': model is not None,
        'model_error': model_loading_error,
        'tensorflow_version': tf.__version__
    })


if __name__ == '__main__':
    # Load model at startup
    load_model_at_startup()
    
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting Flask app on port {port}...", file=sys.stderr)
    app.run(host='0.0.0.0', port=port, debug=False)
else:
    # When run with gunicorn --preload, load model once
    load_model_at_startup()
