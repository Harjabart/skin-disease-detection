import os
import gc

# Limit TensorFlow CPU threads and log verbosity BEFORE importing TF
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'

import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from gradcam import generate_and_save_gradcam

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Ensure upload directory exists at app startup
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load Trained Model
MODEL_PATH = 'skin_disease_cnn_model.h5'
model = tf.keras.models.load_model(MODEL_PATH)

# Supported Disease Categories
CLASS_NAMES = [
    'Actinic keratosis', 
    'Basal cell carcinoma', 
    'Dermatofibroma', 
    'Melanoma', 
    'Melanocytic nevus', 
    'Seborrheic keratosis', 
    'Vascular lesion'
]

# Identify final convolutional layer for MobileNetV2
LAST_CONV_LAYER_NAME = "Conv_1" 

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='No file uploaded.')
        
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error='No selected file.')

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Preprocessing
            img = tf.keras.preprocessing.image.load_img(filepath, target_size=(224, 224))
            img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
            img_tensor = np.expand_dims(img_array, axis=0)

            # Low-memory Model Inference
            predictions = model(img_tensor, training=False).numpy()[0]
            predicted_idx = int(np.argmax(predictions))
            predicted_class = CLASS_NAMES[predicted_idx]
            confidence = round(float(predictions[predicted_idx]) * 100, 2)

            # Class Probabilities Breakdown
            probabilities = {CLASS_NAMES[i]: round(float(predictions[i]) * 100, 2) for i in range(len(CLASS_NAMES))}

            # Generate Grad-CAM Heatmap
            gradcam_filename = 'gradcam_' + filename
            gradcam_path = os.path.join(app.config['UPLOAD_FOLDER'], gradcam_filename)
            generate_and_save_gradcam(filepath, model, LAST_CONV_LAYER_NAME, gradcam_path)

            # Explicit Garbage Collection
            del img, img_array, img_tensor, predictions
            gc.collect()

            return render_template(
                'index.html',
                image_path=filepath,
                gradcam_path=gradcam_path,
                prediction=predicted_class,
                confidence=confidence,
                probabilities=probabilities
            )

    return render_template('index.html')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port, debug=False)