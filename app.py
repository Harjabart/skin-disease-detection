import os
import gc
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load lightweight TFLite model
TFLITE_MODEL_PATH = 'skin_disease_model.tflite'
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

CLASS_NAMES = [
    'Actinic keratosis', 
    'Basal cell carcinoma', 
    'Dermatofibroma', 
    'Melanoma', 
    'Melanocytic nevus', 
    'Seborrheic keratosis', 
    'Vascular lesion'
]

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

            # Preprocess image
            img = tf.keras.preprocessing.image.load_img(filepath, target_size=(224, 224))
            img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
            img_tensor = np.expand_dims(img_array, axis=0).astype(np.float32)

            # Fast TFLite Inference
            interpreter.set_tensor(input_details[0]['index'], img_tensor)
            interpreter.invoke()
            predictions = interpreter.get_tensor(output_details[0]['index'])[0]

            predicted_idx = int(np.argmax(predictions))
            predicted_class = CLASS_NAMES[predicted_idx]
            confidence = round(float(predictions[predicted_idx]) * 100, 2)

            probabilities = {CLASS_NAMES[i]: round(float(predictions[i]) * 100, 2) for i in range(len(CLASS_NAMES))}

            del img, img_array, img_tensor, predictions
            gc.collect()

            return render_template(
                'index.html',
                image_path=filepath,
                prediction=predicted_class,
                confidence=confidence,
                probabilities=probabilities
            )

    return render_template('index.html')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port, debug=False)