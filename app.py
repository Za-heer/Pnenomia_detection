from flask import Flask, render_template, request, send_from_directory
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)

# Load the model
model = tf.keras.models.load_model('model2.h5')

# Directory to save uploaded images
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Preprocess image function (no normalization here — model handles it)
def preprocess_image(image):
    image = image.resize((180, 180))
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    return image

# Home page route
@app.route('/')
def home_page():
    return render_template('home.html', image_file=None, result=None, confidence=None)

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if "file" not in request.files or request.files["file"].filename == "":
        return render_template("predict.html", image_file=None, result="No file uploaded", confidence="N/A")
    
    file = request.files["file"]

    try:
        image_bytes = file.read()
        if not image_bytes:
            return render_template("predict.html", image_file=None, result="Uploaded file is empty", confidence="N/A")

        # Open and process the image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        processed_image = preprocess_image(image)

        # Make prediction
        prediction = model.predict(processed_image)
        predicted_class = np.argmax(prediction[0])
        result = "Pneumonia" if predicted_class == 1 else "Normal"
        confidence = f"{prediction[0][predicted_class]:.2f}"
        print('prediction:', prediction[0])

        # Save the uploaded image
        filename = 'uploaded_image.jpg'
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(filepath)

        return render_template("predict.html", image_file=filename, result=result, confidence=confidence)

    except Exception as e:
        print(f"Error in processing: {e}")
        return render_template("predict.html", image_file=None, result="Error processing image", confidence=str(e))

# Route to serve uploaded images
@app.route('/static/uploads/<filename>')
def send_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True)
    # Set the environment variable to avoid TensorFlow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging (1 = INFO, 2 = WARNING, 3 = ERROR)