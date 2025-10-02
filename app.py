from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

# Initialize FastAPI app
app = FastAPI()

# Load the model
model = tf.keras.models.load_model("model2.h5")

# Directory to save uploaded images
UPLOAD_FOLDER = "static/uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Preprocess image function
def preprocess_image(image: Image.Image):
    image = image.resize((180, 180))
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    return image

# Home route
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request, "image_file": None, "result": None, "confidence": None})

# Prediction route
@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    if not file:
        return templates.TemplateResponse("predict.html", {"request": request, "image_file": None, "result": "No file uploaded", "confidence": "N/A"})

    try:
        contents = await file.read()
        if not contents:
            return templates.TemplateResponse("predict.html", {"request": request, "image_file": None, "result": "Uploaded file is empty", "confidence": "N/A"})

        # Open and process the image
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        processed_image = preprocess_image(image)

        # Make prediction
        prediction = model.predict(processed_image)
        predicted_class = np.argmax(prediction[0])
        result = "Pneumonia" if predicted_class == 1 else "Normal"
        confidence = f"{prediction[0][predicted_class]:.2f}"

        # Save uploaded image
        filename = "uploaded_image.jpg"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        image.save(filepath)

        return templates.TemplateResponse("predict.html", {"request": request, "image_file": filename, "result": result, "confidence": confidence})

    except Exception as e:
        print(f"Error: {e}")
        return templates.TemplateResponse("predict.html", {"request": request, "image_file": None, "result": "Error processing image", "confidence": str(e)})
