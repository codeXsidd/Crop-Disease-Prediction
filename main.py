from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI()

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once
model = tf.keras.models.load_model("crop_disease_model.keras")

@app.get("/")
def home():
    return {"message": "Crop Disease API Running"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image = image.resize((224, 224))

    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    @app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image = image.resize((224, 224))

    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    env_array = np.array([[30, 60, 40]]) 

    prediction = model.predict([img_array, env_array])

    predicted_class = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    return {
        "predicted_class": predicted_class,
        "confidence": confidence
    }
    predicted_class = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    return {
        "predicted_class": predicted_class,
        "confidence": confidence
    }
