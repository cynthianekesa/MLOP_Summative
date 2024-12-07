import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from PIL import Image
import tensorflow as tf
from io import BytesIO

app = FastAPI()

# Load the pre-trained model
MODEL = tf.keras.models.load_model('./models/my_model.h5')
CLASS_NAMES = ['O', 'R']

def read_file_as_image(data) -> np.ndarray:
    img = Image.open(BytesIO(data)).convert('RGB')
    img_resized = img.resize((180, 180), resample=Image.BICUBIC)
    return np.array(img_resized)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = read_file_as_image(await file.read())
        img_batch = np.expand_dims(image, axis=0)
        predictions = MODEL.predict(img_batch)
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = np.max(predictions[0]) * 100  # Convert to percentage
        confidence_rounded = min(round(confidence), 100)  # Round and cap at 100
        return {
            'class': predicted_class,
            'confidence': confidence_rounded  # Return rounded confidence
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
async def main():
    return HTMLResponse(content="<h1>Upload an image for prediction</h1>")