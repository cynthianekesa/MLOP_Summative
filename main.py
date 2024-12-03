from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from src.model import WastePredictor

from keras.losses import SparseCategoricalCrossentropy  # type: ignore

app = FastAPI()

# Directories
TRAIN_DIR = './data 1/DATASET/TRAIN'
TEST_DIR = './data 1/DATASET/TEST'
MODEL_DIR = './models/model.pkl'

# Predictor instance
predictor = WastePredictor(TRAIN_DIR, TEST_DIR, MODEL_DIR)

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = tf.keras.models.load_model("/modelsmodel.pkl", custom_objects={'SparseCategoricalCrossentropy': SparseCategoricalCrossentropy})

CLASS_NAMES = ["O", "R",]

@app.get("/YOH")
async def greeting():
    return "Summative manenoz"


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image



@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    predictions = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }
    

@app.post("/rebuild-model/")
def rebuild_model():
    try:
        predictor.load_data()
        predictor.train_model()
        predictor.make_predictions()
        accuracy, report, matrix = predictor.evaluate_model()
        predictor.plot_confusion_matrix()
        predictor.plot_training_history()
        predictor.save_model()
        return {
            "accuracy": accuracy,
            "classification_report": report,
            "confusion_matrix": matrix.tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/confusion-matrix/")
def get_confusion_matrix():
    try:
        return FileResponse('confusion_matrix.png')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/training-history/")
def get_training_history():
    try:
        return FileResponse('training_history.png')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)