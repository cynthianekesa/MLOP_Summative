from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import io


app = FastAPI()

origins = [
    "http://127.0.0.1:8080"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the existing Keras model at the start of the application
try:
    model = load_model("models/my_model.h5")  # Adjust the path to your model
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

@app.post("/retrain/")
async def retrain_model(
    files: list[UploadFile] = File(...)
):
    global model  # Use the global model variable

    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    # Prepare the data for retraining
    images = []
    for file in files:
        # Load and preprocess the image
        try:
            # Read the file into a PIL Image
            image = Image.open(io.BytesIO(await file.read()))
            image = image.resize((180, 180))  # Resize to 180x180
            image = img_to_array(image)
            images.append(image)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error processing file {file.filename}: {e}")

    # Convert images to numpy array and normalize
    X_new = np.array(images) / 255.0  # Normalize pixel values

    # Check for None values and empty input
    if X_new is None or len(X_new) == 0:
        raise HTTPException(status_code=400, detail="No valid images provided for retraining.")

    # Debugging: Print the shape of the input data
    print(f"Input shape for retraining: {X_new.shape}")

    # Ensure the input shape matches the model's expected input shape
    if X_new.shape[1:] != (180, 180, 3):
        raise HTTPException(status_code=400, detail=f"Expected input shape (None, 180, 180, 3), but got {X_new.shape}")

    # Retrain the model (you can adjust epochs and batch size as needed)
    #model.fit(X_new, epochs=5, batch_size=32)

    # Save the retrained model
    model.save("models/my_model_retrained.h5")  # Save the retrained model
   
    return JSONResponse(content={"message": "Model retrained successfully!"})

@app.post("/evaluate/")
async def evaluate_model(
    file: UploadFile = File(...)
):
    global model  # Use the global model variable

    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    # Load and preprocess the image
    try:
        # Read the file into a PIL Image
        image = Image.open(io.BytesIO(await file.read()))
        image = image.resize((180, 180))  # Resize to 180x180
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        image = image / 255.0  # Normalize pixel values
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing file {file.filename}: {e}")

    # Debugging: Print the shape of the input data
    print(f"Input shape for evaluation: {image.shape}")

    # Ensure the input shape matches the model's expected input shape
    if image.shape[1:] != (180, 180, 3):
        raise HTTPException(status_code=400, detail=f"Expected input shape (None, 180, 180, 3), but got {image.shape}")

    # Make a prediction
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions, axis=1)[0]  # Get the predicted class index
    confidence = np.max(predictions)  # Get the confidence score

    return JSONResponse(content={
        "predicted_class": int(predicted_class),
        "confidence": float(confidence)
    })

# To run the application, use the command:
# uvicorn your_module_name:app --reload