# main.py

import io
import time # هنستخدمه في الخطوة الرابعة عشان الـ Rolling Update
import numpy as np
import tensorflow as tf
from PIL import Image
from fastapi import FastAPI, UploadFile, HTTPException, Depends
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor
from starlette.concurrency import run_in_threadpool 
# لتجنب مشكلة الـ Blocking I/O في الـ Inference
executor = ThreadPoolExecutor(max_workers=4) 

# تحديد المسار الافتراضي للموديل
MODEL_PATH = 'mnist_model.h5' 

# الموديل هيتحمل هنا عند بداية تشغيل التطبيق
model = None 

# ----------------------------------------------------
# 1. تعريف شكل الخرج (JSON Response Schema) باستخدام Pydantic
# ----------------------------------------------------
class PredictionResult(BaseModel):
    """Schema for the prediction JSON response."""
    prediction: int
    confidence: float
    server_time: str # أضفناها عشان نختبر الـ Rolling Updates بعد كده
    message: str

# ----------------------------------------------------
# 2. إعداد تطبيق FastAPI
# ----------------------------------------------------
app = FastAPI(
    title="Scalable MNIST Classifier API",
    description="A high-performance API for handwritten digit classification."
)

@app.get("/")
def read_root():
    """Simple health check endpoint."""
    return {"status": "ok", "service": "MNIST Prediction API", "version": "1.0"}

@app.on_event("startup")
def load_model():
    """Load the pre-trained MNIST model upon server startup."""
    global model
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("INFO: Model loaded successfully.")
    except Exception as e:
        print(f"ERROR: Could not load model from {MODEL_PATH}. Error: {e}")
        model = None 
        # نترك السيرفر يعمل حتى لو فشل تحميل الموديل، ولكن الـ /predict هيطلع 503

# ----------------------------------------------------
# 3. الـ Endpoint الرئيسي (The /predict Route)
# ----------------------------------------------------
# The actual heavy-lifting (model.predict) is done in a separate thread 
# using the executor, preventing the FastAPI event loop from blocking.

# main.py - The corrected inference_blocking_call function

def inference_blocking_call(contents):
    """Function to run the blocking model prediction in a separate thread."""
    
    # 1. Open the image directly from the byte stream
    # The io.BytesIO is correct for wrapping the bytes read from the file.
    image_stream = io.BytesIO(contents)
    
    # 2. Open the image and convert/resize
    image = Image.open(image_stream).convert("L") 
    image = image.resize((28, 28)) 
    
    # 3. Convert to NumPy array and Normalize (0 to 1)
    image_array = np.array(image) / 255.0 
    
    # 4. CRITICAL: Invert Colors for MNIST
    # Real-world drawings are dark on light; MNIST training data is light on dark.
    # The image is inverted by subtracting the normalized array from 1.0.
    image_array = 1.0 - image_array 
    
    # 5. Prepare Input for Model (Add the batch dimension)
    input_data = image_array.reshape(1, 28, 28) 

    # 6. Inference
    predictions = model.predict(input_data, verbose=0)[0]
    probabilities = tf.nn.softmax(predictions).numpy()
    
    # 7. Extract Result
    predicted_class = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_class])
    
    return predicted_class, confidence


@app.post("/predict", response_model=PredictionResult)
async def predict_digit(file: UploadFile):
    """Receives an image and returns the predicted digit (0-9)."""
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not yet loaded or failed to load. Please check server logs.")

    
    try:
        # 1. قراءة الصورة asynchronously
        contents = await file.read()
        
        # 2. تشغيل الـ Inference (العملية الثقيلة) باستخدام الطريقة المعيارية:
        predicted_class, confidence = await run_in_threadpool(
            inference_blocking_call, contents # لا تحتاج لـ executor هنا
        )
        
        return PredictionResult(
            prediction=predicted_class,
            confidence=confidence,
            server_time=time.strftime("%H:%M:%S"),
            message="Prediction completed successfully."
        )
        
    except Exception as e:
        print(f"Prediction failed: {e}")
        raise HTTPException(status_code=400, detail=f"Error processing image or prediction: {e}")