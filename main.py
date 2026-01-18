from fastapi.responses import JSONResponse
from fastapi import FastAPI, File, UploadFile, HTTPException
from tensorflow.keras.preprocessing.image import load_img, img_to_array # type: ignore
from tensorflow.keras.applications.xception import preprocess_input # type: ignore
from tensorflow import keras # type: ignore
import numpy as np
import uvicorn
from io import BytesIO

app = FastAPI(title = "Intel Image Classification API") 

# Load the pre-trained model
model = keras.models.load_model('xception_v2_10_0.917.keras')
classes = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street'] 

@app.get('/')
async def read_root():
    return {"message": "Welcome to the model prediction API!"}

@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    try:
        # Read the uploaded file
        contents = await file.read()
        img = load_img(BytesIO(contents), target_size=(150, 150), color_mode="rgb")
        x = img_to_array(img)
        X = np.array([x])  # Convert single image to a batch
        X = preprocess_input(X)
        # Remove the extra expand_dims - it's not needed

        # Make prediction
        prediction = model.predict(X)
        predicted_class = classes[np.argmax(prediction[0])]
        confidence = float(np.max(prediction[0]))

        # Return the prediction as JSON response
        return {
            "prediction": prediction.tolist(),
            "predicted_class": predicted_class,
            "confidence": confidence
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
    # uvicorn.run(app, host="127.0.0.1", port=9696)