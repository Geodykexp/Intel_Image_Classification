import json
import numpy as np
import pandas as pd
from tensorflow import keras # type: ignore
from tensorflow.keras.preprocessing.image import load_img, img_to_array # type: ignore
from tensorflow.keras.applications.xception import preprocess_input # type: ignore
import base64
from io import BytesIO
from PIL import Image

# Load model at cold start
model = keras.models.load_model('xception_v2_10_0.917.keras')
classes = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

def lambda_handler(event, context):
    try:
        # Extract base64 image from event
        image_data = event.get('image')
        
        if not image_data:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'No image provided'})
            }
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
        
        # Preprocess image
        image = image.resize((150, 150))
        x = img_to_array(image)
        X = np.array([x])
        X = preprocess_input(X)
        
        # Make prediction
        prediction = model.predict(X)
        predicted_class = classes[np.argmax(prediction[0])]
        confidence = float(np.max(prediction[0]))
        
        # Return response
        return {
            'statusCode': 200,
            'body': json.dumps({
                'predicted_class': predicted_class,
                'confidence': confidence,
                'all_predictions': {
                    classes[i]: float(prediction[0][i]) 
                    for i in range(len(classes))
                }
            })
        }
    
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }