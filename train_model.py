from tensorflow import keras # type: ignore
from tensorflow.keras.preprocessing.image import load_img, img_to_array # type: ignore
from tensorflow.keras.applications.xception import preprocess_input # type: ignore
import numpy as np
# import os

path = "C:\\Users\\Ikenna George\\Backup\\Documents\\DATA SCIENCE & PROGRAMMING\\DATA SCIENCE PROJECTS\\Intel_Image_Classification\\seg_test\\forest\\20056.jpg"
# path = os.path.abspath("seg_test\forest\20056.jpg")
model = keras.models.load_model('xception_v2_10_0.917.keras')

img = load_img(path, target_size=(150, 150))
x = img_to_array(img)
X = np.array([x])  # Convert single image to a batch.
X = preprocess_input(X)
pred = model.predict(X)
print(X.shape)  # Should print (1, 150, 150, 3)
print(pred)
print(type(pred))
# pred[0]

# print(train_ds.class_indices)
classes = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
predicted_class = classes[np.argmax(pred[0])]
print(f'The predicted class is: {predicted_class}')





