from tensorflow.keras.models import model_from_json
import json
import cv2
import numpy as np
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    image = image / 255.0 
    return image
with open("my_cnn_model.json", "r") as json_file:
    loaded_model_json = json_file.read()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("my_cnn_model_weights.h5")
def classify_image(image_path, model):
    preprocessed_image = preprocess_image(image_path)
    preprocessed_image = np.expand_dims(preprocessed_image, axis=0)
    prediction = model.predict(preprocessed_image)
    return prediction[0][0]
test_image_path = "test_anemia.jpg"
prediction = classify_image(test_image_path, loaded_model)
if prediction > 0.5:
    print("Anemia")
else:
    print("Non-Anemia")
