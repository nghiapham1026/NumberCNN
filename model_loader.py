from keras.models import load_model
import numpy as np
from PIL import Image

model = load_model('./models/mnist_cnn_model.h5')

def predict(image_file):
    image = Image.open(image_file).convert('L')  # Convert to grayscale
    image = image.resize((28, 28))
    image_array = np.array(image)
    image_array = image_array / 255.0  # Normalize
    image_array = image_array.reshape(1, 28, 28, 1)  # Reshape for the model
    prediction = model.predict(image_array)
    return np.argmax(prediction)
