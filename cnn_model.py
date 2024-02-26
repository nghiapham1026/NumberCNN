import numpy as np
from keras.utils import to_categorical
from keras.datasets import mnist
from kerastuner.tuners import RandomSearch
from tuning import CNNHyperModel
from preprocess import X_train, X_test, y_train, y_test
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Convert DataFrame to NumPy array and reshape the data to fit the model
X_train = X_train.values.reshape(-1, 28, 28, 1)
X_test = X_test.values.reshape(-1, 28, 28, 1)

# Manually split the training set into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Define the search space and start the tuning
hypermodel = CNNHyperModel(input_shape=(28, 28, 1), num_classes=10)

tuner = RandomSearch(
    hypermodel,
    objective='val_accuracy',
    max_trials=3,
    executions_per_trial=2,
    directory='tuning',
    project_name='mnist_cnn'
)

tuner.search(X_train, y_train,
             epochs=10,
             validation_data=(X_val, y_val))

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# Define data augmentation generator for the training data
data_augmentation = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
)

# Create a generator for the training data
train_generator = data_augmentation.flow(X_train, y_train, batch_size=32)

# Use the optimal hyperparameters to build the model
model = tuner.hypermodel.build(best_hps)

# Train the model
history = model.fit(train_generator,
                    epochs=10,
                    validation_data=(X_val, y_val))

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_acc*100:.2f}%')

# Optionally, save the model
model.save('./models/mnist_cnn_model.h5')
print("Model saved successfully.")
