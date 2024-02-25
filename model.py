# Import necessary libraries
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
from preprocess import X_train, X_test, y_train, y_test

# Define the neural network architecture
model = Sequential([
    Flatten(input_shape=(784,)),  # Flatten the 28x28 images into vectors of size 784
    Dense(128, activation='relu'),  # A dense layer with 128 neurons and ReLU activation
    Dense(64, activation='relu'),   # A second dense layer with 64 neurons
    Dense(10, activation='softmax') # Output layer with 10 neurons for each digit and softmax activation
])

# Compile the model
model.compile(optimizer=Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Print a summary of the model
model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=10, validation_split=0.2, batch_size=32)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc*100:.2f}%")

# Optionally, save the model
model.save('./models/mnist_model.h5')
print("Model saved successfully.")
