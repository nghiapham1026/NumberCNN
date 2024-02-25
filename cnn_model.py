# Import necessary libraries
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from preprocess import X_train, X_test, y_train, y_test

# Convert DataFrame to NumPy array and reshape the data to fit the model
X_train_reshaped = X_train.values.reshape(-1, 28, 28, 1)  # Use .values to convert DataFrame to NumPy array
X_test_reshaped = X_test.values.reshape(-1, 28, 28, 1)

# Define the CNN architecture
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer=Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Print a summary of the model
model.summary()

# Train the model
history = model.fit(X_train_reshaped, y_train, epochs=10, validation_split=0.2, batch_size=32)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test_reshaped, y_test)
print(f"Test Accuracy: {test_acc*100:.2f}%")

# Optionally, save the model
model.save('./models/mnist_cnn_model.h5')
print("Model saved successfully.")
