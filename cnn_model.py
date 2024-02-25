import numpy as np
from keras.utils import to_categorical
from keras.datasets import mnist
from kerastuner.tuners import RandomSearch
from tuning import CNNHyperModel
from preprocess import X_train, X_test, y_train, y_test

# Convert DataFrame to NumPy array and reshape the data to fit the model
X_train = X_train.values.reshape(-1, 28, 28, 1)  # Use .values to convert DataFrame to NumPy array
X_test = X_test.values.reshape(-1, 28, 28, 1)

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
             validation_split=0.2)

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The hyperparameter search is complete. The optimal number of units in the first densely-connected
layer is {best_hps.get('dense_units')}.
""")

# Since learning_rate might not be defined, use a conditional check or a default value
learning_rate = best_hps.get('learning_rate') if 'learning_rate' in best_hps.values else 'default value'
print(f"The optimal learning rate for the optimizer is {learning_rate}.")

# Build the model with the optimal hyperparameters and train it on the data
model = tuner.hypermodel.build(best_hps)
history = model.fit(X_train, y_train, epochs=10, validation_split=0.2)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_acc*100:.2f}%')

# Optionally, save the model
model.save('./models/mnist_cnn_model.h5')
print("Model saved successfully.")