import pandas as pd
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

# Load the dataset
df = pd.read_csv('./data/mnist_train.csv')
print("Dataset loaded successfully.")

# Split into features (X) and labels (y)
X = df.drop('label', axis=1)
y = df['label']
print(f"Dataset split into features and labels: {X.shape[0]} samples, {X.shape[1]} features.")

# Normalize the features to be in the range [0, 1]
X = X / 255.0
print("Features normalized.")

# Optionally, convert labels to one-hot encoding
y_one_hot = to_categorical(y, num_classes=10)
print("Labels converted to one-hot encoding.")

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)
print(f"Data split into training and testing sets: Training set {X_train.shape[0]} samples, Testing set {X_test.shape[0]} samples.")

# Output some information about the normalized data
print(f"Sample of normalized features (first row): {X.iloc[0].values[:5]}")
print(f"Sample of one-hot encoded labels (first row): {y_one_hot[0]}")

# Indicate script completion
print("Preprocessing completed.")
