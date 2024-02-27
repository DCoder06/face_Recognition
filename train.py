# Import necessary libraries
import tensorflow as tf  
from tensorflow.keras import layers, models  
from sklearn.model_selection import train_test_split  # For splitting dataset into train and test sets
from sklearn.preprocessing import LabelEncoder  # For label encoding
from sklearn.utils import shuffle  
import os  
import numpy as np  

# Define the path to the dataset directory
dataset_path = "./data/"

# Create the dataset directory if it doesn't exist
os.makedirs(dataset_path, exist_ok=True)

# function to load data from a given file path
def load_data(file_path):
    # Load face data from the specified file path
    face_data = np.load(file_path)
    # Create labels array with zeros
    labels = np.zeros((face_data.shape[0], 1))
    return face_data, labels

# Get a list of file names without extension from the dataset directory
file_names = []
for file in os.listdir(dataset_path):
    if file.endswith(".npy"):
        file_names.append(file.split('.')[0])

# Iterate through each file name in the dataset directory
for file_name in file_names:
    print(f"\nTraining on data for {file_name}")
    
    # Construct the file path for the current file name
    file_path = os.path.join(dataset_path, file_name + ".npy")
    # Load data from the file
    face_data, labels = load_data(file_path)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(face_data, labels, test_size=0.2, random_state=42)

    # Load MobileNetV2 model with pre-trained weights on ImageNet dataset
    base_model = tf.keras.applications.MobileNetV2(input_shape=(96, 96, 3), include_top=False, weights='imagenet')
    base_model.trainable = False

    # Build the classification model
    model = models.Sequential([
        base_model,  # Base MobileNetV2 model
        layers.GlobalAveragePooling2D(),  # Global average pooling layer
        layers.Dense(1, activation='sigmoid')  # Dense output layer with sigmoid activation
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    # Evaluate model performance on the test set
    _, accuracy = model.evaluate(X_test, y_test)
    print(f"Model Accuracy: {accuracy}")

    # Save the trained model to a file
    model.save(f'face_recognition_model_{file_name}.h5')
    print(f"Model for {file_name} saved successfully!")

# Print a message indicating the completion of training for all available files
print("\nTraining complete for all available files.")
