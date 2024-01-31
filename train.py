import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
import os
import numpy as np

dataset_path = "./data/"
os.makedirs(dataset_path, exist_ok=True)

def load_data(file_path):
    face_data = np.load(file_path)
    labels = np.zeros((face_data.shape[0], 1))
    return face_data, labels

file_names = [file.split('.')[0] for file in os.listdir(dataset_path) if file.endswith(".npy")]

for file_name in file_names:
    print(f"\nTraining on data for {file_name}")
    
    file_path = os.path.join(dataset_path, file_name + ".npy")
    face_data, labels = load_data(file_path)

    X_train, X_test, y_train, y_test = train_test_split(face_data, labels, test_size=0.2, random_state=42)

    base_model = tf.keras.applications.MobileNetV2(input_shape=(96, 96, 3), include_top=False, weights='imagenet')
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    _, accuracy = model.evaluate(X_test, y_test)
    print(f"Model Accuracy: {accuracy}")

    model.save(f'face_recognition_model_{file_name}.h5')
    print(f"Model for {file_name} saved successfully!")

print("\nTraining complete for all available files.")
