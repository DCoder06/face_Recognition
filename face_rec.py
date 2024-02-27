# Import necessary libraries
import cv2  # For image capture and operations
import numpy as np  # For numerical operations
import os  # For directory and file operations
import tensorflow as tf  # For loading and using trained models
from mediapipe.python.solutions.face_detection import FaceDetection  # For face detection

# Define the path to the dataset
dataset_path = "./data/"
faceData = []  # Initialize a list to store face data
labels = []  # Initialize a list to store labels (names)

# Load dataset from the specified path
for filename in os.listdir(dataset_path):
    if filename.endswith(".npy"):  # Check if the file is a NumPy file
        # Load face data from NumPy file
        data = np.load(os.path.join(dataset_path, filename))
        person_name = filename.split('.')[0]  # Extract person name from filename
        faceData.extend(data)  # Append face data to the list
        labels.extend([person_name] * data.shape[0])  # Append labels for each face data

# Convert lists to NumPy arrays for better handling
faceData = np.asarray(faceData)
labels = np.asarray(labels)

# Normalize face data by scaling pixel values to be between 0 and 1
faceData = faceData / 255.0

# Initialize a dictionary to store models for each unique label
model_dict = {}

# Load a separate model for each unique label found in the dataset
for unique_label in np.unique(labels):
    model_path = f"face_recognition_model_{unique_label}.h5"  # Define model path
    loaded_model = tf.keras.models.load_model(model_path)  # Load model
    model_dict[unique_label] = loaded_model  # Store the loaded model

# Initialize the MediaPipe face detection model
mp_face_detection = FaceDetection(min_detection_confidence=0.5)

# Define a function to recognize faces using the loaded models
def recognize_face(face_data):
    predictions = {}
    
    # Predict the label for the given face data using each stored model
    for unique_label, model in model_dict.items():
        predictions[unique_label] = model.predict(face_data)
    
    return predictions

# Start video capture from the default camera
cam = cv2.VideoCapture(0)

# Continuously capture frames from the camera
while True:
    success, img = cam.read()  # Read a frame

    if not success:
        print("Reading Camera Failed!")  # Print error message if reading frame fails
        break

    # Convert the captured frame from BGR to RGB color space
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Detect faces in the frame
    results = mp_face_detection.process(rgb_img)

    used_labels = set()  # Initialize a set to store used labels to avoid duplicates

    if results.detections:
        # Loop through each detection in the frame
        for detection in results.detections:
            # Extract bounding box information
            bboxC = detection.location_data.relative_bounding_box

            # Convert bounding box coordinates to pixel values
            ih, iw, _ = img.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

            # Draw a rectangle around the detected face
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Crop and preprocess the face for recognition
            cropped_face = img[y:y+h, x:x+w]
            cropped_face = cv2.resize(cropped_face, (96, 96)) / 255.0
            cropped_face = np.expand_dims(cropped_face, axis=0)

            # Recognize the face using the preloaded models
            predictions = recognize_face(cropped_face)

            # Sort predictions based on confidence
            sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)

            # Select the most confident prediction that hasn't been used already
            for label, _ in sorted_predictions:
                if label not in used_labels:
                    recognized_person = label
                    used_labels.add(label)
                    break

            # Display the name of the recognized person on the frame
            cv2.putText(img, recognized_person, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display the frame with the drawn rectangles and labels
    cv2.imshow("Face Recognition", img)

    # Break the loop if 'q' or Esc is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or key == 27:
        break

# Release the camera and destroy all OpenCV windows
cam.release()
cv2.destroyAllWindows()
