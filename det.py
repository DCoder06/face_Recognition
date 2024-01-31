# Import necessary libraries
import cv2
import numpy as np
import os
import mediapipe as mp

# Define the path to save the dataset
dataset_path = "./data/"
os.makedirs(dataset_path, exist_ok=True)

# Initialize the face detection model from Mediapipe
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Open the webcam
cam = cv2.VideoCapture(0)

# Prompt the user to enter the name of the person
fileName = input("Enter the name of the person: ")

# Initialize an empty list to store face data
faceData = []
# Define offset and skip parameters for face cropping
offset = 20
skip = 0

# Loop to capture face images until 50 images are collected
while len(faceData) <= 50:
    # Read a frame from the camera
    success, img = cam.read()

    # Check if reading from the camera failed
    if not success:
        print("Reading Camera Failed!")
        break

    # Convert the BGR image to RGB
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process the image to detect faces
    results = face_detection.process(rgb_img)

    # Check if faces are detected
    if results.detections:
        # Get the last detected face
        detection = results.detections[-1]
        bboxC = detection.location_data.relative_bounding_box

        ih, iw, _ = img.shape
        # Get the bounding box coordinates
        x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

        # Draw a rectangle around the detected face
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Crop the detected face with an offset
        cropped_face = img[y - offset : y + h + offset, x - offset : x + w + offset]

        # Check if the cropped face is valid
        if cropped_face.shape[0] > 0 and cropped_face.shape[1] > 0:
            # Resize the cropped face to a standard size
            cropped_face = cv2.resize(cropped_face, (96, 96))
            cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
            skip += 1

            # Add the cropped face to the list after every 10 successful detections
            if skip % 10 == 0:
                faceData.append(cropped_face)
                print("Saved so far: " + str(len(faceData)))

    # Display the image window
    cv2.imshow("Image Window", img)

    # Display the cropped face window if available
    if 'cropped_face' in locals() and cropped_face.shape[0] > 0 and cropped_face.shape[1] > 0:
        cv2.imshow("Cropped Face", cropped_face)

    # Wait for a key press and check if it is 'q' or 'Esc' (27) to break the loop
    key = cv2.waitKeyEx(1) & 0xFF
    if key == ord("q") or key == 27:
        break

# Convert faceData list to numpy array and reshape it
faceData = np.asarray(faceData)
faceData = faceData.reshape((-1, 96, 96, 3))

# Print the shape of the collected face data
print(faceData.shape)

# Save the collected face data as a numpy file
filePath = os.path.join(dataset_path, fileName + ".npy")
np.save(filePath, faceData)
print("Data Saved Successfully: " + filePath)

# Release the camera and close all OpenCV windows
cam.release()
cv2.destroyAllWindows()
