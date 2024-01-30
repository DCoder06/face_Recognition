import cv2
import numpy as np
import os
import tensorflow as tf
from mediapipe.python.solutions import face_detection


dataset_path = "./data/"
faceData = []
labels = []

for filename in os.listdir(dataset_path):
    if filename.endswith(".npy"):
        data = np.load(os.path.join(dataset_path, filename))
        person_name = filename.split('.')[0]
        faceData.extend(data)
        labels.extend([person_name] * data.shape[0])

faceData = np.asarray(faceData)
labels = np.asarray(labels)


faceData = faceData / 255.0


model_dict = {}

for unique_label in np.unique(labels):
    model_path = f"face_recognition_model_{unique_label}.h5"
    loaded_model = tf.keras.models.load_model(model_path)
    model_dict[unique_label] = loaded_model


mp_face_detection = face_detection.FaceDetection(min_detection_confidence=0.5)


def recognize_face(face_data):
    predictions = {}
    
    for unique_label, model in model_dict.items():
        predictions[unique_label] = model.predict(face_data)

    return predictions


cam = cv2.VideoCapture(0)

while True:
    success, img = cam.read()

    if not success:
        print("Reading Camera Failed!")
        break

    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = mp_face_detection.process(rgb_img)

    used_labels = set()

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box

            ih, iw, _ = img.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cropped_face = img[y:y+h, x:x+w]
            cropped_face = cv2.resize(cropped_face, (96, 96)) / 255.0
            cropped_face = np.expand_dims(cropped_face, axis=0)

            predictions = recognize_face(cropped_face)

            sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)

            for label, _ in sorted_predictions:
                if label not in used_labels:
                    recognized_person = label
                    used_labels.add(label)
                    break

            cv2.putText(img, recognized_person, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Face Recognition", img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or key == 27:
        break

cam.release()
cv2.destroyAllWindows()
