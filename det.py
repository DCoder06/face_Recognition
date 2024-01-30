import cv2
import numpy as np
import os
import mediapipe as mp

dataset_path = "./data/"
os.makedirs(dataset_path, exist_ok=True)

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

cam = cv2.VideoCapture(0)

fileName = input("Enter the name of the person: ")

faceData = []
offset = 20
skip = 0

while len(faceData) <= 50:
    success, img = cam.read()

    if not success:
        print("Reading Camera Failed!")
        break

    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = face_detection.process(rgb_img)

    if results.detections:
        detection = results.detections[-1]
        bboxC = detection.location_data.relative_bounding_box

        ih, iw, _ = img.shape
        x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cropped_face = img[y - offset : y + h + offset, x - offset : x + w + offset]

        if cropped_face.shape[0] > 0 and cropped_face.shape[1] > 0:
            # cropped_face = cv2.resize(cropped_face, (100, 100))
            cropped_face = cv2.resize(cropped_face, (96, 96))
            cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
            skip += 1

            if skip % 10 == 0:
                faceData.append(cropped_face)
                print("Saved so far: " + str(len(faceData)))

    cv2.imshow("Image Window", img)

    if 'cropped_face' in locals() and cropped_face.shape[0] > 0 and cropped_face.shape[1] > 0:
        cv2.imshow("Cropped Face", cropped_face)

    key = cv2.waitKeyEx(1) & 0xFF
    if key == ord("q") or key == 27:
        break

faceData = np.asarray(faceData)
faceData = faceData.reshape((-1, 96, 96, 3))

print(faceData.shape)

filePath = os.path.join(dataset_path, fileName + ".npy")
np.save(filePath, faceData)
print("Data Saved Successfully: " + filePath)


cam.release()
cv2.destroyAllWindows()
