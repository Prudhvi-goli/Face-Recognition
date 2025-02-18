import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import os

# Load the face detection model
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize video capture
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

# Load the trained Keras model
model = keras.models.load_model('keras_model.h5')

# Font for displaying text
font = cv2.FONT_HERSHEY_COMPLEX

# Dynamically generate class names from the dataset directory
dataset_path = 'dataset'
class_names = [name for name in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, name))]

if not class_names:
    print("[ERROR] No classes found in dataset. Ensure you have images collected.")
    cap.release()
    cv2.destroyAllWindows()
    exit()

print(f"[INFO] Class names: {class_names}")

def get_class_name(class_index):
    if 0 <= class_index < len(class_names):
        return class_names[class_index]
    return "Unknown"

print("[INFO] Starting Real-Time Face Recognition...")

while True:
    success, frame = cap.read()
    if not success:
        break

    # Convert frame to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Crop and preprocess the detected face
        face_crop = frame[y:y + h, x:x + w]
        face_resized = cv2.resize(face_crop, (224, 224))
        face_input = np.expand_dims(face_resized, axis=0) / 255.0  # Normalize to [0, 1]

        # Predict the class and confidence score
        prediction = model.predict(face_input)
        class_index = np.argmax(prediction)  # Get the class with the highest probability
        confidence = np.max(prediction)  # Get the confidence score

        # Get the class label
        label = get_class_name(class_index)

        # Draw bounding box and class label on frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y - 40), (x + w, y), (0, 255, 0), -1)
        cv2.putText(frame, f"{label}: {round(confidence * 100, 2)}%", (x, y - 10), font, 0.75, (255, 255, 255), 2)

    # Display the result frame
    cv2.imshow("Real-Time Face Recognition", frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("[INFO] Face Recognition Stopped.")
