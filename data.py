import cv2
import os

# Load the face detection model
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Capture video from webcam
video = cv2.VideoCapture(0)

# Get user input for name
name_id = input("Enter Your Name: ").strip().lower()
dataset_path = 'dataset'

# Ensure the dataset directory exists
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

# Create a directory for the user
user_path = os.path.join(dataset_path, name_id)

if os.path.exists(user_path):
    print("Name Already Taken. Try Again.")
    exit()
else:
    os.makedirs(user_path)

print(f"Saving images to {user_path}")

count = 0
max_images = 300  # Number of images to capture

while True:
    ret, frame = video.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        count += 1
        face = frame[y:y + h, x:x + w]
        image_path = os.path.join(user_path, f"{count}.jpg")
        cv2.imwrite(image_path, face)
        
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    cv2.imshow("Face Capture", frame)

    # Break when max images reached
    if count >= max_images:
        break

    # Press 'q' to exit early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

print(f"Collected {count} images for {name_id}")
