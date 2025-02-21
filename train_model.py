import os
import django

# Set up Django settings
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "facial_attendance.settings")
django.setup()

# Now you can import your function and use it
# facial_attendance/face_recognition/train_model.py
import cv2
import os
import numpy as np
from PIL import Image

def train_faces(training_dir='media/employee_photos'):
    recognizer = cv2.face.LBPHFaceRecognizer_create()  # Now works
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    image_paths = [os.path.join(training_dir, f) for f in os.listdir(training_dir)]
    face_samples, ids = [], []

    for image_path in image_paths:
        img = Image.open(image_path).convert('L')  # Convert to grayscale
        img_np = np.array(img, 'uint8')
        user_id = int(os.path.split(image_path)[-1].split(".")[0])
        faces = detector.detectMultiScale(img_np)
        for (x, y, w, h) in faces:
            face_samples.append(img_np[y:y + h, x:x + w])
            ids.append(user_id)

    recognizer.train(face_samples, np.array(ids))
    recognizer.write('trained_model.yml')

# Run the training function
train_faces()
