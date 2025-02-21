# import cv2
# import os
# import numpy as np
# from PIL import Image
from io import BytesIO
# from django.conf import settings
# import django
# import sys
# sys.path.append("C:/Projects/facial_attendance")

# # Set up Django environment
# os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'facial_attendance.settings')  # Replace with your Django project name
# if not django.conf.settings.configured:
#     django.setup()

# # Function to capture image from webcam and save it with employee name
# def capture_and_save_image():
#     # Initialize the webcam
#     cam = cv2.VideoCapture(0)
#     cv2.namedWindow("Capture Image")

#     # Get employee name from user
#     employee_name = input("Enter the employee name: ")

#     # Create the directory to store employee images if not already exists
#     employee_photos_path = os.path.join(settings.MEDIA_ROOT, 'employee_photos')
#     if not os.path.exists(employee_photos_path):
#         os.makedirs(employee_photos_path)

#     # Initialize counter for number of images captured
#     captured_images_count = 0

#     print("Capturing images. Press 'c' to capture image. Press 'q' to quit.")

#     # Capture images until we have at least 20 images
#     while captured_images_count < 20:
#         ret, frame = cam.read()
#         if not ret:
#             print("Failed to grab frame")
#             break

#         cv2.imshow("Capture Image", frame)

#         # Press 'c' to capture the image
#         if cv2.waitKey(1) & 0xFF == ord('c'):
#             captured_images_count += 1
#             img_path = os.path.join(employee_photos_path, f"{employee_name}_{captured_images_count}.jpg")
#             cv2.imwrite(img_path, frame)
#             print(f"Image {captured_images_count} saved as {img_path}")

#         # Press 'q' to quit if you decide to stop early
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             print("Image capturing stopped.")
#             break

#     print(f"Captured {captured_images_count} images for {employee_name}.")
#     # Release the webcam and close any open windows
#     cam.release()
#     cv2.destroyAllWindows()

# # Function to train the face recognition model using captured images
# def train_faces(training_dir=None):
#     if training_dir is None:
#         training_dir = os.path.join(settings.MEDIA_ROOT, 'employee_photos')

#     recognizer = cv2.face.LBPHFaceRecognizer_create()
#     detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

#     # Check if directory exists and has images
#     if not os.path.exists(training_dir):
#         print("Training directory does not exist!")
#         return
#     image_paths = [os.path.join(training_dir, f) for f in os.listdir(training_dir) if f.endswith('.jpg')]
    
#     if len(image_paths) == 0:
#         print("No image files found in the training directory.")
#         return

#     face_samples, ids = [], []

#     for image_path in image_paths:
#         img = Image.open(image_path).convert('L')  # Convert to grayscale
#         img_np = np.array(img, 'uint8')

#         # Modify this logic to assign a unique numeric ID for each employee
#         employee_name = os.path.splitext(os.path.basename(image_path))[0]
#         user_id = hash(employee_name) % (10 ** 8)  # Use a hash to generate a unique ID for each employee name

#         faces = detector.detectMultiScale(img_np)

#         for (x, y, w, h) in faces:
#             face_samples.append(img_np[y:y + h, x:x + w])
#             ids.append(user_id)

#     if len(face_samples) == 0:
#         print("No faces found in the training images.")
#         return

#     recognizer.train(face_samples, np.array(ids))

#     # Save the trained model
#     recognizer_path = os.path.join(settings.MEDIA_ROOT, 'trained_model.yml')
#     recognizer.write(recognizer_path)
#     print(f"Model trained and saved at {recognizer_path}")

# # Function to recognize face from an image
# def recognize_face(image_data):
#     # Load the trained model
#     recognizer = cv2.face.LBPHFaceRecognizer_create()
#     recognizer.read(os.path.join(settings.MEDIA_ROOT, 'trained_model.yml'))
#     cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

#     # Convert the image data into a format usable by OpenCV
#     image = Image.open(BytesIO(image_data))
#     gray_image = np.array(image.convert('L'))  # Convert image to grayscale
    
#     faces = cascade.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

#     for (x, y, w, h) in faces:
#         id_, confidence = recognizer.predict(gray_image[y:y + h, x:x + w])
#         if confidence < 50:
#             return id_  # Return recognized ID if confidence is below threshold
#         else:
#             return None  # No face recognized with sufficient confidence

#     return None  # Return None if no faces are detected

# # Main function to capture and train
# def capture_and_train():
#     capture_and_save_image()  # Capture and save employee's image
#     train_faces()  # Train the face recognition model

# if __name__ == "__main__":
#     capture_and_train()



import cv2
import os
import numpy as np
from PIL import Image
from django.conf import settings
import django
import sys
sys.path.append("C:/Projects/facial_attendance")

# Set up Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'facial_attendance.settings')  # Replace with your Django project name
if not django.conf.settings.configured:
    django.setup()

# Function to capture image from webcam and save it with employee name
def capture_and_save_image():
    # Initialize the webcam
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("Capture Image")

    # Get employee name from user
    employee_name = input("Enter the employee name: ")

    # Create the directory to store employee images if not already exists
    employee_photos_path = os.path.join(settings.MEDIA_ROOT, 'employee_photos')
    if not os.path.exists(employee_photos_path):
        os.makedirs(employee_photos_path)

    # Initialize counter for number of images captured
    captured_images_count = 0

    print("Capturing images. Press 'c' to capture 20 images automatically. Press 'q' to quit.")

    # Capture 20 images automatically when 'c' is pressed
    while captured_images_count < 20:
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame")
            break

        cv2.imshow("Capture Image", frame)

        # Press 'c' to start automatic capture of 20 images
        if cv2.waitKey(1) & 0xFF == ord('c'):
            print("Capturing 20 images automatically...")
            while captured_images_count < 20:
                ret, frame = cam.read()
                if not ret:
                    print("Failed to grab frame")
                    break

                captured_images_count += 1
                img_path = os.path.join(employee_photos_path, f"{employee_name}_{captured_images_count}.jpg")
                cv2.imwrite(img_path, frame)
                print(f"Image {captured_images_count} saved as {img_path}")
                cv2.imshow("Capture Image", frame)

            break

        # Press 'q' to quit if you decide to stop early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Image capturing stopped.")
            break

    print(f"Captured {captured_images_count} images for {employee_name}.")
    # Release the webcam and close any open windows
    cam.release()
    cv2.destroyAllWindows()

# Function to train the face recognition model using captured images
def train_faces(training_dir=None):
    if training_dir is None:
        training_dir = os.path.join(settings.MEDIA_ROOT, 'employee_photos')

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # Check if directory exists and has images
    if not os.path.exists(training_dir):
        print("Training directory does not exist!")
        return
    image_paths = [os.path.join(training_dir, f) for f in os.listdir(training_dir) if f.endswith('.jpg')]
    
    if len(image_paths) == 0:
        print("No image files found in the training directory.")
        return

    face_samples, ids = [], []

    for image_path in image_paths:
        img = Image.open(image_path).convert('L')  # Convert to grayscale
        img_np = np.array(img, 'uint8')

        # Modify this logic to assign a unique numeric ID for each employee
        employee_name = os.path.splitext(os.path.basename(image_path))[0]
        user_id = hash(employee_name) % (10 ** 8)  # Use a hash to generate a unique ID for each employee name

        faces = detector.detectMultiScale(img_np)

        for (x, y, w, h) in faces:
            face_samples.append(img_np[y:y + h, x:x + w])
            ids.append(user_id)

    if len(face_samples) == 0:
        print("No faces found in the training images.")
        return

    recognizer.train(face_samples, np.array(ids))

    # Save the trained model
    recognizer_path = os.path.join(settings.MEDIA_ROOT, 'trained_model.yml')
    recognizer.write(recognizer_path)
    print(f"Model trained and saved at {recognizer_path}")

# Function to recognize face from an image
def recognize_face(image_data):
    # Load the trained model
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(os.path.join(settings.MEDIA_ROOT, 'trained_model.yml'))
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # Convert the image data into a format usable by OpenCV
    image = Image.open(BytesIO(image_data))
    gray_image = np.array(image.convert('L'))  # Convert image to grayscale
    
    faces = cascade.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        id_, confidence = recognizer.predict(gray_image[y:y + h, x:x + w])
        if confidence < 50:
            return id_  # Return recognized ID if confidence is below threshold
        else:
            return None  # No face recognized with sufficient confidence

    return None  # Return None if no faces are detected

# Main function to capture and train
def capture_and_train():
    capture_and_save_image()  # Capture and save employee's image
    train_faces()  # Train the face recognition model

if __name__ == "__main__":
    capture_and_train()
