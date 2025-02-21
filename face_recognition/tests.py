import cv2

# Open the webcam
cam = cv2.VideoCapture(0)

if not cam.isOpened():
    print("Error: Could not open webcam.")
else:
    print("Webcam opened successfully. Press 'q' to exit.")

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame.")
            break

        cv2.imshow("Test Webcam", frame)

        # 
