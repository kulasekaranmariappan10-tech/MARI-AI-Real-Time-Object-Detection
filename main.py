import cv2
from ultralytics import YOLO

# 1. Load YOLO AI Model
model = YOLO("yolov8n.pt")

# 2. Open Webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera not found!")
    exit()

print("MARI AI Object Detection Started")
print("Press Q to Exit")

# 3. Real-Time Detection
while True:

    # Capture Camera Frame
    success, frame = camera.read()

    if not success:
        print("Frame not available")
        break

    # 4. Detect Objects
    results = model(frame, verbose=False)

    # 5. Draw Detection Boxes
    output_frame = results[0].plot()

    # 6. Display Result
    cv2.imshow(
        "MARI AI - Object Detection",
        output_frame
    )

    # 7. Exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 8. Release Camera
camera.release()

cv2.destroyAllWindows()

print("Detection Stopped!")