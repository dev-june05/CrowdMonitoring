from ultralytics import YOLO
import cv2
import time

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Open webcam
cap = cv2.VideoCapture(0)

# Set webcam resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

print(f"Resolution: {width} x {height}")

# FPS variables
prev_time = time.time()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame.")
        break

    # YOLOv8 + ByteTrack
    results = model.track(
        frame,
        classes=[0],              # Person class only
        persist=True,
        tracker="bytetrack.yaml",
        device=0,                 # RTX 3050
        verbose=False
    )

    # Current visible people
    person_count = len(results[0].boxes)

    # Process tracking IDs
    if results[0].boxes.id is not None:

        ids = results[0].boxes.id.cpu().numpy().astype(int)

    # Draw boxes
    annotated_frame = results[0].plot()

    # FPS calculation
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # FPS display
    cv2.putText(
        annotated_frame,
        f"FPS: {fps:.2f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Current visible people
    cv2.putText(
        annotated_frame,
        f"Current People: {person_count}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    # Display frame
    cv2.namedWindow("Crowd Monitoring", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Crowd Monitoring", 1280, 720)
    cv2.imshow("Crowd Monitoring", annotated_frame)

    # Quit with Q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()