from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model("data/crowd2.png", classes=[0])

boxes = results[0].boxes
print("People detections:", len(results[0].boxes))