from ultralytics import YOLO

model = YOLO("weights/pretrained/yolo26n.pt")

results = model.predict(
    source="data/raw/basket2.mp4",
    # source=0,
    conf=0.25,
    device=0,
    save=False,
    project="outputs",
    name="predict",
    exist_ok=True,
    show=True,
)

# for box in results[0].boxes:
#     cls_id = int(box.cls[0])
#     class_name = results[0].names[cls_id]
#     confidence = float(box.conf[0])
#     x1, y1, x2, y2 = box.xyxy[0].tolist()

#     print(class_name, confidence, x1, y1, x2, y2)