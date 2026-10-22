from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO(r"weights/pretrained/yolo26n.pt")
    model.train(
        data = r"data/roboflow/data.yaml",
        epochs = 10,
        imgsz = 640,
        batch = 2,
        cache=False,
        workers = 0
    )