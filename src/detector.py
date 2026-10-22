import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np

model = YOLO("runs/detect/train-3/weights/best.pt")


class Detector:
    def __init__(self):
         pass
    def detector(self, frame):
        results = model.predict(
        frame,
        conf=0.6,
        device=0,
        verbose=False,
    )
        results = results[0]
        frame = results.plot()
        return frame, results