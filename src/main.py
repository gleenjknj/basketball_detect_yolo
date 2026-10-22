import cv2
from ultralytics import YOLO
import numpy as np
import camera
import detector

basketball_radius = 0.12
cameraMatrix = np.array([
    [2.95427852e+03, 0.00000000e+00, 1.52927642e+03],
    [0.00000000e+00, 2.95087889e+03, 2.05731533e+03],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
        ]
, dtype = np.float32)

distCoeffs = np.array([ 1.98607362e-01, -8.95678961e-01, -1.01413498e-03,  7.84751031e-04,1.09811863e+00], dtype = np.float32)

camera = camera.Camera()
detector = detector.Detector()


while True:
    frame = camera.read()
    if frame is None:
        break
    frame, result = detector.detector(frame)
    if result is not None:
        for box in result.boxes:
            cx, cy, box_w, box_h = box.xywh[0].tolist()
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            z = (box_h+box_w)/4
            objectPoints = np.array(
                [[-basketball_radius, 0, 0],
                [basketball_radius, 0, 0],
                [0, basketball_radius, 0]]
            , dtype=np.float32)
            imagePoints = np.array(
                [[cx-z, cy],
                [cx+z, cy],
                [cx, cy-z]],dtype=np.float32
            )
            success, rvec, tvec = cv2.solvePnP(
                objectPoints,
                imagePoints,
                cameraMatrix,
                distCoeffs,
                flags=cv2.SOLVEPNP_SQPNP
            )
            if success:
                X = tvec[0][0]
                Y = tvec[1][0]
                Z = tvec[2][0]
                tx = int(x1)
                ty = int(y1+box_h) - 8
                text = f"x = {X:.2f}, y = {Y:.2f}, z = {Z:.2f}"
                cv2.putText(
                    frame,
                    text,
                    (tx,ty),
                    cv2.FONT_HERSHEY_PLAIN,
                    1.0,
                    (0,0,255),
                    3
                )
        cv2.imshow('video', frame)
    if cv2.waitKey(camera.delay) & 0xFF == 27:
        break
camera.release()
cv2.destroyAllWindows()
