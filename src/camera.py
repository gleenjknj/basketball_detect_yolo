import cv2


class Camera:

    def __init__(self):
        self.video = cv2.VideoCapture('data/raw/basket4.mp4')

        if not self.video.isOpened():
            self.video.release()

        fps = self.video.get(cv2.CAP_PROP_FPS)
        self.delay = round(1000 / fps)
        
    
    def read(self):
        ret, frame = self.video.read()

        if not ret:
            return None

        return frame

    def release(self):
        self.video.release()