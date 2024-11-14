import cv2
import subprocess
import numpy as np
from predict import SignLanguagePredictor

class VideoCamera:
    def __init__(self):
        # Initialize any settings here if needed
        pass

    def __del__(self):
        # Release resources if necessary
        pass

    def get_frame(self):
        # Capture an image using libcamera-still
        subprocess.run(["libcamera-still", "-o", "/tmp/frame.jpg", "--timeout", "1", "--nopreview"])

        # Load the image from the file
        frame = cv2.imread("/tmp/frame.jpg")
        return frame if frame is not None else None

def gen_frames(camera, model):
    predictor = SignLanguagePredictor(model)

    while True:
        frame = camera.get_frame()
        if frame is None:
            continue
        action = predictor.predict(frame)
        if action is not None:
            cv2.putText(frame, f'Action: {action}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

