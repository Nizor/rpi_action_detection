import cv2
import numpy as np
import subprocess
from predict import SignLanguagePredictor

class VideoCamera:
    def __init__(self):
        # Start the libcamera-vid subprocess, streaming output to stdout
        self.process = subprocess.Popen(
            ['libcamera-vid', '-t', '0', '--inline', '-o', '-'], stdout=subprocess.PIPE
        )

    def __del__(self):
        self.process.terminate()

    def get_frame(self):
        # Read data from the subprocess's stdout
        data = self.process.stdout.read(1024 * 1024)  # Adjust the buffer size if needed
        if not data:
            return None
        np_arr = np.frombuffer(data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return frame

def gen_frames(camera, model):
    predictor = SignLanguagePredictor(model)

    while True:
        frame = camera.get_frame()
        if frame is None:
            continue
        action = predictor.predict(frame)
        if action is not None:
            cv2.putText(frame, f'Action: {action}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0 , 0), 2, cv2.LINE_AA)
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
