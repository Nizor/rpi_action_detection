import cv2
import numpy as np
import subprocess

# Start libcamera-vid with subprocess and stream the output
process = subprocess.Popen(['libcamera-vid', '-t', '0', '--inline', '-o', '-'], stdout=subprocess.PIPE)

while True:
    # Read a frame from the process's stdout
    data = process.stdout.read(1024 * 1024)  # Adjust buffer size if needed
    np_arr = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is not None:
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

process.terminate()
cv2.destroyAllWindows()
