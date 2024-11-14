def get_frame(self):
    # Read data from the subprocess's stdout
    data = self.process.stdout.read(1024 * 1024)  # Adjust the buffer size if needed
    if not data:
        print("No data from camera subprocess")  # Debugging line
        return None
    np_arr = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    if frame is None:
        print("Failed to decode frame")  # Debugging line
    else:
        print("Frame captured")  # Debugging line

    return frame
