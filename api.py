from flask import Flask, request, Response,jsonify
import cv2
import numpy as np
import threading
import time
from eye_tracking import EyeTracker

app = Flask(__name__)
eye_tracker = EyeTracker()

# Global variables for video capture and processing
video_capture = None
frame_processed = None
frame_available = False

def process_video():
    global video_capture, frame_processed, frame_available
    while True:
        ret, frame = video_capture.read()
        if ret:
            processed_frame = eye_tracker.process_frames(frame)
            frame_processed = processed_frame
            frame_available = True
        else:
            frame_available = False
            break

def start_video_capture():
    global video_capture
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("Error: Unable to open video capture device")
        return
    thread = threading.Thread(target=process_video)
    thread.daemon = True
    thread.start()

@app.route('/start_video', methods=['POST'])
def start_video():
    start_video_capture()
    return "Video capture started"

@app.route('/get_state', methods=['GET'])
def get_state():
    global frame_available
    if frame_available:
        return jsonify({'state': eye_tracker.get_current_state()}), 200
    else:
        return jsonify({'error': 'No processed frame available'}), 404

@app.route('/stop_video', methods=['POST'])
def stop_video():
    global video_capture
    if video_capture is not None:
        video_capture.release()
        video_capture = None
    return "Video capture stopped"

if __name__ == '__main__':
    app.run(debug=True)
