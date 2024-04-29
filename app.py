from flask import Flask, request, Response, jsonify
import cv2
import numpy as np
import threading
import time
from eye_tracking import EyeTracker  # Assuming you have the EyeTracker class implemented
import os

app = Flask(__name__)
eye_tracker = EyeTracker()

# Global variables for video capture and processing
frame_processed = None
frame_available = False

def process_video(frame):
    global frame_processed, frame_available
    if frame is not None:
        processed_frame = eye_tracker.process_frames(frame)
        frame_processed = processed_frame
        frame_available = True
    else:
        frame_available = False

@app.route('/start_video', methods=['POST'])
def start_video():
    global frame_processed, frame_available
    frame_data = request.data  # Receive frame data from the Flutter mobile app
    frame_array = np.frombuffer(frame_data, dtype=np.uint8)  # Convert data to numpy array
    frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)  # Decode frame
    thread = threading.Thread(target=process_video, args=(frame,))
    thread.daemon = True
    thread.start()
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
    global frame_available
    frame_available = False
    return "Video capture stopped"

if __name__ == '__main__':
    port = int(os.environ.get('HOST_PORT', 10000))  # Default to 5000 if not set
    app.run(host='0.0.0.0', port=port)  # Listen on all interfaces (0.0.0.0)
