from flask import Flask, request, jsonify
import cv2
import numpy as np
import mediapipe as mp
import utils
import math

class EyeTracker:
    def __init__(self):
        # Initialize Mediapipe face mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        # Variables
        self.frame_counter = 0
        self.CLOSED_EYES_FRAME = 3
        self.CEF_COUNTER = 0
        self.TOTAL_BLINKS = 0
        self.current_state = "Undetermined"
        self.headpos="Undetermined"

        # face bounder indices 
        self.FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

        # lips indices for Landmarks
        self.LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 185, 40, 39, 37, 0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78]
        self.LOWER_LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
        self.UPPER_LIPS = [185, 40, 39, 37, 0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78] 

        # Left eyes indices 
        self.LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398]
        self.LEFT_EYEBROW = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]

        # Right eyes indices
        self.RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246]  
        self.RIGHT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]

    def landmarksDetection(self, img, results, draw=False):
        img_height, img_width = img.shape[:2]
        mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in results.multi_face_landmarks[0].landmark]
        if draw:
            [cv2.circle(img, p, 2, (0, 255, 0), -1) for p in mesh_coord]
        return mesh_coord

    def euclideanDistance(self, point, point1):
        x, y = point
        x1, y1 = point1
        distance = math.sqrt((x1 - x)**2 + (y1 - y)**2)
        return distance

    def blinkRatio(self, img, landmarks, right_indices, left_indices):
        rh_right = landmarks[right_indices[0]]
        rh_left = landmarks[right_indices[8]]
        rv_top = landmarks[right_indices[12]]
        rv_bottom = landmarks[right_indices[4]]

        lh_right = landmarks[left_indices[0]]
        lh_left = landmarks[left_indices[8]]
        lv_top = landmarks[left_indices[12]]
        lv_bottom = landmarks[left_indices[4]]

        rhDistance = self.euclideanDistance(rh_right, rh_left)
        rvDistance = self.euclideanDistance(rv_top, rv_bottom)
        lvDistance = self.euclideanDistance(lv_top, lv_bottom)
        lhDistance = self.euclideanDistance(lh_right, lh_left)

        reRatio = rhDistance / rvDistance
        leRatio = lhDistance / lvDistance

        ratio = (reRatio + leRatio) / 2
        return ratio 

    def eyesExtractor(self, img, right_eye_coords, left_eye_coords):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dim = gray.shape
        mask = np.zeros(dim, dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(right_eye_coords, dtype=np.int32)], 255)
        cv2.fillPoly(mask, [np.array(left_eye_coords, dtype=np.int32)], 255)
        eyes = cv2.bitwise_and(gray, gray, mask=mask)
        eyes[mask == 0] = 155

        r_max_x = max(right_eye_coords, key=lambda item: item[0])[0]
        r_min_x = min(right_eye_coords, key=lambda item: item[0])[0]
        r_max_y = max(right_eye_coords, key=lambda item: item[1])[1]
        r_min_y = min(right_eye_coords, key=lambda item: item[1])[1]

        l_max_x = max(left_eye_coords, key=lambda item: item[0])[0]
        l_min_x = min(left_eye_coords, key=lambda item: item[0])[0]
        l_max_y = max(left_eye_coords, key=lambda item: item[1])[1]
        l_min_y = min(left_eye_coords, key=lambda item: item[1])[1]

        cropped_right = eyes[r_min_y: r_max_y, r_min_x: r_max_x]
        cropped_left = eyes[l_min_y: l_max_y, l_min_x: l_max_x]
        return cropped_right, cropped_left

    def positionEstimator(self, cropped_eye):
        h, w = cropped_eye.shape
        gaussain_blur = cv2.GaussianBlur(cropped_eye, (9,9), 0)
        median_blur = cv2.medianBlur(gaussain_blur, 3)
        ret, threshed_eye = cv2.threshold(median_blur, 130, 255, cv2.THRESH_BINARY)
        piece = int(w / 3) 
        right_piece = threshed_eye[0:h, 0:piece]
        center_piece = threshed_eye[0:h, piece: piece+piece]
        left_piece = threshed_eye[0:h, piece +piece:w]
        eye_position, color = self.pixelCounter(right_piece, center_piece, left_piece)
        return eye_position, color 

    def pixelCounter(self, first_piece, second_piece, third_piece):
        right_part = np.sum(first_piece == 0)
        center_part = np.sum(second_piece == 0)
        left_part = np.sum(third_piece == 0)
        eye_parts = [right_part, center_part, left_part]
        max_index = eye_parts.index(max(eye_parts))
        pos_eye = '' 
        if max_index == 0:
            pos_eye = "RIGHT"
            color = [utils.BLACK, utils.GREEN]
        elif max_index == 1:
            pos_eye = 'CENTER'
            color = [utils.YELLOW, utils.PINK]
        elif max_index == 2:
            pos_eye = 'LEFT'
            color = [utils.GRAY, utils.YELLOW]
        else:
            pos_eye = "Closed"
            color = [utils.GRAY, utils.YELLOW]
        return pos_eye, color
        
    def estimate_head_position(self, landmarks):
        if landmarks:
            # Extract specific landmarks
            nose = landmarks[168]
            left_eye = landmarks[159]
            right_eye = landmarks[386]
            left_ear = landmarks[234]
            right_ear = landmarks[454]

            # Calculate distances between landmarks
            eye_distance = np.linalg.norm(np.array(left_eye) - np.array(right_eye))
            ear_distance = np.linalg.norm(np.array(left_ear) - np.array(right_ear))
            nose_distance = np.linalg.norm(np.array(left_eye) - np.array(right_eye))
            
            # Calculate head tilt based on eye and ear distances
            if(ear_distance!=0):
             tilt_ratio = eye_distance / ear_distance
            else :
                pass

            # Determine head position based on tilt ratio
            if tilt_ratio > 0.7:
                self.headpos = "Up"
            elif self.headpos < 0.5:
                head_position = "Down"
            else:
                self.headpos = "Straight"

            return self.headpos
        else:
            return "Unknown"

    def process_frames(self, frame):
        frame_height, frame_width = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            mesh_coords = self.landmarksDetection(frame, results, False)
            ratio = self.blinkRatio(frame, mesh_coords, self.RIGHT_EYE, self.LEFT_EYE)
            if ratio > 5.5:
                self.CEF_COUNTER += 1
            else:
                if self.CEF_COUNTER > self.CLOSED_EYES_FRAME:
                    self.TOTAL_BLINKS += 1
                    self.CEF_COUNTER = 0

            right_coords = [mesh_coords[p] for p in self.RIGHT_EYE]
            left_coords = [mesh_coords[p] for p in self.LEFT_EYE]
            crop_right, crop_left = self.eyesExtractor(frame, right_coords, left_coords)

            if self.positionEstimator(crop_right) != self.positionEstimator(crop_left):
                self.current_state="Distracted"
            else:
                self.current_state="Definitely Focused"

            eye_position, color = self.positionEstimator(crop_right)
            eye_position_left, color = self.positionEstimator(crop_left)
            

        return frame
    def get_current_state(self):
        return self.current_state
