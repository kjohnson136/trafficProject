import cv2
import numpy as np
import csv
from ultralytics import YOLO
import os


# Function to stabilize frames using ECC method
def stabilize_frame(prev_gray, gray, warp_mode=cv2.MOTION_TRANSLATION):
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000, 1e-10)
    try:
        (cc, warp_matrix) = cv2.findTransformECC(prev_gray, gray, warp_matrix, warp_mode, criteria)
    except cv2.error as e:
        print(f"ECC stabilization failed: {e}")
        return gray
    
    stabilized_gray = cv2.warpAffine(gray, warp_matrix, (gray.shape[1], gray.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    return stabilized_gray

# Function to detect moving objects using background subtraction
def detect_moving_objects(frame, bg_subtractor, min_size=500):
    fg_mask = bg_subtractor.apply(frame)
    
    # Threshold and filter the mask to reduce noise
    _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    
    # Find contours of moving objects
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    moving_objects = []
    for contour in contours:
        if cv2.contourArea(contour) >= min_size:
            x, y, w, h = cv2.boundingRect(contour)
            moving_objects.append((x + w // 2, y + h // 2))  # x, y of the center
    
    return moving_objects

# Function to assign unique track IDs to detected objects
def assign_track_ids(current_objects, previous_tracks, max_distance=50):
    new_tracks = {}
    used_ids = set()

    for obj in current_objects:
        best_id = None
        best_distance = float('inf')
        for track_id, prev_obj in previous_tracks.items():
            distance = np.sqrt((obj[0] - prev_obj[0])**2 + (obj[1] - prev_obj[1])**2)
            if distance < max_distance and distance < best_distance:
                best_id = track_id
                best_distance = distance
        
        if best_id is not None:
            new_tracks[best_id] = obj
            used_ids.add(best_id)
        else:
            new_id = max(previous_tracks.keys(), default=0) + 1
            while new_id in used_ids:
                new_id += 1
            new_tracks[new_id] = obj
            used_ids.add(new_id)

    return new_tracks

# Initialize video capture
cap = cv2.VideoCapture('DJI_0104R.mp4')

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Read the first frame
ret, first_frame = cap.read()
if not ret:
    print("Error: Could not read video frame.")
    exit()

# Reduce frame resolution for faster processing
scaling_factor = 0.5
first_frame = cv2.resize(first_frame, None, fx=scaling_factor, fy=scaling_factor)
prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

# Initialize background subtractor
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=False)

# Track objects across frames
tracks = {}

# Output CSV file for trajectories
with open('trajectoriesTEST.csv', 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['frame', 'trackID', 'xCenter', 'yCenter', 'Class'])

    frame_count = 0
    model = YOLO('yolov8s.pt')
    class_name = ""
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        results = model.track(frame, stream=True)
        for result in results:
         classes_names = result.names
         for box in result.boxes:
            if box.conf[0] > 0.4: 
                [x1, y1, x2, y2] = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cls = int(box.cls[0])
                class_name = classes_names[cls]
                color = (255, 0, 0)
                cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                cv2.putText(frame, f'{classes_names[int(box.cls[0])]} {box.conf[0]:.2f}', (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow('frame', frame)
        
        frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Skip frames to reduce processing load
        if frame_count % 5 != 0:
            frame_count += 1
            continue

        # Stabilize the current frame
        stabilized_gray = stabilize_frame(prev_gray, gray)

        # Detect moving objects using background subtraction
        moving_objects = detect_moving_objects(stabilized_gray, bg_subtractor)
        
        # Assign track IDs to the detected objects
        tracks = assign_track_ids(moving_objects, tracks)
        
        # Write detected tracks to CSV and display on video
        for track_id, (x, y) in tracks.items():
            csv_writer.writerow([frame_count, track_id, x, y, class_name])
            cv2.circle(frame, (x, y), 10, (0, 255, 0), 2)
            cv2.putText(frame, str(track_id), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imshow('Tracking', frame)
        
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
        
        prev_gray = stabilized_gray
        frame_count += 1

cap.release()
cv2.destroyAllWindows()

print("CSV file 'trajectoriesTEST.csv' has been created.")