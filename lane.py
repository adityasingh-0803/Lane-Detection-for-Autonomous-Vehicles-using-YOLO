from ultralytics import YOLO
import cv2
import torch
import numpy as np

# Load YOLOv8 nano model
model = YOLO('yolov8n.pt')
device = "cuda" if torch.cuda.is_available() else "cpu"

# Video input
cap = cv2.VideoCapture("test3_video.mp4")
frame_id = 0

def detect_lane(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Canny edge detection
    edges = cv2.Canny(blur, 50, 150)
    
    # Region of interest mask
    height, width = frame.shape[:2]
    mask = np.zeros_like(edges)
    roi_vertices = np.array([[(0, height), (width // 2 - 50, height // 2 + 50), (width // 2 + 50, height // 2 + 50), (width, height)]], dtype=np.int32)
    cv2.fillPoly(mask, roi_vertices, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    # Hough line detection
    lines = cv2.HoughLinesP(masked_edges, rho=1, theta=np.pi/180, threshold=50, minLineLength=40, maxLineGap=100)
    lane_frame = frame.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(lane_frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
    return lane_frame

while True:
    success, frame = cap.read()
    if not success:
        break

    frame_id += 1
    if frame_id % 2 != 0:
        continue

    # Resize for speed
    resized_frame = cv2.resize(frame, (640, 360))

    # Lane detection
    lane_frame = detect_lane(resized_frame)

    # YOLO object detection
    results = model(resized_frame, imgsz=320, conf=0.4, device=device)
    annotated_yolo = results[0].plot()

    # Combine YOLO + Lane overlays (weighted blend)
    combined_output = cv2.addWeighted(annotated_yolo, 0.7, lane_frame, 0.3, 0)

    cv2.imshow("YOLO + Lane Detection", combined_output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
