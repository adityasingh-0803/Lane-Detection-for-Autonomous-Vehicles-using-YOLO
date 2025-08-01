import cv2
import numpy as np

def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([[
        (100, height), 
        (image.shape[1] - 100, height), 
        (image.shape[1] // 2, int(height * 0.6))
    ]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    return cv2.bitwise_and(image, mask)

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 5)
    return line_image

def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    roi = region_of_interest(edges)
    lines = cv2.HoughLinesP(roi, 2, np.pi / 180, 50, minLineLength=40, maxLineGap=100)
    line_image = display_lines(frame, lines)
    return cv2.addWeighted(frame, 0.8, line_image, 1, 1)

def run_lane_detection(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        result = process_frame(frame)
        cv2.imshow("Lane Detection - Dashcam View", result)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_lane_detection("test_video.mp4")
