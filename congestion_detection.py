import cv2
import numpy as np

# Load YOLO model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

cap = cv2.VideoCapture("test1_video.mp4")

conf_threshold = 0.5
nms_threshold = 0.4   

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, channels = frame.shape

    # Prepare input for YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0,0,0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold and classes[class_id] in ['car', 'bus', 'truck', 'motorbike']:
                
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    vehicle_count = len(indexes)

    
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = (0, 255, 0)  
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {int(confidence*100)}%", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    
    if vehicle_count > 10:
        cv2.putText(frame, f"Congestion Detected! Vehicles: {vehicle_count}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(frame, f"Traffic Normal. Vehicles: {vehicle_count}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    
    cv2.imshow("Traffic Detection", frame)

   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
