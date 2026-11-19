import cv2
from ultralytics import YOLO

print("Loading YOLO model...")
model = YOLO("models/yolov8n-face.pt")

print("Opening video...")
cap = cv2.VideoCapture("data/sample.mp4")

if not cap.isOpened():
    print("ERROR: Cannot open video")
    exit()

print("Video opened successfully!")

frame_count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        print("Video ended or frame could not be read.")
        break

    frame_count += 1

    if frame_count == 1:
        print("First frame read successfully!")

    results = model.predict(frame, conf=0.25, verbose=False)

    if frame_count == 1:
        print("YOLO prediction completed!")
        print("Faces detected:", len(results[0].boxes))

    annotated_frame = results[0].plot()
    cv2.imshow("YOLO Face Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("Total frames processed:", frame_count)
print("Detection test completed!")