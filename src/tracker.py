
from ultralytics import YOLO


class FaceTracker:

    def __init__(self, model_path="models/yolov8n-face.pt"):
        print("Loading YOLO face tracking model...")

        self.model = YOLO(model_path)

        print("Face tracking model loaded!")

    def track_faces(self, frame):
        results = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False
        )

        tracked_faces = []

        for result in results:

            if result.boxes is None:
                continue

            boxes = result.boxes

            for box in boxes:

                if box.id is None:
                    continue

                x1, y1, x2, y2 = map(
                    int, box.xyxy[0].tolist()
                )

                track_id = int(box.id[0].item())

                confidence = float(box.conf[0].item())

                tracked_faces.append({
                    "track_id": track_id,
                    "bbox": (x1, y1, x2, y2),
                    "confidence": confidence
                })

        return tracked_faces


if __name__ == "__main__":
    tracker = FaceTracker()
    print("Tracker initialized successfully!")

