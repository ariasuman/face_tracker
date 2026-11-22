
from pathlib import Path
from datetime import datetime
import cv2


class ImageStorage:

    def __init__(self):
        self.base_directory = Path("data/faces")
        self.base_directory.mkdir(parents=True, exist_ok=True)

    def save_face_image(self, face_id, face_image, event_type):

        event_type = event_type.lower()

        if event_type not in ["entry", "exit"]:
            raise ValueError("Event type must be entry or exit")

        # Create a separate folder for each visitor and event
        save_directory = self.base_directory / face_id / event_type
        save_directory.mkdir(parents=True, exist_ok=True)

        # Generate a unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        image_path = save_directory / f"{timestamp}.jpg"

        # Save the cropped face image
        success = cv2.imwrite(str(image_path), face_image)

        if success:
            print(f"{event_type.upper()} image saved:", image_path)
            return str(image_path)

        print("Failed to save face image")
        return None
