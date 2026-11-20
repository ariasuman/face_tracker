
import cv2

from src.face_recognition import FaceRecognizer
from src.face_matcher import FaceMatcher

recognizer = FaceRecognizer()
matcher = FaceMatcher(threshold=0.45)

image_paths = [
    "data/face_1.png",
    "data/face_2.png",
    "data/face_3.png"
]

for image_path in image_paths:
    print(f"\nProcessing: {image_path}")

    image = cv2.imread(image_path)

    if image is None:
        print("Error: Could not load image")
        continue

    embedding = recognizer.get_embedding(image)

    if embedding is None:
        print("No face detected!")
        continue

    face_id, similarity = matcher.match_face(embedding)

    if face_id is None:
        face_id = matcher.register_face(embedding)
        print("New visitor registered:", face_id)
    else:
        print("Returning visitor recognized:", face_id)

    print("Similarity:", round(similarity, 4))
