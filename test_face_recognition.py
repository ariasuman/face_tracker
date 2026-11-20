
import cv2

from src.face_recognition import FaceRecognizer
from src.face_matcher import FaceMatcher

recognizer = FaceRecognizer()
matcher = FaceMatcher(threshold=0.45)

image = cv2.imread("data/test_face.png")

if image is None:
    print("Error: Image not found!")
    exit()

# Simulate the same person appearing twice
for appearance in range(1, 3):

    print(f"\nAppearance {appearance}")

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
