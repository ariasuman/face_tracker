
import cv2

from src.face_recognition import FaceRecognizer
from src.face_matcher import FaceMatcher
from src.database import Database

recognizer = FaceRecognizer()
matcher = FaceMatcher(threshold=0.45)
db = Database()

if not db.connect():
    exit()

# Load existing faces from MySQL
saved_faces = db.get_all_faces()
matcher.load_faces(saved_faces)

# Read test image
image = cv2.imread("data/face_1.png")

if image is None:
    print("Image not found!")
    db.close()
    exit()

# Generate face embedding
embedding = recognizer.get_embedding(image)

if embedding is None:
    print("No face detected!")
    db.close()
    exit()

# Match or register
face_id, similarity = matcher.match_face(embedding)

if face_id is None:
    face_id = matcher.register_face(embedding)

    if db.save_face(face_id, embedding):
        print("New visitor saved:", face_id)
    else:
        print("Database save failed!")
else:
    print("Returning visitor:", face_id)

print("Similarity:", round(similarity, 4))

db.close()
