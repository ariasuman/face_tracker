
import cv2
import numpy as np

from src.face_recognition import FaceRecognizer

# Initialize the recognizer
recognizer = FaceRecognizer()

# Read the test image
image = cv2.imread("data/test_face.png")
if image is None:
    print("Error: Could not read test image!")
    exit()

# Generate ArcFace embedding
embedding = recognizer.get_embedding(image)

if embedding is not None:
    print("Face detected successfully!")
    print("Embedding shape:", embedding.shape)
    print("Embedding dimensions:", len(embedding))
    print("First 5 values:", embedding[:5])
else:
    print("No face detected in the image.")
