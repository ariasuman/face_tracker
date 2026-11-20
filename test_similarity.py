
import cv2
import numpy as np
from src.face_recognition import FaceRecognizer

# Load our recognition model
recognizer = FaceRecognizer()

# Read the original test image
image1 = cv2.imread("data/test_face.png")

if image1 is None:
    print("Error: Test image not found!")
    exit()

# Create a slightly brighter version of the same image
image2 = cv2.convertScaleAbs(image1, alpha=1.1, beta=10)

# Generate embeddings
embedding1 = recognizer.get_embedding(image1)
embedding2 = recognizer.get_embedding(image2)

if embedding1 is None or embedding2 is None:
    print("Face detection failed in one of the images.")
    exit()

# Calculate cosine similarity
similarity = np.dot(embedding1, embedding2) / (
    np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
)

print("Embedding 1 dimensions:", len(embedding1))
print("Embedding 2 dimensions:", len(embedding2))
print("Cosine similarity:", round(float(similarity), 4))
