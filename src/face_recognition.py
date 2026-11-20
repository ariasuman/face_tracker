
from insightface.app import FaceAnalysis
import numpy as np


class FaceRecognizer:

    def __init__(self):
        print("Loading InsightFace model...")

        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"]
        )

        self.app.prepare(ctx_id=-1, det_size=(640, 640))

        print("InsightFace model loaded successfully!")

    def get_embedding(self, face_image):

        faces = self.app.get(face_image)

        if len(faces) == 0:
            return None

        # Get the 512-dimensional ArcFace embedding
        embedding = faces[0].normed_embedding

        return embedding


if __name__ == "__main__":
    recognizer = FaceRecognizer()
    print("Face recognizer initialized successfully!")
