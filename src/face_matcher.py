
import numpy as np


class FaceMatcher:

    def __init__(self, threshold=0.45):
        self.threshold = threshold
        self.known_faces = {}
        self.next_face_number = 1

    # Load previously registered faces from MySQL
    def load_faces(self, known_faces):
        self.known_faces = known_faces

        numbers = []

        for face_id in self.known_faces:
            try:
                number = int(face_id.split("_")[1])
                numbers.append(number)
            except (IndexError, ValueError):
                continue

        if numbers:
            self.next_face_number = max(numbers) + 1

        print("Loaded faces:", len(self.known_faces))

    def cosine_similarity(self, embedding1, embedding2):

        embedding1 = np.asarray(embedding1)
        embedding2 = np.asarray(embedding2)

        denominator = (
            np.linalg.norm(embedding1) *
            np.linalg.norm(embedding2)
        )

        if denominator == 0:
            return 0.0

        return float(
            np.dot(embedding1, embedding2) / denominator
        )

    def match_face(self, embedding):

        if embedding is None:
            return None, 0.0

        best_face_id = None
        best_similarity = -1.0

        for face_id, stored_embedding in self.known_faces.items():

            similarity = self.cosine_similarity(
                embedding,
                stored_embedding
            )

            if similarity > best_similarity:
                best_similarity = similarity
                best_face_id = face_id

        if best_similarity >= self.threshold:
            return best_face_id, best_similarity

        return None, best_similarity

    def register_face(self, embedding):

        face_id = f"FACE_{self.next_face_number:03d}"

        self.known_faces[face_id] = np.asarray(
            embedding, dtype=np.float32
        ).copy()

        self.next_face_number += 1

        return face_id
