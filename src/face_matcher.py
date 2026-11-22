
import numpy as np


class FaceMatcher:
    """
    Matches face embeddings against registered visitor identities.

    Each Face ID can contain multiple embeddings representing
    different viewing angles and appearances.

    Matching decisions:
        KNOWN          - confident identity match
        UNCERTAIN      - possible match, but not confident enough
        NEW_CANDIDATE  - no sufficiently similar identity found
    """

    def __init__(
        self,
        threshold=0.45,
        uncertain_threshold=0.30,
        minimum_margin=0.05,
        max_embeddings_per_face=6,
        duplicate_threshold=0.98
    ):

        self.threshold = threshold
        self.uncertain_threshold = uncertain_threshold

        # Difference required between the best and second-best
        # identity scores for a confident match.
        self.minimum_margin = minimum_margin

        self.max_embeddings_per_face = max_embeddings_per_face
        self.duplicate_threshold = duplicate_threshold

        # FACE_ID -> list of normalized embeddings
        self.known_faces = {}

        self.next_face_number = 1

    # --------------------------------------------------
    # EMBEDDING NORMALIZATION
    # --------------------------------------------------

    @staticmethod
    def normalize_embedding(embedding):
        """Convert an embedding into a normalized float32 vector."""

        if embedding is None:
            return None

        embedding = np.asarray(
            embedding,
            dtype=np.float32
        )

        # Face embeddings must be one-dimensional.
        if embedding.ndim != 1:
            return None

        if embedding.size == 0:
            return None

        if not np.all(np.isfinite(embedding)):
            return None

        norm = np.linalg.norm(embedding)

        if norm <= 1e-10:
            return None

        return embedding / norm

    # --------------------------------------------------
    # LOAD REGISTERED FACES
    # --------------------------------------------------

    def load_faces(self, known_faces):
        """
        Load embeddings from the database.

        Supports:
            FACE_001 -> one embedding
            FACE_002 -> list of embeddings
            FACE_003 -> NumPy array of embeddings
        """

        self.known_faces = {}

        for face_id, embeddings in known_faces.items():

            if embeddings is None:
                continue

            embeddings_array = np.asarray(
                embeddings,
                dtype=np.float32
            )

            # A single embedding has shape (512,).
            if embeddings_array.ndim == 1:
                embeddings = [embeddings_array]

            # Multiple embeddings have shape (N, 512).
            elif embeddings_array.ndim == 2:
                embeddings = list(embeddings_array)

            else:
                print(
                    "Invalid embedding format:",
                    face_id
                )
                continue

            valid_embeddings = []

            for embedding in embeddings:

                normalized = self.normalize_embedding(
                    embedding
                )

                if normalized is not None:
                    valid_embeddings.append(normalized)

            if valid_embeddings:
                self.known_faces[face_id] = valid_embeddings

        # Find the next available Face ID number.
        numbers = []

        for face_id in self.known_faces:

            try:
                number = int(face_id.split("_")[1])
                numbers.append(number)

            except (IndexError, ValueError):
                continue

        self.next_face_number = (
            max(numbers) + 1 if numbers else 1
        )

        total_embeddings = sum(
            len(embeddings)
            for embeddings in self.known_faces.values()
        )

        print("Loaded Face IDs:", len(self.known_faces))
        print("Loaded embeddings:", total_embeddings)

    # --------------------------------------------------
    # COSINE SIMILARITY
    # --------------------------------------------------

    @staticmethod
    def cosine_similarity(embedding1, embedding2):
        """Calculate cosine similarity between two embeddings."""

        first = FaceMatcher.normalize_embedding(
            embedding1
        )

        second = FaceMatcher.normalize_embedding(
            embedding2
        )

        if first is None or second is None:
            return -1.0

        if first.shape != second.shape:
            return -1.0

        return float(np.dot(first, second))

    # --------------------------------------------------
    # SCORE EACH IDENTITY
    # --------------------------------------------------

    def find_top_matches(self, embedding):
        """
        Calculate the best template similarity for each Face ID.

        Returns a list sorted from highest to lowest score:

            [
                (FACE_001, 0.82),
                (FACE_002, 0.31)
            ]
        """

        query = self.normalize_embedding(embedding)

        if query is None:
            return []

        identity_scores = []

        for face_id, stored_embeddings in self.known_faces.items():

            similarities = []

            for stored_embedding in stored_embeddings:

                if stored_embedding.shape != query.shape:
                    continue

                similarity = float(
                    np.dot(query, stored_embedding)
                )

                similarities.append(similarity)

            if similarities:

                # Keep the strongest template match for this identity.
                best_score = max(similarities)

                identity_scores.append(
                    (face_id, best_score)
                )

        identity_scores.sort(
            key=lambda item: item[1],
            reverse=True
        )

        return identity_scores

    # --------------------------------------------------
    # FIND BEST MATCH
    # --------------------------------------------------

    def find_best_match(self, embedding):
        """
        Backward-compatible method.

        Returns:
            (best_face_id, best_similarity)
        """

        matches = self.find_top_matches(embedding)

        if not matches:
            return None, 0.0

        return matches[0]

    # --------------------------------------------------
    # CLASSIFY MATCH
    # --------------------------------------------------

    def classify_match(self, embedding):
        """
        Return:

            (status, face_id, similarity)

        A confident match requires:
            1. Similarity >= threshold
            2. Best score sufficiently separated from
               the second-best identity score.

        The margin is provisional and must be calibrated
        using labelled real-video samples.
        """

        query = self.normalize_embedding(embedding)

        if query is None:
            return "UNCERTAIN", None, 0.0

        if not self.known_faces:
            return "NEW_CANDIDATE", None, 0.0

        matches = self.find_top_matches(query)

        if not matches:
            return "NEW_CANDIDATE", None, 0.0

        best_face_id, best_similarity = matches[0]

        # If only one identity exists, there is no second
        # identity to compare against.
        if len(matches) >= 2:
            second_similarity = matches[1][1]
        else:
            second_similarity = -1.0

        margin = best_similarity - second_similarity

        # First check the similarity score.
        if best_similarity < self.uncertain_threshold:

            return (
                "NEW_CANDIDATE",
                None,
                best_similarity
            )

        # Similarity is reasonable, but not high enough.
        if best_similarity < self.threshold:

            return (
                "UNCERTAIN",
                best_face_id,
                best_similarity
            )

        # The two strongest identities are too close.
        if margin < self.minimum_margin:

            return (
                "UNCERTAIN",
                best_face_id,
                best_similarity
            )

        return (
            "KNOWN",
            best_face_id,
            best_similarity
        )

    # --------------------------------------------------
    # BACKWARD-COMPATIBLE MATCHING
    # --------------------------------------------------

    def match_face(self, embedding):
        """
        Return:
            (Face ID, similarity)

        Only confident matches return a Face ID.
        """

        status, face_id, similarity = (
            self.classify_match(embedding)
        )

        if status == "KNOWN":
            return face_id, similarity

        return None, similarity

    # --------------------------------------------------
    # REGISTER NEW VISITOR
    # --------------------------------------------------

    def register_face(self, embedding):
        """Register a new visitor with the first embedding."""

        normalized = self.normalize_embedding(
            embedding
        )

        if normalized is None:
            return None

        face_id = f"FACE_{self.next_face_number:03d}"

        self.known_faces[face_id] = [
            normalized.copy()
        ]

        self.next_face_number += 1

        return face_id

    # --------------------------------------------------
    # ADD ANOTHER EMBEDDING
    # --------------------------------------------------

    def add_embedding(self, face_id, embedding):
        """
        Add another embedding to an existing visitor.

        Duplicate embeddings are rejected.

        A maximum number of templates is maintained
        to prevent unlimited database growth.
        """

        if face_id not in self.known_faces:
            return False

        normalized = self.normalize_embedding(
            embedding
        )

        if normalized is None:
            return False

        stored_embeddings = self.known_faces[face_id]

        # All embeddings must have matching dimensions.
        if stored_embeddings:

            if stored_embeddings[0].shape != normalized.shape:
                return False

        # Reject almost identical templates.
        for stored_embedding in stored_embeddings:

            similarity = self.cosine_similarity(
                normalized,
                stored_embedding
            )

            if similarity >= self.duplicate_threshold:
                print(
                    "Duplicate embedding rejected:",
                    face_id,
                    f"| Similarity: {similarity:.4f}"
                )
                return False

        # Avoid unlimited template growth.
        if len(stored_embeddings) >= self.max_embeddings_per_face:

            print(
                "Maximum embedding limit reached:",
                face_id
            )
            return False

        stored_embeddings.append(
            normalized.copy()
        )

        return True


if __name__ == "__main__":

    matcher = FaceMatcher()

    print("Face Matcher initialized successfully!")
    print("Matching threshold:", matcher.threshold)
    print("Uncertain threshold:", matcher.uncertain_threshold)
    print("Minimum margin:", matcher.minimum_margin)
    print("Maximum templates per visitor:", matcher.max_embeddings_per_face)
