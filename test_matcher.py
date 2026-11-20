
import numpy as np
from src.face_matcher import FaceMatcher

matcher = FaceMatcher(threshold=0.45)

# Create first sample embedding
embedding1 = np.random.rand(512).astype(np.float32)
embedding1 /= np.linalg.norm(embedding1)

# Register first face
face_id = matcher.register_face(embedding1)
print("Registered Face ID:", face_id)

# Match the same face
matched_id, similarity = matcher.match_face(embedding1)

print("Matched Face ID:", matched_id)
print("Similarity:", round(similarity, 4))

# Test a different random embedding
embedding2 = np.random.randn(512).astype(np.float32)
embedding2 /= np.linalg.norm(embedding2)

matched_id, similarity = matcher.match_face(embedding2)

print("Second face match:", matched_id)
print("Second face similarity:", round(similarity, 4))
