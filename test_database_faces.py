
from src.database import Database
from src.face_matcher import FaceMatcher

# Create database and matcher objects
db = Database()
matcher = FaceMatcher(threshold=0.45)

# Connect to MySQL
if db.connect():

    # Load saved faces from MySQL
    saved_faces = db.get_all_faces()

    # Load them into FaceMatcher
    matcher.load_faces(saved_faces)

    print("Faces loaded from MySQL:", len(matcher.known_faces))

    db.close()
