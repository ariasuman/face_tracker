
import os
import mysql.connector
import numpy as np
from dotenv import load_dotenv

load_dotenv()


class Database:

    def __init__(self):
        self.connection = None

    def connect(self):
        try:
            self.connection = mysql.connector.connect(
                host=os.getenv("DB_HOST"),
                user=os.getenv("DB_USER"),
                password=os.getenv("DB_PASSWORD"),
                database=os.getenv("DB_NAME")
            )

            print("MySQL database connected successfully!")
            return True

        except mysql.connector.Error as error:
            print("MySQL connection failed:", error)
            return False

    def save_face(self, face_id, embedding):
        try:
            cursor = self.connection.cursor()

            embedding_bytes = np.asarray(
                embedding, dtype=np.float32
            ).tobytes()

            query = """
                INSERT INTO faces (face_id, embedding)
                VALUES (%s, %s)
            """

            cursor.execute(query, (face_id, embedding_bytes))
            self.connection.commit()
            cursor.close()

            print("Face saved:", face_id)
            return True

        except mysql.connector.Error as error:
            self.connection.rollback()
            print("Face saving failed:", error)
            return False

    def get_all_faces(self):
        try:
            cursor = self.connection.cursor()

            cursor.execute(
                "SELECT face_id, embedding FROM faces"
            )

            rows = cursor.fetchall()
            cursor.close()

            known_faces = {}

            for face_id, embedding_bytes in rows:
                embedding = np.frombuffer(
                    embedding_bytes, dtype=np.float32
                ).copy()

                known_faces[face_id] = embedding

            return known_faces

        except mysql.connector.Error as error:
            print("Face retrieval failed:", error)
            return {}

    def close(self):
        if self.connection and self.connection.is_connected():
            self.connection.close()
            print("MySQL connection closed.")


if __name__ == "__main__":
    db = Database()

    if db.connect():
        faces = db.get_all_faces()
        print("Previously registered faces:", len(faces))
        db.close()
