
import os
from datetime import datetime
import mysql.connector
import numpy as np
from dotenv import load_dotenv

load_dotenv()


class Database:

    def __init__(self):
        self.connection = None

    # --------------------------------------------------
    # DATABASE CONNECTION
    # --------------------------------------------------

    def connect(self):

        try:
            self.connection = mysql.connector.connect(
                host=os.getenv("DB_HOST"),
                user=os.getenv("DB_USER"),
                password=os.getenv("DB_PASSWORD"),
                database=os.getenv("DB_NAME")
            )

            print("MySQL database connected successfully!")

            self.migrate_existing_embeddings()

            return True

        except mysql.connector.Error as error:

            print("MySQL connection failed:", error)
            return False

    # --------------------------------------------------
    # TRANSACTION HYGIENE
    # --------------------------------------------------

    def _end_read(self):
        """
        mysql.connector runs with autocommit OFF, so a plain SELECT leaves an
        implicit transaction open. A later start_transaction() then fails with
        "Transaction already in progress" and the ENTRY/EXIT is lost.
        Always close it after read-only queries.
        """
        try:
            if self.connection and self.connection.in_transaction:
                self.connection.rollback()
        except mysql.connector.Error:
            pass

    # --------------------------------------------------
    # EMBEDDING MIGRATION
    # --------------------------------------------------

    def migrate_existing_embeddings(self):

        cursor = self.connection.cursor()

        try:

            query = """
                INSERT INTO face_embeddings (face_id, embedding)
                SELECT f.face_id, f.embedding
                FROM faces f
                WHERE NOT EXISTS (
                    SELECT 1
                    FROM face_embeddings e
                    WHERE e.face_id = f.face_id
                )
            """

            cursor.execute(query)
            self.connection.commit()

            if cursor.rowcount > 0:
                print(
                    "Existing embeddings migrated:",
                    cursor.rowcount
                )
            else:
                print("No existing embeddings to migrate.")

        except mysql.connector.Error as error:

            self.connection.rollback()

            print(
                "Embedding migration failed:",
                error
            )

        finally:
            cursor.close()

    # --------------------------------------------------
    # FACE REGISTRATION
    # --------------------------------------------------

    def save_face(self, face_id, embedding):

        cursor = self.connection.cursor()

        try:

            embedding_bytes = np.asarray(
                embedding,
                dtype=np.float32
            ).tobytes()

            cursor.execute(
                """
                INSERT INTO faces (face_id, embedding)
                VALUES (%s, %s)
                """,
                (face_id, embedding_bytes)
            )

            cursor.execute(
                """
                INSERT INTO face_embeddings (face_id, embedding)
                VALUES (%s, %s)
                """,
                (face_id, embedding_bytes)
            )

            self.connection.commit()

            print(
                "Face and first embedding saved:",
                face_id
            )

            return True

        except mysql.connector.Error as error:

            self.connection.rollback()

            print("Face saving failed:", error)

            return False

        finally:
            cursor.close()

    # --------------------------------------------------
    # ADDITIONAL EMBEDDING
    # --------------------------------------------------

    def add_embedding(self, face_id, embedding):

        cursor = self.connection.cursor()

        try:

            embedding_bytes = np.asarray(
                embedding,
                dtype=np.float32
            ).tobytes()

            cursor.execute(
                """
                INSERT INTO face_embeddings (face_id, embedding)
                VALUES (%s, %s)
                """,
                (face_id, embedding_bytes)
            )

            self.connection.commit()

            print(
                "Additional embedding saved:",
                face_id
            )

            return True

        except mysql.connector.Error as error:

            self.connection.rollback()

            print(
                "Embedding saving failed:",
                error
            )

            return False

        finally:
            cursor.close()

    # --------------------------------------------------
    # GET ALL EMBEDDINGS
    # --------------------------------------------------

    def get_all_embeddings(self):

        try:

            cursor = self.connection.cursor()

            cursor.execute(
                """
                SELECT face_id, embedding
                FROM face_embeddings
                ORDER BY embedding_id
                """
            )

            rows = cursor.fetchall()
            cursor.close()
            self._end_read()

            known_faces = {}

            for face_id, embedding_bytes in rows:

                embedding = np.frombuffer(
                    embedding_bytes,
                    dtype=np.float32
                ).copy()

                if face_id not in known_faces:
                    known_faces[face_id] = []

                known_faces[face_id].append(
                    embedding
                )

            return known_faces

        except mysql.connector.Error as error:

            print(
                "Embedding retrieval failed:",
                error
            )

            return {}

    # --------------------------------------------------
    # BACKWARD-COMPATIBLE FACE RETRIEVAL
    # --------------------------------------------------

    def get_all_faces(self):

        all_embeddings = self.get_all_embeddings()

        return {
            face_id: embeddings[0]
            for face_id, embeddings in all_embeddings.items()
            if embeddings
        }

    # --------------------------------------------------
    # GET ACTIVE VISIT
    # --------------------------------------------------

    def get_active_visit(self, face_id):

        cursor = self.connection.cursor()

        try:

            cursor.execute(
                """
                SELECT visit_id
                FROM visits
                WHERE face_id = %s
                AND status = 'ACTIVE'
                ORDER BY visit_id DESC
                LIMIT 1
                """,
                (face_id,)
            )

            result = cursor.fetchone()

            if result:
                return result[0]

            return None

        except mysql.connector.Error as error:

            print(
                "Active visit retrieval failed:",
                error
            )

            return None

        finally:
            cursor.close()
            self._end_read()

    # --------------------------------------------------
    # REPORTING HELPERS
    # --------------------------------------------------

    def get_all_active_visits(self):
        """Return [(visit_id, face_id, entry_time)] for visits still ACTIVE."""

        cursor = self.connection.cursor()

        try:
            cursor.execute(
                """
                SELECT visit_id, face_id, entry_time
                FROM visits
                WHERE status = 'ACTIVE'
                ORDER BY visit_id
                """
            )
            return cursor.fetchall()

        except mysql.connector.Error as error:
            print("Active visits retrieval failed:", error)
            return []

        finally:
            cursor.close()
            self._end_read()

    def get_unique_visitor_count(self):
        """Number of distinct registered faces (unique visitors)."""

        cursor = self.connection.cursor()

        try:
            cursor.execute("SELECT COUNT(*) FROM faces")
            return int(cursor.fetchone()[0])

        except mysql.connector.Error as error:
            print("Unique count retrieval failed:", error)
            return None

        finally:
            cursor.close()
            self._end_read()

    # --------------------------------------------------
    # RECORD VISITOR ENTRY
    # --------------------------------------------------

    def start_visit(self, face_id, image_path=None, event_time=None):
        """
        Create a new visit and ENTRY event.

        If the visitor already has an active visit,
        return that visit instead of creating a duplicate.

        Returns:
            (visit_id, created)

            event_time: real crossing time (datetime). Defaults to now.
                        Needed so delayed identification does not shift
                        the recorded ENTRY time.

            created=True  -> New visit created.
            created=False -> Existing active visit returned.
            visit_id=None -> Database operation failed.
        """

        event_time = event_time or datetime.now()

        cursor = self.connection.cursor()

        try:

            self._end_read()
            self.connection.start_transaction()

            # Lock the registered face row to serialize
            # simultaneous visit creation for this identity.
            cursor.execute(
                """
                SELECT face_id
                FROM faces
                WHERE face_id = %s
                FOR UPDATE
                """,
                (face_id,)
            )

            if cursor.fetchone() is None:

                self.connection.rollback()

                print(
                    "Cannot create visit: unknown Face ID",
                    face_id
                )

                return None, False

            # Check for an existing active visit.
            cursor.execute(
                """
                SELECT visit_id
                FROM visits
                WHERE face_id = %s
                AND status = 'ACTIVE'
                ORDER BY visit_id DESC
                LIMIT 1
                FOR UPDATE
                """,
                (face_id,)
            )

            existing = cursor.fetchone()

            if existing:

                self.connection.commit()

                return existing[0], False

            # Create a new visit.
            cursor.execute(
                """
                INSERT INTO visits (
                    face_id,
                    entry_time,
                    status
                )
                VALUES (
                    %s,
                    %s,
                    'ACTIVE'
                )
                """,
                (face_id, event_time)
            )

            visit_id = cursor.lastrowid

            # Create the corresponding ENTRY event.
            cursor.execute(
                """
                INSERT INTO events (
                    visit_id,
                    face_id,
                    event_type,
                    event_time,
                    image_path
                )
                VALUES (
                    %s,
                    %s,
                    'ENTRY',
                    %s,
                    %s
                )
                """,
                (
                    visit_id,
                    face_id,
                    event_time,
                    image_path
                )
            )

            self.connection.commit()

            print(
                f"ENTRY RECORDED: {face_id} "
                f"| Visit ID: {visit_id}"
            )

            return visit_id, True

        except mysql.connector.Error as error:

            self.connection.rollback()

            print(
                "Visitor entry recording failed:",
                error
            )

            return None, False

        finally:
            cursor.close()

    # --------------------------------------------------
    # RECORD VISITOR EXIT
    # --------------------------------------------------

    def complete_visit(self, face_id, image_path=None, event_time=None):
        """
        Complete the active visit and create one EXIT event.

        Returns:
            True  -> Visit completed.
            False -> No active visit or database failure.
        """

        event_time = event_time or datetime.now()

        cursor = self.connection.cursor()

        try:

            self._end_read()
            self.connection.start_transaction()

            # Lock the face row before changing its visit.
            cursor.execute(
                """
                SELECT face_id
                FROM faces
                WHERE face_id = %s
                FOR UPDATE
                """,
                (face_id,)
            )

            if cursor.fetchone() is None:

                self.connection.rollback()

                print(
                    "Cannot complete visit: unknown Face ID",
                    face_id
                )

                return False

            # Find and lock the active visit.
            cursor.execute(
                """
                SELECT visit_id, entry_time
                FROM visits
                WHERE face_id = %s
                AND status = 'ACTIVE'
                ORDER BY visit_id DESC
                LIMIT 1
                FOR UPDATE
                """,
                (face_id,)
            )

            result = cursor.fetchone()

            if result is None:

                self.connection.rollback()

                print(
                    "No active visit to complete:",
                    face_id
                )

                return False

            visit_id = result[0]

            # An EXIT can never be earlier than its ENTRY.
            if result[1] is not None and event_time < result[1]:
                event_time = result[1]

            # Mark the visit completed.
            cursor.execute(
                """
                UPDATE visits
                SET
                    exit_time = %s,
                    status = 'COMPLETED'
                WHERE visit_id = %s
                AND status = 'ACTIVE'
                """,
                (event_time, visit_id)
            )

            if cursor.rowcount != 1:

                self.connection.rollback()

                print(
                    "Visit was not updated:",
                    visit_id
                )

                return False

            # Record exactly one EXIT event
            # within the same database transaction.
            cursor.execute(
                """
                INSERT INTO events (
                    visit_id,
                    face_id,
                    event_type,
                    event_time,
                    image_path
                )
                VALUES (
                    %s,
                    %s,
                    'EXIT',
                    %s,
                    %s
                )
                """,
                (
                    visit_id,
                    face_id,
                    event_time,
                    image_path
                )
            )

            self.connection.commit()

            print(
                f"EXIT RECORDED: {face_id} "
                f"| Visit ID: {visit_id}"
            )

            return True

        except mysql.connector.Error as error:

            self.connection.rollback()

            print(
                "Visitor exit recording failed:",
                error
            )

            return False

        finally:
            cursor.close()

    # --------------------------------------------------
    # CLOSE DATABASE
    # --------------------------------------------------

    def close(self):

        if (
            self.connection
            and self.connection.is_connected()
        ):

            self.connection.close()

            print("MySQL connection closed.")


# --------------------------------------------------
# TEST DATABASE CONNECTION
# --------------------------------------------------

if __name__ == "__main__":

    db = Database()

    if db.connect():

        known_faces = db.get_all_embeddings()

        print("\nRegistered Face IDs:")

        for face_id, embeddings in known_faces.items():

            print(
                face_id,
                "| Number of embeddings:",
                len(embeddings)
            )

        db.close()
