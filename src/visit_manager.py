
from src.database import Database


class VisitManager:

    def __init__(self, database):
        self.db = database

    def record_entry(self, face_id):

        cursor = self.db.connection.cursor()

        try:
            # Check whether this visitor already has an active visit
            cursor.execute(
                """
                SELECT visit_id
                FROM visits
                WHERE face_id = %s AND status = 'ACTIVE'
                """,
                (face_id,)
            )

            existing_visit = cursor.fetchone()

            if existing_visit:
                print("Visitor already inside:", face_id)
                return existing_visit[0]

            # Create a new visit
            cursor.execute(
                """
                INSERT INTO visits (face_id, entry_time, status)
                VALUES (%s, NOW(), 'ACTIVE')
                """,
                (face_id,)
            )

            visit_id = cursor.lastrowid

            # Record ENTRY event
            cursor.execute(
                """
                INSERT INTO events
                (visit_id, face_id, event_type, event_time)
                VALUES (%s, %s, 'ENTRY', NOW())
                """,
                (visit_id, face_id)
            )

            self.db.connection.commit()

            print("ENTRY recorded:", face_id)
            print("Visit ID:", visit_id)

            return visit_id

        except Exception as error:
            self.db.connection.rollback()
            print("Entry recording failed:", error)
            return None

        finally:
            cursor.close()

    def record_exit(self, face_id):

        cursor = self.db.connection.cursor()

        try:
            # Find the active visit
            cursor.execute(
                """
                SELECT visit_id
                FROM visits
                WHERE face_id = %s AND status = 'ACTIVE'
                ORDER BY visit_id DESC
                LIMIT 1
                """,
                (face_id,)
            )

            visit = cursor.fetchone()

            if visit is None:
                print("No active visit found:", face_id)
                return False

            visit_id = visit[0]

            # Complete the visit
            cursor.execute(
                """
                UPDATE visits
                SET exit_time = NOW(),
                    status = 'COMPLETED'
                WHERE visit_id = %s AND status = 'ACTIVE'
                """,
                (visit_id,)
            )

            if cursor.rowcount != 1:
                self.db.connection.rollback()
                return False

            # Record EXIT event
            cursor.execute(
                """
                INSERT INTO events
                (visit_id, face_id, event_type, event_time)
                VALUES (%s, %s, 'EXIT', NOW())
                """,
                (visit_id, face_id)
            )

            self.db.connection.commit()

            print("EXIT recorded:", face_id)
            print("Visit ID:", visit_id)

            return True

        except Exception as error:
            self.db.connection.rollback()
            print("Exit recording failed:", error)
            return False

        finally:
            cursor.close()
