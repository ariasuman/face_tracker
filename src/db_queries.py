"""Read-only MySQL queries used by streamlit_app.py.

Uses the same DB_HOST / DB_USER / DB_PASSWORD / DB_NAME values from .env as
src/database.py. Nothing here writes to the database.
"""

import os
from collections import Counter
from contextlib import contextmanager

import mysql.connector
from dotenv import load_dotenv

load_dotenv()


@contextmanager
def _cursor():
    connection = mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
        connection_timeout=3,
        # The C extension crashes with an access violation when it is used
        # from Streamlit's script thread on Windows. Pure Python is safe.
        use_pure=True,
    )
    try:
        cursor = connection.cursor()
        try:
            yield cursor
        finally:
            cursor.close()
    finally:
        connection.close()


def _int(value):
    return int(value) if value is not None else 0


def _scalar(cursor, sql):
    cursor.execute(sql)
    return _int(cursor.fetchone()[0])


# ---------------------------------------------------------------------------
# Dashboard queries
# ---------------------------------------------------------------------------
def get_kpis():
    """Return the four numbers shown on the Overview page."""
    with _cursor() as cursor:
        unique_visitors = _scalar(cursor, "SELECT COUNT(*) FROM faces")
        active_visits = _scalar(
            cursor, "SELECT COUNT(*) FROM visits WHERE status = 'ACTIVE'"
        )
        entries_today = _scalar(
            cursor,
            "SELECT COUNT(*) FROM events "
            "WHERE event_type = 'ENTRY' AND DATE(event_time) = CURDATE()",
        )
        exits_today = _scalar(
            cursor,
            "SELECT COUNT(*) FROM events "
            "WHERE event_type = 'EXIT' AND DATE(event_time) = CURDATE()",
        )

    return {
        "unique_visitors": unique_visitors,
        "active_visits": active_visits,
        "entries_today": entries_today,
        "exits_today": exits_today,
    }


def get_daily_counts(days=30):
    """Return [(date, entries, exits), ...] for the last `days` days."""
    with _cursor() as cursor:
        cursor.execute(
            """
            SELECT
                DATE(event_time) AS day,
                SUM(event_type = 'ENTRY') AS entries,
                SUM(event_type = 'EXIT') AS exits
            FROM events
            WHERE event_time >= (CURDATE() - INTERVAL %s DAY)
            GROUP BY DATE(event_time)
            ORDER BY day
            """,
            (int(days),),
        )
        return [(day, _int(entries), _int(exits)) for day, entries, exits in cursor.fetchall()]


def get_visits(limit=1000):
    """Return [(visit_id, face_id, entry_time, exit_time, status), ...]."""
    with _cursor() as cursor:
        cursor.execute(
            """
            SELECT visit_id, face_id, entry_time, exit_time, status
            FROM visits
            ORDER BY visit_id DESC
            LIMIT %s
            """,
            (int(limit),),
        )
        return cursor.fetchall()


def get_recent_events(limit=500):
    """Return [(event_id, visit_id, face_id, type, time, image), ...].

    event_id is a running number for display, so this does not depend on the
    events table's own key column name.
    """
    with _cursor() as cursor:
        cursor.execute(
            """
            SELECT visit_id, face_id, event_type, event_time, image_path
            FROM events
            ORDER BY event_time DESC
            LIMIT %s
            """,
            (int(limit),),
        )
        rows = cursor.fetchall()

    return [(index,) + tuple(row) for index, row in enumerate(rows, start=1)]


# ---------------------------------------------------------------------------
# Verification queries
# ---------------------------------------------------------------------------
def get_faces():
    """Return [(face_id, number_of_embeddings), ...] for every registered face."""
    with _cursor() as cursor:
        cursor.execute(
            """
            SELECT f.face_id, COUNT(e.face_id) AS embeddings
            FROM faces f
            LEFT JOIN face_embeddings e ON e.face_id = f.face_id
            GROUP BY f.face_id
            ORDER BY f.face_id
            """
        )
        return [(face_id, _int(count)) for face_id, count in cursor.fetchall()]


def get_integrity_summary():
    """Counts used to check that visits, events and faces agree."""
    with _cursor() as cursor:
        return {
            "faces": _scalar(cursor, "SELECT COUNT(*) FROM faces"),
            "faces_with_visits": _scalar(
                cursor, "SELECT COUNT(DISTINCT face_id) FROM visits"
            ),
            "visits_total": _scalar(cursor, "SELECT COUNT(*) FROM visits"),
            "visits_active": _scalar(
                cursor, "SELECT COUNT(*) FROM visits WHERE status = 'ACTIVE'"
            ),
            "visits_completed": _scalar(
                cursor, "SELECT COUNT(*) FROM visits WHERE status = 'COMPLETED'"
            ),
            "entry_events": _scalar(
                cursor, "SELECT COUNT(*) FROM events WHERE event_type = 'ENTRY'"
            ),
            "exit_events": _scalar(
                cursor, "SELECT COUNT(*) FROM events WHERE event_type = 'EXIT'"
            ),
        }


def find_similar_pairs(rows, threshold=0.35, limit=50):
    """Pure helper: rows are [(face_id, embedding_bytes), ...].

    Averages each face's stored embeddings, then returns
    [(face_a, face_b, cosine_similarity), ...] for pairs at or above
    `threshold`, most similar first. A high score between two different
    Face IDs is a hint that one person may have been registered twice.
    """
    import numpy as np

    grouped = {}
    for face_id, blob in rows:
        if blob is None:
            continue
        vector = np.frombuffer(blob, dtype=np.float32)
        if vector.size == 0:
            continue
        norm = float(np.linalg.norm(vector))
        if norm == 0:
            continue
        grouped.setdefault(face_id, []).append(vector / norm)

    if len(grouped) < 2:
        return []

    # Keep only vectors of the most common length, to be safe.
    sizes = Counter(vec.size for vecs in grouped.values() for vec in vecs)
    size = sizes.most_common(1)[0][0]

    face_ids = []
    centroids = []
    for face_id in sorted(grouped):
        vecs = [v for v in grouped[face_id] if v.size == size]
        if not vecs:
            continue
        centroid = np.mean(vecs, axis=0)
        norm = float(np.linalg.norm(centroid))
        if norm == 0:
            continue
        face_ids.append(face_id)
        centroids.append(centroid / norm)

    if len(face_ids) < 2:
        return []

    matrix = np.stack(centroids)
    similarity = matrix @ matrix.T
    first, second = np.triu_indices(len(face_ids), k=1)
    scores = similarity[first, second]

    pairs = [
        (face_ids[i], face_ids[j], float(score))
        for i, j, score in zip(first, second, scores)
        if score >= threshold
    ]
    pairs.sort(key=lambda item: item[2], reverse=True)
    return pairs[: int(limit)]


def get_similar_face_pairs(threshold=0.35, limit=50):
    """Read every stored embedding and return the most similar Face ID pairs."""
    with _cursor() as cursor:
        cursor.execute("SELECT face_id, embedding FROM face_embeddings")
        rows = cursor.fetchall()
    return find_similar_pairs(rows, threshold, limit)