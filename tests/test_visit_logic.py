
import sys
import types
import os
import shutil
import glob
import tempfile
from datetime import datetime, timedelta

import numpy as np

# Stub packages not required for these logic tests
for name in ("mysql", "mysql.connector", "dotenv"):
    sys.modules[name] = types.ModuleType(name)

sys.modules["mysql.connector"].Error = Exception
sys.modules["mysql"].connector = sys.modules["mysql.connector"]
sys.modules["dotenv"].load_dotenv = lambda *a, **k: None

for name, cls in (
    ("src.tracker", "FaceTracker"),
    ("src.face_recognition", "FaceRecognizer")
):
    m = types.ModuleType(name)
    setattr(m, cls, object)
    sys.modules[name] = m

# Set project path and use a temporary folder for test logs
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(tempfile.mkdtemp(prefix="facetracker_test_"))

from src.video_processor import VideoProcessor
from src.boundary_counter import BoundaryCounter
from src.event_logger import EventLogger


class FakeDB:
    """Mimics database visit rules; can be told to fail."""

    def __init__(self):
        self.active = {}
        self.rows = []
        self.fail = False
        self.preactive = set()

    def start_visit(self, f, p=None, t=None):
        if self.fail:
            return None, False

        if f in self.active or f in self.preactive:
            return 99, False

        self.active[f] = t
        self.rows.append(("ENTRY", f, t, p))
        return len(self.rows), True

    def complete_visit(self, f, p=None, t=None):
        if self.fail or (f not in self.active and f not in self.preactive):
            return False

        self.active.pop(f, None)
        self.preactive.discard(f)
        self.rows.append(("EXIT", f, t, p))
        return True

    def get_unique_visitor_count(self):
        return 3

    def get_all_active_visits(self):
        return []


def make(dry=False, db=None):
    import logging

    lg = logging.getLogger("FaceTrackerEvents")

    for h in list(lg.handlers):
        h.close()
        lg.removeHandler(h)

    shutil.rmtree("logs", ignore_errors=True)

    vp = VideoProcessor.__new__(VideoProcessor)

    vp.dry_run = dry
    vp.database = db or FakeDB()
    vp.event_logger = EventLogger("logs")
    vp.boundary_counter = BoundaryCounter()

    # Tracking and identity information
    vp.track_to_face = {}
    vp.track_to_group = {}
    vp.track_stats = {}
    vp.track_reject_reasons = {}
    vp.pending_groups = {}

    # Visit and session information
    vp.session_visits = {}
    vp.pending_crossings = {}
    vp.track_last_crop = {}
    vp.session_faces = set()

    # Embedding information
    vp.last_embedding_frame = {}
    vp.last_template_frame = {}

    # Configuration
    vp.duplicate_track_window = 15
    vp.close_visits_on_end = True

    # Event statistics
    vp.stats = {
        "entries": 0,
        "exits": 0,
        "suppressed": 0,
        "unidentified": 0
    }

    return vp


W, H = 1000, 1000

IN = (500, 500)
OUT = (10, 500)
GREY = (60, 500)

crop = lambda: np.full((80, 80, 3), 120, np.uint8)

T0 = datetime(2026, 11, 22, 9, 0, 0)


def step(vp, tid, pos, f, fid=None, dt=None):
    now = T0 + timedelta(seconds=f)

    ev = vp.boundary_counter.update(
        tid, pos[0], pos[1], W, H, f
    )

    if fid:
        vp.track_to_face[tid] = fid

    vp.process_track_event(
        tid,
        vp.track_to_face.get(tid),
        ev,
        crop(),
        now,
        f
    )

    return ev


def lines():
    return open("logs/events.log").read().splitlines()


def count(db, kind):
    return sum(1 for r in db.rows if r[0] == kind)


def imgs(sub):
    return glob.glob(f"logs/{sub}/*/*.jpg")


# TEST 1: Person already inside, then exits
vp = make()
db = vp.database

assert step(vp, 1, IN, 1, "FACE_006") == "INITIAL_INSIDE"
assert step(vp, 1, OUT, 5, "FACE_006") == "EXIT"

assert (count(db, "ENTRY"), count(db, "EXIT")) == (1, 1)
assert len(imgs("entries")) == 1
assert len(imgs("exits")) == 1

print("1 OK: Initial-inside person gets paired ENTRY/EXIT")


# TEST 2: EXIT without an existing visit
vp = make()
db = vp.database

vp.track_to_face[7] = "FACE_009"

vp.process_track_event(
    7, "FACE_009", "EXIT", crop(), T0, 3
)

assert db.rows == []
assert not imgs("exits")
assert any("EXIT_SUPPRESSED" in l for l in lines())

print("2 OK: Unpaired EXIT suppressed")


# TEST 3: Delayed recognition preserves original crossing time
vp = make()
db = vp.database

step(vp, 17, OUT, 1)
ev = step(vp, 17, IN, 2)

assert ev == "ENTRY"
assert db.rows == []
assert 17 in vp.pending_crossings

step(vp, 17, IN, 40, "FACE_010")

assert count(db, "ENTRY") == 1
assert db.rows[0][2] == T0 + timedelta(seconds=2)

print("3 OK: Pending ENTRY uses original crossing time")


# TEST 4: ByteTrack ID change
vp = make()
db = vp.database

step(vp, 5, OUT, 1, "FACE_001")
step(vp, 5, IN, 2, "FACE_001")
step(vp, 9, IN, 10, "FACE_001")

for f in range(20, 95, 5):
    step(vp, 9, IN, f, "FACE_001")

exp = vp.boundary_counter.cleanup(94)

assert [e[0] for e in exp] == [5], exp

vp.handle_expired_tracks(exp, 94)

assert (count(db, "ENTRY"), count(db, "EXIT")) == (1, 0)

step(vp, 9, OUT, 100, "FACE_001")

assert (count(db, "ENTRY"), count(db, "EXIT")) == (1, 1)

print("4 OK: Track ID change does not create duplicate visits")


# TEST 4B: Real exit on the new tracking ID
vp = make()
db = vp.database

step(vp, 5, OUT, 1, "FACE_001")
step(vp, 5, IN, 2, "FACE_001")
step(vp, 9, IN, 10, "FACE_001")
step(vp, 9, OUT, 30, "FACE_001")

assert (count(db, "ENTRY"), count(db, "EXIT")) == (1, 1)

print("4b OK: New track ID closes the same visit")


# TEST 5: Never-identified crossing
vp = make()
db = vp.database

step(vp, 30, OUT, 1)
step(vp, 30, IN, 2)

exp = vp.boundary_counter.cleanup(500)
vp.handle_expired_tracks(exp, 500)

assert db.rows == []
assert len(imgs("unidentified")) == 1
assert any("UNIDENTIFIED_ENTRY" in l for l in lines())

print("5 OK: Unidentified crossing evidence saved")


# TEST 6: Track disappears while inside
vp = make()
db = vp.database

step(vp, 2, OUT, 1, "FACE_002")
step(vp, 2, IN, 2, "FACE_002")
step(vp, 2, IN, 50, "FACE_002")

vp.handle_expired_tracks(
    vp.boundary_counter.cleanup(400), 400
)

assert count(db, "EXIT") == 1
assert db.rows[-1][2] == T0 + timedelta(seconds=50)

print("6 OK: Lost track gets EXIT at last-seen time")


# TEST 7: Video ends with open visits
vp = make()
db = vp.database

step(vp, 2, IN, 1, "FACE_002")
step(vp, 3, IN, 1, "FACE_003")

vp.finalize_session(100)

assert (count(db, "ENTRY"), count(db, "EXIT")) == (2, 2)
assert not db.active

print("7 OK: Video end closes all open visits")


# TEST 8: Database failure
vp = make()
db = vp.database
db.fail = True

step(vp, 1, OUT, 1, "FACE_001")
step(vp, 1, IN, 2, "FACE_001")

assert not imgs("entries")
assert not glob.glob("logs/_staging/*")
assert not any(" EVENT: ENTRY " in l for l in lines())
assert any("DATABASE_ENTRY_FAILED" in l for l in lines())

print("8 OK: Database failure does not create an ENTRY record")


# TEST 9: Existing active database visit
vp = make()
db = vp.database
db.preactive.add("FACE_006")

step(vp, 1, IN, 1, "FACE_006")

assert not imgs("entries")
assert db.rows == []

step(vp, 1, OUT, 5, "FACE_006")

assert count(db, "EXIT") == 1
assert not vp.session_visits

print("9 OK: Existing active visit handled without orphan image")


# TEST 10: Dry-run mode
vp = make(dry=True)
db = vp.database

step(vp, 1, IN, 1, "FACE_006")
step(vp, 1, OUT, 5, "FACE_006")

assert db.rows == []
assert len(imgs("entries")) == 1
assert len(imgs("exits")) == 1

print("10 OK: Dry-run pairs and logs without database writes")


# TEST 11: Boundary grey-zone handling
bc = BoundaryCounter()

assert bc.update(1, 60, 500, W, H, 1) is None
assert bc.update(1, 500, 500, W, H, 2) == "ENTRY"

assert bc.update(2, 10, 500, W, H, 1) is None
assert bc.update(2, 10, 500, W, H, 2) is None

print("11 OK: Grey-zone boundary handling")

print("\nALL TESTS PASSED")
