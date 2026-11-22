"""Streamlit dashboard for the face-detection project.

Run from the project root:
    python -m streamlit run streamlit_app.py

The "Process Video" page starts `python -m src.video_processor` as a separate
subprocess, so the AI backend never runs inside Streamlit itself.
"""

from __future__ import annotations

import glob
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


# -----------------------------------------------------------------------------
# App configuration and paths
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
LOG_FILE = PROJECT_ROOT / "logs" / "events.log"
SAMPLE_LOG_FILE = PROJECT_ROOT / "sample_output" / "events.log"
ENTRY_DIR = PROJECT_ROOT / "logs" / "entries"
EXIT_DIR = PROJECT_ROOT / "logs" / "exits"
UNIDENTIFIED_DIR = PROJECT_ROOT / "logs" / "unidentified"
SAMPLE_OUTPUT_DIR = PROJECT_ROOT / "sample_output"
REGISTERED_FACES_DIR = PROJECT_ROOT / "data" / "registered_faces"

# Used by the Process Video page
UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"
RUN_LOG = PROJECT_ROOT / "logs" / "processor_stdout.log"
STOP_FILE = PROJECT_ROOT / "logs" / "STOP"

st.set_page_config(
    page_title="Face Tracker Dashboard",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------------------------------------------------------
# Safe file helpers
# -----------------------------------------------------------------------------
def read_text_file(path: Path, max_lines: int = 1000) -> list[str]:
    try:
        if not path.is_file():
            return []
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            return handle.readlines()[-max_lines:]
    except OSError as exc:
        st.warning(f"Could not read {path.name}: {exc}")
        return []


def image_files(folder: Path) -> list[str]:
    if not folder.is_dir():
        return []
    patterns = ("*.jpg", "*.jpeg", "*.png", "*.webp")
    found: list[str] = []
    for pattern in patterns:
        found.extend(glob.glob(str(folder / "**" / pattern), recursive=True))
    return sorted(found, key=os.path.getmtime)


def identity_folders(base: Path = REGISTERED_FACES_DIR) -> list[Path]:
    if not base.is_dir():
        return []
    return sorted((p for p in base.iterdir() if p.is_dir()), key=lambda p: p.name)


def show_image_gallery(paths: list[str], columns: int = 5, limit: int = 40) -> None:
    if not paths:
        st.info("No images found in this folder yet.")
        return
    selected = paths[-limit:]
    cols = st.columns(columns)
    for index, image_path in enumerate(selected):
        with cols[index % columns]:
            st.image(image_path, caption=Path(image_path).name, use_container_width=True)


# -----------------------------------------------------------------------------
# Optional read-only database query integration.
# The dashboard still works if src.db_queries is not present or MySQL is offline.
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_db_queries_module():
    """Load src/db_queries.py straight from its file.

    Importing it as `from src import db_queries` would also run
    src/__init__.py, which can pull in heavy AI libraries and freeze or
    crash Streamlit. Loading the file directly avoids that.
    """
    import importlib.util

    path = PROJECT_ROOT / "src" / "db_queries.py"
    if not path.is_file():
        return None
    try:
        spec = importlib.util.spec_from_file_location("db_queries_direct", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception:
        return None


def call_db_function(function_name: str, *args, **kwargs):
    module = get_db_queries_module()
    if module is None:
        raise RuntimeError(
            "Optional module src.db_queries is unavailable. "
            "The dashboard can still display local logs and sample output."
        )
    function = getattr(module, function_name, None)
    if not callable(function):
        raise RuntimeError(f"src.db_queries does not provide {function_name}().")
    return function(*args, **kwargs)


def database_status() -> tuple[bool, str]:
    try:
        call_db_function("get_kpis")
        return True, "MySQL connection available through src.db_queries."
    except Exception as exc:
        return False, str(exc)


def parse_event_lines(lines: list[str]) -> pd.DataFrame:
    """Parse the project's pipe-separated events.log format."""
    records: list[dict[str, str]] = []
    for raw_line in lines:
        line = raw_line.strip()
        if not line or "| EVENT:" not in line:
            continue
        parts = [part.strip() for part in line.split("|")]
        if len(parts) < 4:
            continue
        record = {
            "Timestamp": parts[0],
            "Event": parts[1].replace("EVENT:", "", 1).strip(),
            "Face ID": parts[2].replace("FACE:", "", 1).strip(),
            "Track ID": parts[3].replace("TRACK:", "", 1).strip(),
            "Details": " | ".join(parts[4:]) if len(parts) > 4 else "",
        }
        records.append(record)
    return pd.DataFrame(records)


# -----------------------------------------------------------------------------
# Sidebar navigation
# -----------------------------------------------------------------------------
st.sidebar.title("🎭 Face Tracker")
st.sidebar.caption("Visitor monitoring dashboard")
page = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Process Video",
        "Visit Log",
        "Event Log",
        "Registered Faces",
        "Sample Output",
        "Verification",
    ],
)
st.sidebar.divider()
st.sidebar.caption("Video processing runs as a separate backend process.")


# -----------------------------------------------------------------------------
# Overview
# -----------------------------------------------------------------------------
if page == "Overview":
    st.title("🎭 Face Detection Dashboard")
    st.write("Monitor registered identities, visitor events and saved evidence.")

    db_ok, db_message = database_status()
    if db_ok:
        st.success(db_message)
        try:
            kpis = call_db_function("get_kpis")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Unique Visitors", kpis.get("unique_visitors", "—"))
            c2.metric("Active Visits", kpis.get("active_visits", "—"))
            c3.metric("Entries Today", kpis.get("entries_today", "—"))
            c4.metric("Exits Today", kpis.get("exits_today", "—"))
        except Exception as exc:
            st.warning(f"Could not load database summary: {exc}")

        try:
            daily_rows = call_db_function("get_daily_counts")
            if daily_rows:
                daily_df = pd.DataFrame(daily_rows, columns=["Date", "Entries", "Exits"])
                daily_df["Date"] = pd.to_datetime(daily_df["Date"], errors="coerce")
                daily_df = daily_df.dropna(subset=["Date"]).set_index("Date")
                st.subheader("Daily visitor activity")
                st.line_chart(daily_df)
        except Exception as exc:
            st.info(f"Daily database chart is unavailable: {exc}")
    else:
        st.info("MySQL dashboard queries are unavailable. Showing local project evidence instead.")
        st.caption(db_message)
        event_lines = read_text_file(SAMPLE_LOG_FILE, 2000) or read_text_file(LOG_FILE, 2000)
        event_df = parse_event_lines(event_lines)
        if not event_df.empty:
            entries = int((event_df["Event"] == "ENTRY").sum())
            exits = int((event_df["Event"] == "EXIT").sum())
            registered = len(identity_folders())
            unidentified = len(image_files(UNIDENTIFIED_DIR))
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Entry Events", entries)
            c2.metric("Exit Events", exits)
            c3.metric("Registered Face Folders", registered)
            c4.metric("Unidentified Images", unidentified)
        else:
            st.warning("No parseable event log was found yet.")

    st.subheader("Registered faces preview")
    folders = identity_folders()
    if folders:
        cols = st.columns(8)
        for index, folder in enumerate(folders[:16]):
            candidates = []
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
                candidates.extend(glob.glob(str(folder / ext)))
            with cols[index % 8]:
                if candidates:
                    st.image(candidates[0], caption=folder.name, use_container_width=True)
                else:
                    st.caption(folder.name)
        if len(folders) > 16:
            st.caption(f"Showing 16 of {len(folders)} registered identity folders.")
    else:
        st.info(f"No registered-face image folders found at {REGISTERED_FACES_DIR}.")


# -----------------------------------------------------------------------------
# Process Video: starts src.video_processor as a separate subprocess.
# -----------------------------------------------------------------------------
elif page == "Process Video":
    st.title("🎬 Process Video")
    st.caption(
        "Upload a video and start the existing backend. It runs in a separate "
        "process, so this page stays responsive."
    )

    uploaded = st.file_uploader("Choose a video", type=["mp4", "avi", "mov", "mkv"])
    if uploaded is not None:
        st.video(uploaded)

    proc = st.session_state.get("proc")
    running = proc is not None and proc.poll() is None

    col1, col2 = st.columns(2)

    if col1.button("▶ Start processing", disabled=(uploaded is None or running)):
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
        STOP_FILE.unlink(missing_ok=True)

        video_path = UPLOAD_DIR / Path(uploaded.name).name
        video_path.write_bytes(uploaded.getbuffer())

        log_handle = open(RUN_LOG, "w", encoding="utf-8")
        st.session_state.proc = subprocess.Popen(
            [sys.executable, "-m", "src.video_processor"],
            cwd=PROJECT_ROOT,
            env={
                **os.environ,
                "VIDEO_PATH": str(video_path),
                "FACE_TRACKER_HEADLESS": "1",
                "FACE_TRACKER_STOP_FILE": str(STOP_FILE),
                "PYTHONUNBUFFERED": "1",
                "PYTHONIOENCODING": "utf-8",
            },
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        st.rerun()

    if col2.button("⏹ Stop (finish cleanly)", disabled=not running):
        STOP_FILE.write_text("stop")
        st.info("Stop requested. The backend will close open visits and exit.")

    @st.fragment(run_every=2)
    def monitor():
        p = st.session_state.get("proc")
        if p is None:
            st.info("Nothing running yet.")
            return
        code = p.poll()
        if code is None:
            st.info("⏳ Processing... (model loading can take a minute at first)")
        elif code == 0:
            st.success("✅ Finished. Open Event Log or Sample Output to see results.")
        else:
            st.error(f"❌ Backend exited with code {code}. See output below.")
        st.text_area("Backend output", "".join(read_text_file(RUN_LOG, 200)), height=300)

    monitor()


# -----------------------------------------------------------------------------
# Visit Log
# -----------------------------------------------------------------------------
elif page == "Visit Log":
    st.title("📋 Visit History")
    db_ok, message = database_status()
    if not db_ok:
        st.warning("MySQL visit history is not available through the optional query module.")
        st.caption(message)
        st.stop()

    try:
        rows = call_db_function("get_visits")
    except Exception as exc:
        st.error(f"Could not retrieve visits: {exc}")
        st.stop()

    if not rows:
        st.info("No visits found.")
        st.stop()

    df = pd.DataFrame(rows, columns=["Visit ID", "Face ID", "Entry Time", "Exit Time", "Status"])
    left, right = st.columns(2)
    status_filter = left.selectbox("Visit status", ["All", "ACTIVE", "COMPLETED"])
    face_filter = right.text_input("Filter by Face ID")
    if status_filter != "All":
        df = df[df["Status"].astype(str) == status_filter]
    if face_filter:
        df = df[df["Face ID"].astype(str).str.contains(face_filter, case=False, na=False)]
    st.caption(f"{len(df)} visit records")
    st.dataframe(df, use_container_width=True, hide_index=True)


# -----------------------------------------------------------------------------
# Event Log
# -----------------------------------------------------------------------------
elif page == "Event Log":
    st.title("📜 Event History")
    source = st.radio(
        "Event source",
        ["Current logs/events.log", "Sample output log", "Live MySQL"],
        horizontal=True,
    )

    if source == "Live MySQL":
        db_ok, message = database_status()
        if not db_ok:
            st.warning("MySQL event queries are unavailable.")
            st.caption(message)
            st.stop()
        try:
            rows = call_db_function("get_recent_events")
            df = pd.DataFrame(
                rows,
                columns=["Event ID", "Visit ID", "Face ID", "Type", "Time", "Image"],
            )
            event_types = st.multiselect(
                "Event types",
                sorted(df["Type"].dropna().astype(str).unique().tolist()),
                default=sorted(df["Type"].dropna().astype(str).unique().tolist()),
            )
            if event_types:
                df = df[df["Type"].isin(event_types)]
            st.dataframe(df, use_container_width=True, hide_index=True)
        except Exception as exc:
            st.error(f"Could not retrieve MySQL events: {exc}")
    else:
        path = LOG_FILE if source == "Current logs/events.log" else SAMPLE_LOG_FILE
        lines = read_text_file(path, 2000)
        filter_text = st.text_input("Filter log text")
        if filter_text:
            lines = [line for line in lines if filter_text.casefold() in line.casefold()]
        if lines:
            st.text_area("Log contents", "".join(lines), height=520)
            parsed = parse_event_lines(lines)
            if not parsed.empty:
                st.subheader("Parsed events")
                st.dataframe(parsed, use_container_width=True, hide_index=True)
        else:
            st.info(f"No log lines found at {path}.")


# -----------------------------------------------------------------------------
# Registered Faces
# -----------------------------------------------------------------------------
elif page == "Registered Faces":
    st.title("🧑 Registered Faces")
    folders = identity_folders()

    # The database is the source of truth for who is registered. Image folders
    # left over from earlier runs or a database reset are not real identities.
    db_ok, _db_message = database_status()
    if db_ok:
        try:
            db_ids = {row[0] for row in call_db_function("get_faces")}
            orphan = [f for f in folders if f.name not in db_ids]
            folders = [f for f in folders if f.name in db_ids]
            st.caption(
                f"{len(db_ids)} registered in the database | "
                f"{len(folders)} have saved images | "
                f"{len(orphan)} leftover image folders ignored"
            )
        except Exception as exc:
            st.warning(f"Could not read registered faces from MySQL: {exc}")

    if not folders:
        st.info(f"No registered-face image folders found at {REGISTERED_FACES_DIR}.")
        st.caption("The database may contain identities even if image folders are stored elsewhere.")
        st.stop()

    search = st.text_input("Search Face ID")
    if search:
        folders = [folder for folder in folders if search.casefold() in folder.name.casefold()]
    column_count = st.slider("Gallery columns", min_value=2, max_value=8, value=4)
    st.caption(f"{len(folders)} identity folders")
    cols = st.columns(column_count)
    for index, folder in enumerate(folders):
        images: list[str] = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
            images.extend(glob.glob(str(folder / ext)))
        with cols[index % column_count]:
            if images:
                st.image(images[0], caption=f"{folder.name} ({len(images)} image(s))", use_container_width=True)
                if len(images) > 1:
                    with st.expander(f"View images for {folder.name}"):
                        for image_path in images[1:]:
                            st.image(image_path, use_container_width=True)
            else:
                st.caption(folder.name)


# -----------------------------------------------------------------------------
# Sample Output
# -----------------------------------------------------------------------------
elif page == "Sample Output":
    st.title("🖼️ Sample Output")
    tabs = st.tabs(["Entries", "Exits", "Unidentified", "Event Log"])

    with tabs[0]:
        st.subheader("Sample entry images")
        paths = image_files(SAMPLE_OUTPUT_DIR / "entries") or image_files(ENTRY_DIR)
        show_image_gallery(paths)

    with tabs[1]:
        st.subheader("Sample exit images")
        paths = image_files(SAMPLE_OUTPUT_DIR / "exits") or image_files(EXIT_DIR)
        show_image_gallery(paths)

    with tabs[2]:
        st.subheader("Unidentified crossings")
        paths = image_files(SAMPLE_OUTPUT_DIR / "unidentified") or image_files(UNIDENTIFIED_DIR)
        show_image_gallery(paths)

    with tabs[3]:
        st.subheader("Sample event log")
        lines = read_text_file(SAMPLE_LOG_FILE, 2000)
        if not lines:
            lines = read_text_file(LOG_FILE, 2000)
        if lines:
            st.text_area("Events", "".join(lines), height=520)
        else:
            st.info("No sample or current event log found.")


# -----------------------------------------------------------------------------
# Verification: do the database, events.log and saved images agree?
# -----------------------------------------------------------------------------
elif page == "Verification":
    st.title("✅ Verification")
    st.caption("Checks that the database, events.log and saved images agree with each other.")

    db_ok, db_message = database_status()
    if not db_ok:
        st.warning("MySQL is unavailable, so the database checks cannot run.")
        st.caption(db_message)
        st.stop()

    try:
        summary = call_db_function("get_integrity_summary")
    except Exception as exc:
        st.error(f"Could not read the database summary: {exc}")
        st.stop()

    # ---- 1. Unique visitors ------------------------------------------------
    st.subheader("1. Unique visitor count")
    log_df = parse_event_lines(read_text_file(LOG_FILE, 100000))
    log_faces = set()
    if not log_df.empty:
        entry_rows = log_df[log_df["Event"] == "ENTRY"]
        log_faces = {f for f in entry_rows["Face ID"] if f and f != "UNKNOWN"}

    c1, c2, c3 = st.columns(3)
    c1.metric("Registered faces (database)", summary["faces"])
    c2.metric("Faces with at least one visit", summary["faces_with_visits"])
    c3.metric("Faces with an ENTRY in current events.log", len(log_faces))
    st.caption(
        "The official unique-visitor count is the number of registered faces in the "
        "database. The events.log figure only covers the current log file, so it is "
        "lower if the log was renamed or restarted."
    )
    if summary["faces"] == summary["faces_with_visits"]:
        st.success("Every registered face has at least one recorded visit.")
    else:
        gap = summary["faces"] - summary["faces_with_visits"]
        st.info(
            f"{gap} registered face(s) never had a visit recorded. This can happen "
            "when a person was registered but never crossed the entry boundary."
        )

    # ---- 2. Entry / exit pairing --------------------------------------------
    st.subheader("2. Entry and exit pairing")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ENTRY events", summary["entry_events"])
    c2.metric("EXIT events", summary["exit_events"])
    c3.metric("Completed visits", summary["visits_completed"])
    c4.metric("Still active", summary["visits_active"])
    if (
        summary["entry_events"] == summary["visits_total"]
        and summary["exit_events"] == summary["visits_completed"]
        and summary["visits_active"] == 0
    ):
        st.success("Every visit has exactly one ENTRY and one EXIT event.")
    else:
        st.warning(
            "Counts do not line up: each visit should have one ENTRY event, each "
            "completed visit one EXIT event, and no visits should stay active after "
            "a run finishes."
        )

    # ---- 3. Saved images ----------------------------------------------------
    st.subheader("3. Event images on disk")
    try:
        event_rows = call_db_function("get_recent_events", 2000)
        missing = []
        for _n, visit_id, face_id, event_type, event_time, image_path in event_rows:
            if not image_path:
                missing.append((visit_id, face_id, event_type, event_time, "(no path)"))
                continue
            path = Path(str(image_path))
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            if not path.is_file():
                missing.append((visit_id, face_id, event_type, event_time, str(image_path)))
        st.caption(f"Checked the newest {len(event_rows)} database events.")
        if not missing:
            st.success("Every checked database event points to an image that exists.")
        else:
            st.warning(f"{len(missing)} event(s) point to a missing image file.")
            st.dataframe(
                pd.DataFrame(
                    missing,
                    columns=["Visit ID", "Face ID", "Type", "Time", "Image path"],
                ),
                use_container_width=True,
                hide_index=True,
            )
    except Exception as exc:
        st.error(f"Could not check event images: {exc}")

    # ---- 4. Possible duplicate identities -----------------------------------
    st.subheader("4. Possible duplicate identities")
    st.caption(
        "Compares the stored face embeddings of every pair of registered identities. "
        "A high score means two Face IDs look like the same person, which would "
        "inflate the unique-visitor count. Treat the scores as hints and confirm by eye."
    )
    threshold = st.slider("Show pairs with similarity at or above", 0.20, 0.80, 0.35, 0.05)
    try:
        pairs = call_db_function("get_similar_face_pairs", threshold, 30)
        if not pairs:
            st.success("No pairs at or above this similarity.")
        else:
            st.warning(f"{len(pairs)} pair(s) found. The matcher treats about 0.45 and above as the same person.")
            for face_a, face_b, score in pairs[:10]:
                left, middle, right = st.columns([1, 1, 2])
                for column, face in ((left, face_a), (middle, face_b)):
                    images = []
                    folder = REGISTERED_FACES_DIR / face
                    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
                        images.extend(glob.glob(str(folder / ext)))
                    with column:
                        if images:
                            st.image(images[0], caption=face, use_container_width=True)
                        else:
                            st.caption(f"{face} (no image)")
                right.metric(f"{face_a} vs {face_b}", f"{score:.2f}")
            if len(pairs) > 10:
                st.dataframe(
                    pd.DataFrame(pairs, columns=["Face A", "Face B", "Similarity"]),
                    use_container_width=True,
                    hide_index=True,
                )
    except Exception as exc:
        st.error(f"Could not compare embeddings: {exc}")

    # ---- 5. Image folders vs database ---------------------------------------
    st.subheader("5. Face image folders vs database")
    try:
        db_ids = {row[0] for row in call_db_function("get_faces")}
        folder_names = {f.name for f in identity_folders()}
        orphan = sorted(folder_names - db_ids)
        no_image = sorted(db_ids - folder_names)
        c1, c2, c3 = st.columns(3)
        c1.metric("Faces in database", len(db_ids))
        c2.metric("Image folders", len(folder_names))
        c3.metric("Leftover folders", len(orphan))
        if orphan:
            st.info(
                "These folders have no matching database record. They usually come "
                "from earlier runs or a database reset: " + ", ".join(orphan)
            )
        if no_image:
            st.info("Registered in the database but no image folder: " + ", ".join(no_image))
        if not orphan and not no_image:
            st.success("Image folders and database identities match exactly.")
    except Exception as exc:
        st.error(f"Could not compare folders with the database: {exc}")


# Footer
st.sidebar.divider()
st.sidebar.caption("Face Tracker • Backend runs as a separate process")