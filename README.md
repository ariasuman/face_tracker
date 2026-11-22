<div align="center">

# 🎭 Intelligent Face Tracker with Auto-Registration and Visitor Counting

**Detect, recognise and track faces in video. Automatically register new visitors, log every entry and exit with a timestamped face image, and count unique visitors.**

[![Watch the demo on Loom](https://img.shields.io/badge/%E2%96%B6%20Watch%20the%20Demo-Loom-625DF5?style=for-the-badge&logo=loom&logoColor=white)](https://www.loom.com/share/9f6458a8ad5846448371ffd2f75f8067)

![Python](https://img.shields.io/badge/Python-3-3776AB?logo=python&logoColor=white)
![YOLO](https://img.shields.io/badge/Ultralytics-YOLO-111F68)
![InsightFace](https://img.shields.io/badge/InsightFace-buffalo__l-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=white)
![MySQL](https://img.shields.io/badge/MySQL-4479A1?logo=mysql&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)

[Demo](#-demo) · [Features](#-features) · [Architecture](#-system-architecture) · [Getting Started](#-getting-started) · [Usage](#-usage) · [Dashboard](#-dashboard) · [Testing](#-testing)

</div>

---

## 🎥 Demo

**▶ [Watch the full project walkthrough on Loom](https://www.loom.com/share/9f6458a8ad5846448371ffd2f75f8067)**

`https://www.loom.com/share/9f6458a8ad5846448371ffd2f75f8067`

---

## 📌 About the Project

This project reads a video (a file, or a network stream such as RTSP), finds every face, works out **who it is**, and records when that person **enters** and **leaves** the monitored area. Each person gets a permanent `FACE_###` ID. Every visit, with the cropped face image from the moment of entry and exit, is stored in MySQL.

**The problem it solves:** counting "line crossings" over-counts anyone who comes back, and counting tracker IDs over-counts whenever a tracker loses and re-acquires a person. This project keeps three things separate:

| Concept | Identified by | Stored as |
|---|---|---|
| **Person** | Face embedding (permanent `FACE_###`) | one row in `faces` |
| **Visit** | One ENTRY paired with one EXIT | one row in `visits` |
| **Track** | Temporary ByteTrack ID | memory only, never the identity |

**In short**
- **Detect and track:** a YOLO face model with ByteTrack gives face boxes and temporary track IDs.
- **Recognise:** InsightFace (`buffalo_l`) gives a 512-d embedding, compared by cosine similarity with every stored template. The track is then linked to a permanent Face ID.
- **Auto-register:** an unrecognised face is *not* registered at once. Observations are gathered in a pending group, and a new identity is created only after several consistent, good-quality, front-facing observations.
- **Count unique visitors:** the number of rows in the `faces` table (`SELECT COUNT(*) FROM faces`). A returning person gets a new *visit*, not a new *face*.

---

## ✨ Features

| Feature | Implemented in |
|---|---|
| Face detection (YOLO face model) | `src/tracker.py` |
| Facial embedding generation (InsightFace, 512-d) | `src/video_processor.py`, `src/face_recognition.py` |
| Recognition and re-identification (cosine similarity, best template, margin check) | `src/face_matcher.py` |
| Automatic registration of new visitors | `VideoProcessor.register_confirmed_candidate`, `Database.save_face` |
| Multi-frame identity validation (pending groups, consistency check) | `VideoProcessor.process_identity` |
| Face tracking (ByteTrack) | `src/tracker.py` |
| Entry / exit detection with hysteresis | `src/boundary_counter.py` |
| Timestamped cropped face image for every ENTRY / EXIT | `src/event_logger.py` |
| Database persistence (faces, embeddings, visits, events) | `src/database.py`, `schema.sql` |
| Unique visitor counting | `Database.get_unique_visitor_count`, `db_queries.get_kpis` |
| Event logging to `logs/events.log` | `src/event_logger.py` |
| Duplicate-event suppression and ENTRY/EXIT pairing | `VideoProcessor.record_line_crossing` |
| Delayed identification: crossings seen before the ID is known are replayed with their original time | `queue_pending_crossing` |
| Unidentified-crossing evidence (image kept, no DB record) | `EventLogger.save_unidentified_image` |
| Live stream input with newest-frame reader and auto-reconnect | `src/video_source.py` |
| Streamlit dashboard (7 pages, including video upload and integrity checks) | `streamlit_app.py` |

---

## 🛠 Tech Stack

| Technology | Purpose in this project |
|---|---|
| **Python** | Whole project |
| **YOLO** (Ultralytics, `yolov8n-face.pt`) | Face detection through `YOLO(...).track(...)` |
| **ByteTrack** | Temporary track IDs, enabled with `tracker="bytetrack.yaml"` in the Ultralytics call (no separate ByteTrack code) |
| **OpenCV** | Video input, crop quality checks (sharpness, brightness), cropping, saving JPEGs, preview window |
| **InsightFace** | `buffalo_l` on CPU: embeddings and head-yaw estimate |
| **ArcFace** | The 512-d embeddings come from InsightFace's `buffalo_l` pack (code comments call them ArcFace embeddings). There is no separate ArcFace code |
| **NumPy** | Normalisation, dot products, weighted averages, BLOB ⇄ `float32` |
| **Cosine similarity** | Dot product of L2-normalised embeddings in `FaceMatcher` |
| **MySQL** (`mysql-connector-python`) | Storage for faces, embeddings, visits, events |
| **python-dotenv** | Loads DB credentials from `.env` |
| **Streamlit + pandas** | Dashboard (runs separately from the pipeline) |

---

## 🏗 System Architecture

```mermaid
flowchart TD
    A["Video input<br/>file via cv2.VideoCapture<br/>live stream via StreamReader"] --> B["Frame sampling<br/>config.json: processing.frame_skip"]
    B --> C["Face detection + ByteTrack<br/>YOLO face model, bytetrack.yaml<br/>src/tracker.py"]

    C --> D["Quality gate<br/>confidence, size, sharpness, brightness"]
    D --> E["InsightFace buffalo_l on full frame<br/>matched to tracked box by IoU >= 0.10<br/>512-d embedding + yaw"]
    E --> F["Identity matching<br/>cosine similarity vs stored templates<br/>src/face_matcher.py"]

    F -->|"KNOWN"| G["track_to_face<br/>Track ID linked to permanent Face ID"]
    F -->|"UNCERTAIN or NEW_CANDIDATE"| H["Pending identity group<br/>multi-frame validation"]
    H -->|"resolves to KNOWN"| G
    H -->|"confirmed new person<br/>front-facing views only"| I["Automatic registration<br/>MySQL: faces + face_embeddings<br/>crop in data/registered_faces"]
    I --> G
    G -.->|"new angle, max 10 templates"| J["Additional embedding template"]

    C --> K["BoundaryCounter<br/>face-box centre vs inner/outer rectangle<br/>src/boundary_counter.py"]
    K -->|"ENTRY / EXIT / INITIAL_INSIDE"| L{"Face ID known?"}
    G --> L
    L -->|"no"| M["Pending crossing queue"]
    M -->|"identified later"| O
    M -->|"never identified"| N["logs/unidentified image<br/>no MySQL record"]
    L -->|"yes"| O["Pairing and duplicate suppression<br/>record_line_crossing"]

    O --> P["1. Stage cropped image<br/>logs/_staging"]
    P --> Q["2. MySQL transaction<br/>visits + events"]
    Q -->|"accepted"| R["3. Move image to logs/entries or logs/exits<br/>write logs/events.log"]
    Q -->|"rejected"| S["Staged image deleted"]

    R --> T["Streamlit dashboard<br/>streamlit_app.py + src/db_queries.py"]
    I --> T

    classDef store fill:#e8f1ff,stroke:#3b6fd6,color:#0b2a63;
    class I,Q,R,T store;
```

Each face box follows two parallel paths: the **identity path** decides *who* it is, and the **boundary path** decides *whether it crossed*. An event is written only when both answers exist.

---

## 🚀 Getting Started

> **Not in the repository:** there is no `requirements.txt`, `pyproject.toml` or `.env.example`. The package list below was derived from the code's `import` statements. The YOLO weights, the sample video and the test images are also not included (they are git-ignored).

### Prerequisites
- Python 3 (3.8 or newer)
- Git
- A running MySQL server, with the `mysql` client on your `PATH`
- YOLO face weights `yolov8n-face.pt` and an input video

### 1. Clone and create a virtual environment (Windows CMD)
```bat
git clone https://github.com/ariasuman/face_tracker.git
cd face_tracker
python -m venv venv
venv\Scripts\activate
```

### 2. Install dependencies
```bat
python -m pip install --upgrade pip
pip install numpy opencv-python ultralytics insightface onnxruntime mysql-connector-python python-dotenv streamlit pandas
```
`onnxruntime` is not imported directly, but InsightFace needs it at runtime (the code uses `CPUExecutionProvider`). On Windows, `insightface` may need the Microsoft C++ Build Tools if no pre-built wheel matches your Python version.

### 3. Set up MySQL
`schema.sql` has no `CREATE DATABASE` line, so create the database first:
```bat
mysql -u root -p -e "CREATE DATABASE IF NOT EXISTS face_tracker;"
mysql -u root -p face_tracker < schema.sql
```
Then create a `.env` file in the project root (for example `notepad .env`):
```ini
DB_HOST=localhost
DB_USER=root
DB_PASSWORD=your_mysql_password
DB_NAME=face_tracker
```
`.env` is in `.gitignore`. Never commit it.

> Use `schema.sql`, **not** `database/face_tracker_schema.sql`. The latter is an older schema without the `face_embeddings` table that the code requires.

### 4. Add the model and a video
```bat
mkdir models
mkdir data
```
- Put the YOLO face weights at **`models\yolov8n-face.pt`** (this exact path is loaded by `src/tracker.py`).
- Put your video at **`data\sample.mp4`**, or change `camera.source` in `config.json`.

InsightFace may download its `buffalo_l` model files on first run.

### 5. Configure `config.json`
The defaults work for a first run. Most useful to adjust per camera: `boundary.margin_x_ratio` / `margin_y_ratio`, `processing.frame_skip`, `processing.show_window`, `camera.source`.

<details>
<summary><b>Full <code>config.json</code> reference</b></summary>

```json
{
  "boundary":   { "margin_x_ratio": 0.08, "margin_y_ratio": 0.08, "hysteresis_ratio": 0.02,
                  "max_track_gap": 90, "report_initial_inside": true, "duplicate_track_window": 15 },
  "processing": { "frame_skip": 2, "show_window": true, "display_max_width": 1280, "close_visits_on_end": true },
  "identity":   { "max_register_yaw": 60, "insightface_det_size": 1280 },
  "camera":     { "source": "data/sample.mp4" }
}
```

| Key | Meaning |
|---|---|
| `boundary.margin_x_ratio`, `margin_y_ratio` | Border width as a fraction of frame width / height (8 %) |
| `boundary.hysteresis_ratio` | Dead-band half-width (2 %) separating "clearly inside" from "clearly outside" |
| `boundary.max_track_gap` | A track unseen for more than this many frames is expired (90) |
| `boundary.report_initial_inside` | Report a person first seen already inside as an entry |
| `boundary.duplicate_track_window` | Tracks seen within this many frames count as "still present" (15) |
| `processing.frame_skip` | For **files**: analyse 1 of every `frame_skip + 1` frames (here 1 of 3). Ignored for live streams |
| `processing.show_window` | Show the OpenCV preview (press `q` to quit) |
| `processing.close_visits_on_end` | Give an EXIT to anyone still inside when processing ends |
| `identity.max_register_yaw` | Maximum head turn (degrees) for registering a **new** person (60 here; code default 35) |
| `identity.insightface_det_size` | InsightFace detection size (1280 here; code default 640) |
| `camera.source` | Video file path **or** stream URL, used when `VIDEO_PATH` is not set |

Optional keys read by the code that are **not** in `config.json` (defaults apply): `identity.min_face_size` (60), `template_min_similarity` (0.50), `template_min_margin` (0.08), `uncertain_resolve_observations` (7), `uncertain_new_min_observations` (5), `uncertain_new_max_similarity` (0.40), `uncertain_new_min_coherence` (0.60), `uncertain_new_min_frontal` (3), `two_observation_min_quality` (0.40), `two_observation_min_agreement` (0.75), and `processing.reconnect_delay_seconds` (2) / `reconnect_max_seconds` (60).

</details>

---

## ▶ Usage

Run from the project root with the virtual environment active.

### Process the sample video
```bat
python -m src.video_processor
```
A preview window opens (press **`q`** to stop). When processing ends, a session summary prints entries, exits, suppressed events, unidentified crossings and the unique-visitor count.

Optional overrides (the dashboard uses these):
```bat
set VIDEO_PATH=data\another_video.mp4
set FACE_TRACKER_HEADLESS=1
python -m src.video_processor
```

> `DRY_RUN = False` at the top of `src/video_processor.py` is a constant. Setting it to `True` stops all MySQL writes and registered-face crops. The processor still connects to MySQL at start-up, and event images and `events.log` are still written.

### Start the dashboard
```bat
python -m streamlit run streamlit_app.py
```
Open **http://localhost:8501**.

### RTSP / live streams
Stream input **is implemented** (`src/video_source.py`). Set a stream URL as the source:
```json
"camera": { "source": "rtsp://user:password@camera-ip:554/stream" }
```
Sources starting with `rtsp://`, `rtsps://`, `rtmp://`, `http://` or `https://` are treated as live streams: the newest frame is always analysed (`frame_skip` is ignored), RTSP is requested over TCP, and the reader reconnects automatically (retry every 2 s, give up after 60 s). The dashboard's upload page supports files only.

> **Validation status:** the repository has no RTSP test, sample URL or recorded live-camera result, so live RTSP is **implemented but not demonstrated here**.

### Check a run
```bat
mysql -u root -p face_tracker < verify_integrity.sql
```
After a clean run, the "open visits" and "unpaired events" queries should return no rows. The dashboard's **Verification** page runs equivalent checks.

---

## 📊 Dashboard

`streamlit_app.py` has seven pages. Database pages read through `src/db_queries.py`, which is **read-only**. The **Process Video** page starts the backend as a **separate subprocess**, so the AI models never load inside Streamlit.

| Page | What it shows |
|---|---|
| **Overview** | KPIs: **Unique Visitors**, **Active Visits**, **Entries Today**, **Exits Today**; a 30-day entries-vs-exits line chart; a preview of up to 16 registered faces. Falls back to counts from the event log and image folders if MySQL is unreachable |
| **Process Video** | Upload `.mp4 / .avi / .mov / .mkv`, start the headless backend, **Stop (finish cleanly)** so open visits are closed, live backend output every 2 s |
| **Visit Log** | Latest 1,000 visits (visit ID, face ID, entry, exit, status), filter by status and Face ID |
| **Event Log** | `logs/events.log`, a sample log, or **live MySQL events**; keyword and event-type filters; parsed events table |
| **Registered Faces** | Gallery of saved face crops per identity. The database is the source of truth, so leftover folders without a DB record are ignored |
| **Sample Output** | Entry, exit and unidentified images plus the event log |
| **Verification** | Five checks: unique-visitor count vs visits; ENTRY/EXIT/visit counts agree; every event's `image_path` exists; **possible duplicate identities** by embedding similarity; image folders vs database |

The dashboard looks for an optional `sample_output/` folder first. It is not in the repository, so it falls back to `logs/`.

---

## 🧠 How It Works

<details>
<summary><b>Face recognition logic</b> (embeddings, thresholds, multi-frame validation, registration, duplicate avoidance)</summary>

### Generating embeddings
1. YOLO + ByteTrack produces a tracked face box.
2. The crop must pass the **quality gate** (cheap, runs *before* the expensive model): detection confidence ≥ 0.35 · at least 60 × 60 px · Laplacian sharpness ≥ 20 · mean brightness between 30 and 225. Passing faces get a quality score (30 % size, 30 % sharpness, 25 % confidence, 15 % brightness) used to weight averaging.
3. InsightFace `buffalo_l` analyses the **full frame** once, and only if some face needs it.
4. The InsightFace face overlapping the tracked box (**IoU ≥ 0.10**) supplies its 512-d `normed_embedding` and **yaw** (`pose[1]`).
5. Unidentified tracks are sampled every **5** frames, identified tracks every **15**.

### Cosine similarity
Embeddings are L2-normalised, so cosine similarity is a dot product. Each person can hold several templates; their score is the **best template similarity** (`FaceMatcher.find_top_matches`).

### Known / Uncertain / New Candidate
Runtime thresholds (set where `VideoProcessor` creates the `FaceMatcher`):

| Condition (best score `s`, margin = best − second best) | Result |
|---|---|
| No registered identities | `NEW_CANDIDATE` |
| `s < 0.30` | `NEW_CANDIDATE` |
| `0.30 ≤ s < 0.45` | `UNCERTAIN` |
| `s ≥ 0.45`, margin `< 0.05` | `UNCERTAIN` (two identities too close) |
| `s ≥ 0.45`, margin `≥ 0.05` | `KNOWN` |

With one registered identity there is no runner-up, so the margin test passes. The code comments call the margin *provisional*, to be calibrated on labelled real video.

### Multi-frame validation
A face that is not `KNOWN` joins a **pending group** instead of being registered:
- **Cross-track merging:** an observation joins a group if it matches the group's weighted average at **≥ 0.82** with a **≥ 0.08** lead over the next group.
- **Consistency check:** an observation must reach **≥ 0.70** against its group's average or it is dropped.
- **Confirmation:** **3 observations** are needed. The quality-weighted group average is re-classified.
- **Expiry:** a group expires after 300 frames with no new observation.

| Group average is… | Action |
|---|---|
| `KNOWN` | Assigned to the existing Face ID |
| `NEW_CANDIDATE` | Register a new person (front-facing view required) |
| `UNCERTAIN` | **Not registered**, unless one of two strict rules (below) resolves it as a new person |

**Resolving `UNCERTAIN` as new** (only if similarity is below 0.45):
- **Rule A:** ≥ 7 observations *and* the best-matching identity is on screen right now on a different track.
- **Rule B** (all must hold): ≥ 5 independent observations (≥ 5 frames apart) · group similarity < 0.40 · no single observation reaches 0.45 against anyone · ≥ 3 frontal observations · frontal observations agree (mean coherence ≥ 0.60).

**Short tracks:** if all tracks of a group ended with only **2** observations, both must have quality ≥ 0.40, be frontal, agree at ≥ 0.75, be ≥ 5 frames apart, and classify as `NEW_CANDIDATE`.

### Registration
`register_face()` allocates the next ID (`FACE_001`, `FACE_002`, …). `Database.save_face()` writes the `faces` row and the first `face_embeddings` row in **one transaction**, and the in-memory registration is rolled back if it fails. The stored embedding is the quality-weighted average of the **frontal** observations. A `REGISTRATION` line goes to `events.log`, and a crop to `data/registered_faces/<FACE_ID>/track_<id>.jpg`.

### Face-angle filtering
Applied to **new registrations only**: `|yaw| ≤ identity.max_register_yaw` (**60°** in `config.json`). If no observation is frontal enough, the group keeps waiting for a better view. A missing pose counts as frontal. Pitch and roll are not used.

### Additional embedding templates
An identified track adds a template only if **all** hold: best match is its own identity with a lead ≥ **0.08**; similarity ≥ **0.50**; ≥ **30** frames since that track last added one; not a near-duplicate (similarity < **0.98**); the identity has fewer than **10** templates.

### How duplicate registration is avoided
1. Every candidate is checked against **all templates of all identities** before registration.
2. The **margin rule** sends ambiguous matches to `UNCERTAIN`, not to "new person".
3. The 0.30–0.45 band **never auto-registers** except via Rules A/B.
4. **Pending groups** merge observations across track IDs, so a tracker ID switch cannot create two identities.
5. The **consistency check** drops outliers.
6. Registration needs **frontal** views.
7. The stored embedding is a **quality-weighted average**, not one frame.
8. The dashboard's **Verification** page lists identity pairs with similar embeddings (default threshold 0.35) for human review.

</details>

<details>
<summary><b>Entry and exit event management</b> (boundary, hysteresis, duplicate suppression, visits, image storage)</summary>

### Boundary crossing
`BoundaryCounter` uses the **centre of the face box** against rectangles derived from the frame size, so it is resolution-independent.

```text
 ┌──────────────────────────────────────┐
 │  OUTSIDE  (beyond the outer rectangle)│
 │   ┌──────────────────────────────┐   │
 │   │  GREY ZONE (undecided)       │   │
 │   │   ┌──────────────────────┐   │   │
 │   │   │   INSIDE             │   │   │
 │   │   └──────────────────────┘   │   │
 │   └──────────────────────────────┘   │
 └──────────────────────────────────────┘
```

| Boundary | Distance from each frame edge (current config) |
|---|---|
| **Inner** (clearly inside) | margin + hysteresis = 8 % + 2 % = **10 %** |
| **Outer** (clearly outside) | margin − hysteresis = 8 % − 2 % = **6 %** |

| Track transition | Result |
|---|---|
| Outside / grey / unknown → clearly inside | `ENTRY` |
| Clearly inside → clearly outside | `EXIT` |
| **First seen** already clearly inside | `INITIAL_INSIDE` (recorded as an ENTRY, note *initial_presence*) |
| First seen in grey zone, later clearly inside | `ENTRY` |

### Hysteresis
Because the inner and outer rectangles differ, a person near an edge must move **clearly** inside or outside to change state, so jitter in the grey zone cannot generate repeated events. (Test 11 exercises the grey-zone behaviour.)

### Duplicate-event suppression
Handled in `record_line_crossing()` **before** anything is written:

| Situation | Behaviour |
|---|---|
| ENTRY while the Face ID already has an open visit (e.g. ByteTrack gave a new ID) | Suppressed; track attached to the existing visit |
| EXIT with no open visit | Suppressed |
| EXIT while another live track of the same visit is still inside | Suppressed |
| MySQL already has an ACTIVE visit for that face (e.g. from an earlier run) | No new ENTRY; image discarded |
| MySQL has no ACTIVE visit for an EXIT | Ignored; image discarded |

In the database, `start_visit()` locks the face row (`SELECT … FOR UPDATE`) and returns the existing ACTIVE visit instead of creating another; `complete_visit()` never lets an EXIT precede its ENTRY.

### Visit management
- A visit is keyed by the **Face ID**, never the track ID.
- **ENTRY** → `visits` row (`ACTIVE`) + `events` row (`ENTRY`) in one transaction.
- **EXIT** → visit set to `COMPLETED` with `exit_time` + `events` row (`EXIT`) in one transaction.
- **Track lost while inside** → EXIT recorded at the *last-seen* time.
- **Video ends / Stop pressed** → with `close_visits_on_end: true`, every open visit gets an EXIT.
- **Crossing before identification** → queued, then replayed in order with its **original time**. If never identified, the crop goes to `logs/unidentified/` and **no database record** is created.

### How each event is saved (three phases)
1. **Stage:** write the face crop to `logs/_staging/`.
2. **Database:** run the MySQL transaction, storing the final image path in `events.image_path`.
3. **Commit or discard:** on success move the image to its final folder and write the `events.log` line; if the database rejected the event, delete the staged image.

So a final image and an ENTRY/EXIT log line exist **only** for events MySQL accepted.

### Where images and timestamps go

| Item | Location |
|---|---|
| Entry crop | `logs/entries/YYYY-MM-DD/<FACE_ID>_ENTRY_<HHMMSS_micro>_track_<id>.jpg` |
| Exit crop | `logs/exits/YYYY-MM-DD/<FACE_ID>_EXIT_<HHMMSS_micro>_track_<id>.jpg` |
| Unidentified crop | `logs/unidentified/YYYY-MM-DD/UNIDENTIFIED_<ENTRY or EXIT>_<HHMMSS_micro>_track_<id>.jpg` |
| Registered face crop | `data/registered_faces/<FACE_ID>/track_<id>.jpg` |
| Text log | `logs/events.log`: `<ISO timestamp> \| EVENT: … \| FACE: … \| TRACK: … \| [AT: <real time> \|] message` |
| Database | `visits.entry_time` / `exit_time`, `events.event_time`, `events.image_path` |

> **Timestamp note:** times come from the system clock (`datetime.now()`) during processing. For a pre-recorded file they reflect *when it was processed*, not the video's own timeline.

</details>

---

## 🗄 Database Design

```mermaid
erDiagram
    faces ||--o{ face_embeddings : "has templates"
    faces ||--o{ visits : "makes"
    faces ||--o{ events : "is logged in"
    visits ||--o{ events : "contains"

    faces {
        varchar face_id PK
        blob embedding
        datetime first_seen
        datetime last_seen
    }
    face_embeddings {
        int embedding_id PK
        varchar face_id FK
        blob embedding
        datetime created_at
    }
    visits {
        int visit_id PK
        varchar face_id FK
        datetime entry_time
        datetime exit_time
        enum status
    }
    events {
        int event_id PK
        int visit_id FK
        varchar face_id FK
        enum event_type
        datetime event_time
        varchar image_path
    }
```

| Table | What it stores |
|---|---|
| `faces` | One row per **unique visitor**. `embedding` holds the first (registration) embedding as `float32` bytes |
| `face_embeddings` | All templates per visitor. This is what the matcher loads at start-up. Removed with its face (`ON DELETE CASCADE`) |
| `visits` | One row per ENTRY→EXIT pair; `ACTIVE` until the EXIT completes it |
| `events` | One `ENTRY` or `EXIT` row per visit boundary, with real event time and saved image path |

> The `event_type` enum also lists `REGISTRATION` and `RECOGNITION`, but the code inserts only `ENTRY` and `EXIT` into MySQL. Registration and recognition events go to `logs/events.log` only.

<details>
<summary><b>Full SQL schema</b> (from <code>schema.sql</code>, MySQL 8.0 dump, version-specific comments omitted)</summary>

```sql
CREATE TABLE `faces` (
  `face_id` varchar(50) NOT NULL,
  `embedding` blob NOT NULL,
  `first_seen` datetime DEFAULT CURRENT_TIMESTAMP,
  `last_seen` datetime DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`face_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE `face_embeddings` (
  `embedding_id` int NOT NULL AUTO_INCREMENT,
  `face_id` varchar(50) NOT NULL,
  `embedding` blob NOT NULL,
  `created_at` datetime DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`embedding_id`),
  KEY `face_id` (`face_id`),
  CONSTRAINT `face_embeddings_ibfk_1` FOREIGN KEY (`face_id`) REFERENCES `faces` (`face_id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE `visits` (
  `visit_id` int NOT NULL AUTO_INCREMENT,
  `face_id` varchar(50) NOT NULL,
  `entry_time` datetime DEFAULT CURRENT_TIMESTAMP,
  `exit_time` datetime DEFAULT NULL,
  `status` enum('ACTIVE','COMPLETED') DEFAULT 'ACTIVE',
  PRIMARY KEY (`visit_id`),
  KEY `face_id` (`face_id`),
  CONSTRAINT `visits_ibfk_1` FOREIGN KEY (`face_id`) REFERENCES `faces` (`face_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE `events` (
  `event_id` int NOT NULL AUTO_INCREMENT,
  `visit_id` int DEFAULT NULL,
  `face_id` varchar(50) NOT NULL,
  `event_type` enum('REGISTRATION','RECOGNITION','ENTRY','EXIT') NOT NULL,
  `event_time` datetime DEFAULT CURRENT_TIMESTAMP,
  `image_path` varchar(255) DEFAULT NULL,
  PRIMARY KEY (`event_id`),
  KEY `face_id` (`face_id`),
  KEY `visit_id` (`visit_id`),
  CONSTRAINT `events_ibfk_1` FOREIGN KEY (`face_id`) REFERENCES `faces` (`face_id`),
  CONSTRAINT `events_ibfk_2` FOREIGN KEY (`visit_id`) REFERENCES `visits` (`visit_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
```

</details>

---

## 📂 Project Structure

```text
face_tracker/
├── README.md
├── config.json                  # boundary / processing / identity / camera settings
├── schema.sql                   # MySQL dump: faces, face_embeddings, visits, events
├── verify_integrity.sql         # manual SQL checks after a live test
├── streamlit_app.py             # Streamlit dashboard (7 pages)
├── .gitignore
│
├── src/
│   ├── video_processor.py       # main pipeline and orchestration
│   ├── face_matcher.py          # embedding matching, Known / Uncertain / New
│   ├── face_recognition.py      # InsightFace (buffalo_l) loader
│   ├── tracker.py               # YOLO face detection + ByteTrack
│   ├── boundary_counter.py      # four-boundary ENTRY/EXIT with hysteresis
│   ├── video_source.py          # file / RTSP input with auto-reconnect
│   ├── database.py              # MySQL read/write, visit transactions
│   ├── db_queries.py            # read-only queries for the dashboard
│   ├── event_logger.py          # events.log + staged/committed event images
│   ├── visit_manager.py         # (not used by the pipeline)
│   └── image_storage.py         # (not used by the pipeline)
│
├── tests/
│   └── test_visit_logic.py      # automated scenario tests
├── test_*.py                    # 8 manual smoke scripts (see Testing)
├── database/
│   └── face_tracker_schema.sql  # older schema, lacks face_embeddings
└── docs/
    └── streamlit_dashboard.md   # earlier dashboard notes
```

Created at run time or supplied by you (not in the repo): `.env`, `models/yolov8n-face.pt`, `data/sample.mp4`, `data/registered_faces/`, `data/uploads/`, `logs/`.

### Core modules

| File | Responsibility |
|---|---|
| **`src/video_processor.py`** | The orchestrator (~2,500 lines). Loads config, runs the frame loop, applies the quality gate, manages pending identity groups, registers new people, applies ENTRY/EXIT pairing, handles track expiry, closes open visits. Entry point: `python -m src.video_processor` |
| **`src/face_matcher.py`** | `FaceMatcher`: registry `FACE_ID → [embeddings]`; normalises, scores each identity by its best template, classifies `KNOWN` / `UNCERTAIN` / `NEW_CANDIDATE`, allocates IDs, adds templates (rejecting near-duplicates) |
| **`src/video_source.py`** | `cv2.VideoCapture` for files; for stream URLs a `StreamReader` with a background thread, newest-frame-only buffering, RTSP over TCP and auto-reconnect |
| **`src/tracker.py`** | `FaceTracker`: YOLO face model + ByteTrack (`persist=True`), returns `{track_id, bbox, confidence}` |
| **`src/face_recognition.py`** | `FaceRecognizer`: loads InsightFace `buffalo_l` on CPU. The pipeline uses its `.app`; `get_embedding()` is used only by test scripts |
| **`src/boundary_counter.py`** | `BoundaryCounter`: inside/outside state machine with hysteresis; returns `ENTRY`, `EXIT`, `INITIAL_INSIDE` or `None`; expires stale tracks |
| **`src/database.py`** | `Database`: connection, embedding storage, transactional `start_visit()` / `complete_visit()`, unique visitor count. Migrates legacy `faces.embedding` into `face_embeddings` on connect |
| **`src/db_queries.py`** | **Read-only** dashboard queries (KPIs, daily counts, visits, events, integrity summary, similar-identity finder) |
| **`src/event_logger.py`** | `events.log` plus the stage → database → commit / discard image sequence; unidentified-crossing evidence |
| **`streamlit_app.py`** | The dashboard; launches the backend as a subprocess for uploaded videos |
| **`config.json`** / **`schema.sql`** | Settings / database schema |
| `src/visit_manager.py` ⚠️ | **Not imported anywhere.** Superseded by `Database.start_visit()` / `complete_visit()` |
| `src/image_storage.py` ⚠️ | **Not imported anywhere.** The pipeline saves images through `EventLogger` |

---

## 🧪 Testing

### Automated: `tests/test_visit_logic.py`
Assert-based scenarios (a plain script, not pytest functions). It **stubs** MySQL, dotenv, YOLO and InsightFace and uses a `FakeDB`, so it needs **no database, models or video**:

```bat
python tests\test_visit_logic.py
```
Expected final line: `ALL TESTS PASSED`.

| # | Scenario verified |
|---|---|
| 1 | Person already inside at start gets a **paired** ENTRY and EXIT |
| 2 | EXIT with no visit is **suppressed** |
| 3 | Crossing before identification is held and recorded with its **original time** |
| 4 / 4b | A ByteTrack **ID change** creates no duplicate visit, and the new ID can close the visit |
| 5 | A **never-identified** crossing is saved as unidentified evidence, with no DB record |
| 6 | Track lost while inside gets an EXIT at its **last-seen time** |
| 7 | Video end **closes all open visits** |
| 8 | **Database failure** leaves no ENTRY record, image or staging file |
| 9 | Existing ACTIVE database visit handled without an orphan image |
| 10 | **Dry-run** pairs and logs events without database writes |
| 11 | **Grey-zone / hysteresis** boundary behaviour |

<details>
<summary><b>Manual smoke scripts in the project root</b></summary>

These print results rather than assert them. Several need files that are **not in the repository** (`data/test_face.png`, `data/face_1.png` … `face_3.png` are git-ignored) and/or MySQL.

| Script | What it checks | Needs |
|---|---|---|
| `test_yolo.py` | YOLO face detection preview on a video | `models/yolov8n-face.pt`, `data/sample.mp4` |
| `test_embedding.py` | InsightFace returns a 512-d embedding | `data/test_face.png` |
| `test_similarity.py` | Cosine similarity of an image vs a brightened copy | `data/test_face.png` |
| `test_matcher.py` | `FaceMatcher` register / match with random vectors | nothing |
| `test_face_recognition.py` | Same image twice: first registers, second is recognised | `data/test_face.png` |
| `test_multiple_faces.py` | Three images through recognise-or-register | `data/face_1..3.png` |
| `test_database_faces.py` | Load stored faces from MySQL into the matcher | MySQL |
| `test_save_face.py` | Match or register one image and save to MySQL | MySQL, `data/face_1.png` |

</details>

There are no automated tests for registration logic, `FaceMatcher` classification, `Database` transactions or `video_source.py`, and no accuracy evaluation exists in the repository.

---

## 💡 Engineering Decisions

Design choices visible in the code, and why each helps. **No accuracy figures are claimed**: the repository has no labelled evaluation.

| Decision | Problem addressed | Why it should help reliability |
|---|---|---|
| **Multi-frame validation** instead of single-frame registration | One blurry frame could create a bogus identity | Registration needs 3 consistent observations (or a strict 2-observation fallback) |
| **Three-state classification** (Known / Uncertain / New) | A binary match forces a guess in the ambiguous middle | The 0.30–0.45 band and margin check send ambiguous faces to *Uncertain*, which does not register |
| **Multiple embedding templates** (up to 10) | One embedding cannot cover every angle or lighting | Best-of-several matching; near-duplicates and weak additions are rejected |
| **Face-angle filtering** for registration | Profile views score low against the same person's frontal view | Frontal-only registration lowers the chance of two IDs for one person |
| **Track-based identity continuity** | ByteTrack IDs change after occlusion | Identity is tied to the Face ID, and groups merge observations across track IDs |
| **Boundary hysteresis** | A face at an edge would flip inside/outside repeatedly | State changes only after a clear crossing |
| **Duplicate-event suppression** | ID switches would create extra ENTRYs or orphan EXITs | One open visit per Face ID; an EXIT needs an ENTRY |
| **Stage → database → commit** image writes | A crash or DB error could leave images/log lines for events that never happened | Images and log lines exist only for events MySQL accepted |
| **Delayed identification** with original timestamps | People often cross before their face is confirmed | Crossings are queued and replayed with their real time |
| **Cheap quality gate before InsightFace** | The heavy model on every face is slow and noisy | Poor crops are rejected first; InsightFace runs lazily, at most once per frame |

---

## ⚠️ Limitations and Future Improvements

### Limitations
- **No measured accuracy.** Thresholds (0.45 / 0.30 / margin 0.05 / group 0.82 …) are fixed in `video_processor.py`, and the code calls the margin *provisional*. There is no precision/recall report.
- **Reproducibility.** No `requirements.txt` or `.env.example`; YOLO weights, sample video and test images are not in the repo.
- **Unique visitors = registered faces.** A person registered but never crossing the boundary still counts, and a person registered twice inflates the count. There is no merge tool.
- **Event times are processing wall-clock times**, not video timestamps, for recorded files.
- **Face-only identification.** Faces smaller than 60 px, blurry, dark or heavily turned may never be identified. Those crossings are saved as unidentified with no DB record.
- **Camera-specific boundary.** "Inside" is an image rectangle judged by the face-box centre, not a physical doorway or directional line.
- **CPU-only InsightFace**, single camera, single process.
- **`REGISTRATION` / `RECOGNITION`** are in the `events` enum but never written to MySQL.
- **Unused modules:** `visit_manager.py` and `image_storage.py`.
- **Narrow test coverage:** only the visit/event logic is automated.
- **Live RTSP is not demonstrated** in the repository.
- **Biometric data** (embeddings and face crops) is stored with no retention or consent mechanism.

### Future improvements
- Add a pinned `requirements.txt`, a `.env.example`, and optionally Docker; document where to get the YOLO weights.
- **Calibrate thresholds** on a labelled multi-camera set and report precision, false-merge and false-split rates; move them into `config.json`.
- **Validate live RTSP** end to end (long run, forced network drop and reconnect) and add a mocked `StreamReader` test.
- Use the video's own timestamps for recorded files.
- Add pytest tests for `FaceMatcher`, `BoundaryCounter`, pending-group logic and `Database` against a test MySQL.
- Add an **identity merge / delete** tool to the dashboard, driven by the Verification similarity pairs.
- Add GPU execution for InsightFace and YOLO.
- Support direction-aware line crossing or a configurable polygon.
- Add a data-retention policy and a purge option.
- Remove or wire in `visit_manager.py` and `image_storage.py`.

---

<div align="center">

**▶ [Watch the project demo on Loom](https://www.loom.com/share/9f6458a8ad5846448371ffd2f75f8067)**

</div>
