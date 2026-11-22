# Streamlit Dashboard

A read-only web frontend for the face detection system.  
It never writes to the database or modifies any backend files.

## Running

```bash
# from the project root (D:\face_detection)
streamlit run streamlit_app.py
```

The app opens at `http://localhost:8501` by default.

## Dependencies

Install once into the project's virtual environment:

```bash
pip install streamlit pandas
```

All other imports (`mysql-connector-python`, `python-dotenv`) are already
required by the existing backend.

## Pages

| Page | Description |
|---|---|
| Overview | KPI cards (unique visitors, active visits, entries/exits today) + daily trend chart + face gallery preview. Falls back to `sample_output/` when MySQL is unreachable. |
| Visit Log | Paginated, filterable table of visits queried from MySQL. |
| Event Log | Browse live MySQL events **or** tail `logs/events.log` / `sample_output/events.log` with keyword filtering. |
| Registered Faces | Gallery of face crops from `data/registered_faces/`. Expandable per-identity crop viewer. |
| Sample Output | Entry/exit images and the events log from `sample_output/`. |

## File Layout

```
D:\face_detection\
├── streamlit_app.py          ← Streamlit entry point  (NEW)
├── src\
│   └── db_queries.py         ← Read-only DB helpers   (NEW)
└── docs\
    └── streamlit_dashboard.md  ← This file             (NEW)
```

## Architecture Notes

- `src/db_queries.py` opens its own short-lived MySQL connection per call
  and never shares a connection with the video processor.
- `streamlit_app.py` uses `@st.cache_data(ttl=30)` so the DB is polled at
  most once every 30 seconds, not on every UI interaction.
- The dashboard imports nothing from `video_processor.py` or any module that
  loads YOLO / InsightFace, so it starts instantly without GPU dependencies.

## Credentials

The dashboard reads the same `.env` file as the backend:

```
DB_HOST=...
DB_USER=...
DB_PASSWORD=...
DB_NAME=...
```

Never commit `.env` to version control (already in `.gitignore`).
