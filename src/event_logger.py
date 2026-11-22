import os
import cv2
import logging

from datetime import datetime


class EventLogger:
    """
    Writes events.log and saves cropped event images.

    Consistency model (ENTRY / EXIT):
        1. stage_event_image()   -> writes the crop to logs/_staging/
                                    and returns the FINAL path (to store in MySQL)
        2. database write        -> done by the caller
        3a. commit_event_image() -> success: move to final folder, write events.log
        3b. discard_staged_image() -> failure: delete the staged file

    So a final image and an ENTRY/EXIT line in events.log only exist when
    the database (or the in-memory visit check in dry-run) accepted the event.
    """

    def __init__(self, log_folder="logs"):

        self.log_folder = log_folder
        self.staging_folder = os.path.join(self.log_folder, "_staging")

        os.makedirs(self.log_folder, exist_ok=True)
        os.makedirs(self.staging_folder, exist_ok=True)

        self.logger = logging.getLogger("FaceTrackerEvents")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        # Prevent duplicate file handlers.
        if not self.logger.handlers:

            log_path = os.path.join(
                self.log_folder,
                "events.log"
            )

            handler = logging.FileHandler(
                log_path,
                encoding="utf-8"
            )

            formatter = logging.Formatter(
                "%(message)s"
            )

            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    # --------------------------------------------------
    # TEXT LOG
    # --------------------------------------------------

    def log_event(
        self,
        event_type,
        face_id=None,
        track_id=None,
        message="",
        event_time=None
    ):
        """
        event_time: when the event REALLY happened. If it differs from the
        write time (delayed identification) it is added as "AT: ...".
        """

        timestamp = datetime.now().astimezone().isoformat(
            timespec="milliseconds"
        )

        at_part = ""
        if event_time is not None:
            at_part = f"AT: {event_time.isoformat(timespec='milliseconds')} | "

        self.logger.info(
            f"{timestamp} | "
            f"EVENT: {event_type} | "
            f"FACE: {face_id or 'UNKNOWN'} | "
            f"TRACK: {track_id if track_id is not None else 'UNKNOWN'} | "
            f"{at_part}"
            f"{message}"
        )

    # --------------------------------------------------
    # CROPPING
    # --------------------------------------------------

    @staticmethod
    def crop_from_frame(frame, bbox):
        """Return an independent copy of the bbox region, or None."""

        if frame is None or frame.size == 0:
            return None

        height, width = frame.shape[:2]

        x1, y1, x2, y2 = map(int, bbox)

        x1 = max(0, min(x1, width))
        x2 = max(0, min(x2, width))
        y1 = max(0, min(y1, height))
        y2 = max(0, min(y2, height))

        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            return None

        return crop.copy()

    # --------------------------------------------------
    # ENTRY / EXIT IMAGES (two-phase)
    # --------------------------------------------------

    def _build_paths(self, face_id, track_id, event_type, event_time):

        when = event_time or datetime.now()

        folder = os.path.join(
            self.log_folder,
            "entries" if event_type == "ENTRY" else "exits",
            when.strftime("%Y-%m-%d")
        )

        filename = (
            f"{face_id}_{event_type}_"
            f"{when.strftime('%H%M%S_%f')}_"
            f"track_{track_id}.jpg"
        )

        final_path = os.path.join(folder, filename)
        staged_path = os.path.join(self.staging_folder, filename)

        return final_path, staged_path

    def stage_event_image(
        self,
        crop,
        face_id,
        track_id,
        event_type,
        event_time=None
    ):
        """
        Write the crop to the staging folder.

        Returns (final_path, staged_path) or None on failure.
        Nothing is logged to events.log here.
        """

        if event_type not in ("ENTRY", "EXIT"):
            raise ValueError("Invalid event type")

        if not face_id:
            return None

        if crop is None or crop.size == 0:
            return None

        final_path, staged_path = self._build_paths(
            face_id, track_id, event_type, event_time
        )

        os.makedirs(self.staging_folder, exist_ok=True)

        if not cv2.imwrite(staged_path, crop):
            self.log_event(
                "IMAGE_SAVE_FAILED",
                face_id,
                track_id,
                f"Could not stage {event_type} image"
            )
            return None

        return final_path, staged_path

    def commit_event_image(
        self,
        final_path,
        staged_path,
        face_id,
        track_id,
        event_type,
        event_time=None,
        message=""
    ):
        """Move the staged image to its final folder and write events.log."""

        try:
            os.makedirs(os.path.dirname(final_path), exist_ok=True)
            os.replace(staged_path, final_path)
        except OSError as error:
            self.log_event(
                "IMAGE_FINALIZE_FAILED",
                face_id,
                track_id,
                f"{event_type}: {error}"
            )
            return False

        self.log_event(
            event_type,
            face_id,
            track_id,
            f"Image: {final_path}" + (f" | {message}" if message else ""),
            event_time=event_time
        )

        print(f"{event_type} IMAGE SAVED: {final_path}")

        return True

    def discard_staged_image(self, staged_path):
        """Delete a staged image after a rejected / failed event."""

        try:
            if staged_path and os.path.exists(staged_path):
                os.remove(staged_path)
        except OSError:
            pass

    # --------------------------------------------------
    # UNIDENTIFIED CROSSINGS (never silently dropped)
    # --------------------------------------------------

    def save_unidentified_image(
        self,
        crop,
        track_id,
        event_type,
        event_time=None
    ):
        """
        Keep evidence for a crossing whose identity was never confirmed.
        Saved under logs/unidentified/YYYY-MM-DD/. No database record.
        """

        if crop is None or crop.size == 0:
            return None

        when = event_time or datetime.now()

        folder = os.path.join(
            self.log_folder,
            "unidentified",
            when.strftime("%Y-%m-%d")
        )

        os.makedirs(folder, exist_ok=True)

        path = os.path.join(
            folder,
            f"UNIDENTIFIED_{event_type}_{when.strftime('%H%M%S_%f')}"
            f"_track_{track_id}.jpg"
        )

        if not cv2.imwrite(path, crop):
            return None

        return path

    # --------------------------------------------------
    # LEGACY ONE-STEP API (kept for compatibility)
    # --------------------------------------------------

    def save_event_image(
        self,
        frame,
        bbox,
        face_id,
        track_id,
        event_type
    ):
        """Old behaviour: crop, save and log immediately (no DB check)."""

        crop = self.crop_from_frame(frame, bbox)

        staged = self.stage_event_image(
            crop, face_id, track_id, event_type
        )

        if staged is None:
            return None

        final_path, staged_path = staged

        if not self.commit_event_image(
            final_path, staged_path, face_id, track_id, event_type
        ):
            return None

        return final_path
