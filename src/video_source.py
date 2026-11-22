"""
Video input for the face tracker.

open_video_source(source) returns an object with the same methods the main
loop already uses on cv2.VideoCapture:

    isOpened()  grab()  retrieve()  release()

* Video FILE   -> a plain cv2.VideoCapture (behaviour unchanged).
* Live STREAM  -> StreamReader:
      - a background thread reads the camera continuously and keeps ONLY the
        newest frame, so slow analysis never builds up a delay;
      - if the stream drops or stalls, it reconnects automatically
        (every `retry_delay` seconds, for up to `max_downtime` seconds);
      - TCP transport is requested for RTSP (fewer corrupted frames than UDP).
"""

import os
import threading
import time

import cv2

STREAM_PREFIXES = ("rtsp://", "rtsps://", "rtmp://", "http://", "https://")


def is_stream(source):
    return str(source).strip().lower().startswith(STREAM_PREFIXES)


def open_video_source(
    source,
    on_event=None,
    retry_delay=2.0,
    max_downtime=60.0,
    should_stop=None
):
    """Open a file with cv2, or a live stream with automatic reconnect."""

    if is_stream(source):
        return StreamReader(
            str(source).strip(),
            on_event=on_event,
            retry_delay=retry_delay,
            max_downtime=max_downtime,
            should_stop=should_stop
        )

    return cv2.VideoCapture(source)


class StreamReader:
    """cv2.VideoCapture-compatible reader for RTSP / network streams."""

    # Generous: after (re)connecting, a camera may need a few seconds to send
    # its next keyframe before the first frame can be decoded.
    OPEN_TIMEOUT_MS = 10000
    READ_TIMEOUT_MS = 15000

    def __init__(
        self,
        url,
        on_event=None,
        retry_delay=2.0,
        max_downtime=60.0,
        should_stop=None
    ):
        self.url = url
        # Optional callable (e.g. the dashboard's stop-file check). grab()
        # returns False promptly when it says True, even if the camera is down.
        self.should_stop = should_stop or (lambda: False)
        self.on_event = on_event or (lambda name, message: None)
        self.retry_delay = float(retry_delay)
        self.max_downtime = float(max_downtime)

        self._lock = threading.Condition()
        self._frame = None
        self._frame_id = 0          # increases for every new frame stored
        self._delivered_id = 0      # last frame id handed to the caller
        self._stop = threading.Event()
        self._given_up = False
        self._connected = False
        self._ever_connected = False
        self._capture = None

        # RTSP over TCP, no internal buffering (lower latency).
        # Must be set before the first VideoCapture is created.
        os.environ.setdefault(
            "OPENCV_FFMPEG_CAPTURE_OPTIONS",
            "rtsp_transport;tcp|fflags;nobuffer"
        )

        # Try the first connection here so the caller learns immediately
        # whether the URL works.
        self._capture = self._open_capture()
        self._connected = self._capture is not None and self._capture.isOpened()
        self._ever_connected = self._connected

        self._thread = threading.Thread(
            target=self._reader_loop,
            name="stream-reader",
            daemon=True
        )

        if not self._connected:
            self.on_event(
                "STREAM_WAITING",
                "Camera not reachable yet; will keep retrying"
            )

        # Always start the reader: if the camera is not up yet it keeps
        # retrying until max_downtime instead of exiting immediately.
        self._thread.start()

    # --------------------------------------------------
    # connection helpers
    # --------------------------------------------------

    def _open_capture(self):
        try:
            capture = cv2.VideoCapture(
                self.url,
                cv2.CAP_FFMPEG,
                [
                    cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self.OPEN_TIMEOUT_MS,
                    cv2.CAP_PROP_READ_TIMEOUT_MSEC, self.READ_TIMEOUT_MS
                ]
            )
        except (TypeError, cv2.error):
            # Older OpenCV builds do not accept the parameter list.
            capture = cv2.VideoCapture(self.url)

        return capture

    def _reader_loop(self):
        down_since = None

        while not self._stop.is_set():

            if self._capture is None or not self._capture.isOpened():

                if down_since is None:
                    down_since = time.time()

                if time.time() - down_since > self.max_downtime:
                    self.on_event(
                        "STREAM_GIVE_UP",
                        f"No stream for {self.max_downtime:.0f}s; stopping"
                    )
                    break

                self._stop.wait(self.retry_delay)

                if self._stop.is_set():
                    break

                self._release_capture()
                self._capture = self._open_capture()

                if self._capture.isOpened():
                    self._connected = True
                    down_since = None
                    self.on_event(
                        "STREAM_RECONNECTED",
                        "Camera stream is back; processing resumes"
                    )
                continue

            ok, frame = self._capture.read()

            if ok and frame is not None:
                down_since = None
                with self._lock:
                    self._frame = frame
                    self._frame_id += 1
                    self._lock.notify_all()
                continue

            # read failed: stream dropped or stalled
            if self._connected:
                self._connected = False
                self.on_event(
                    "STREAM_DISCONNECTED",
                    "No frame received; trying to reconnect"
                )

            down_since = down_since or time.time()
            self._release_capture()

        with self._lock:
            self._given_up = True
            self._lock.notify_all()

    def _release_capture(self):
        try:
            if self._capture is not None:
                self._capture.release()
        except Exception:
            pass
        self._capture = None

    # --------------------------------------------------
    # cv2.VideoCapture-style interface
    # --------------------------------------------------

    def isOpened(self):
        # A stream is considered "open" even before the first connection;
        # grab() returns False only after max_downtime without any frames.
        return True

    def grab(self):
        """
        Wait for a frame newer than the last one handed out.
        Returns False only when the stream has been gone for max_downtime
        (or release() was called).
        """
        with self._lock:
            while (
                self._frame_id == self._delivered_id
                and not self._given_up
                and not self._stop.is_set()
            ):
                self._lock.wait(timeout=0.5)

                if self.should_stop():
                    return False

            if self._frame_id == self._delivered_id:
                return False

            self._delivered_id = self._frame_id
            return True

    def retrieve(self):
        with self._lock:
            if self._frame is None:
                return False, None
            return True, self._frame.copy()

    def release(self):
        self._stop.set()

        with self._lock:
            self._lock.notify_all()

        if self._thread.is_alive():
            self._thread.join(timeout=3)

        self._release_capture()
