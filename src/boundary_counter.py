class BoundaryCounter:
    """
    Resolution-independent four-boundary counter with hysteresis.

    Returned events from update():
        "ENTRY"          Outside (or grey zone) -> clearly inside
        "EXIT"           Inside -> clearly outside
        "INITIAL_INSIDE" Track is FIRST seen already clearly inside
                         (person present when the video started, or a
                         brand-new ByteTrack ID for someone already inside).
                         The caller decides whether this is a real ENTRY or
                         a duplicate of an existing visit.
        None             No event.

    Changes from the previous version:
      * First observation inside the area is no longer silent
        (it returns "INITIAL_INSIDE"), so a later EXIT always has a visit.
      * A track first seen in the grey margin zone that later moves clearly
        inside now returns "ENTRY" (previously lost, because old_state was None).
      * cleanup() now RETURNS the expired tracks so the caller can close
        visits for people whose track disappeared while inside.
      * Helper accessors: is_inside(), last_seen(), pop_all().
    """

    def __init__(
        self,
        margin_x_ratio=0.08,
        margin_y_ratio=0.08,
        hysteresis_ratio=0.02,
        max_track_gap=90,
        # Backward-compatible fixed-pixel arguments:
        margin=None,
        hysteresis=None,
        report_initial_inside=True
    ):
        self.margin_x_ratio = float(margin_x_ratio)
        self.margin_y_ratio = float(margin_y_ratio)
        self.hysteresis_ratio = float(hysteresis_ratio)
        self.max_track_gap = int(max_track_gap)
        self.report_initial_inside = bool(report_initial_inside)

        # Preserve compatibility with older code that passes margin= / hysteresis=.
        self.fixed_margin = margin
        self.fixed_hysteresis = hysteresis

        for name, value in (
            ("margin_x_ratio", self.margin_x_ratio),
            ("margin_y_ratio", self.margin_y_ratio),
            ("hysteresis_ratio", self.hysteresis_ratio)
        ):
            if not 0 <= value < 0.5:
                raise ValueError(f"{name} must be between 0 and 0.5.")

        if self.max_track_gap < 1:
            raise ValueError("max_track_gap must be at least 1.")

        self.track_states = {}

    def get_bounds(self, frame_width, frame_height, boundary="inner"):
        """Return (left, top, right, bottom) for the selected boundary."""
        if frame_width <= 0 or frame_height <= 0:
            raise ValueError("Frame width and height must be positive.")

        if self.fixed_margin is not None:
            margin_x = margin_y = int(self.fixed_margin)
        else:
            margin_x = int(round(frame_width * self.margin_x_ratio))
            margin_y = int(round(frame_height * self.margin_y_ratio))

        if self.fixed_hysteresis is not None:
            hysteresis_x = hysteresis_y = int(self.fixed_hysteresis)
        else:
            hysteresis_x = int(round(frame_width * self.hysteresis_ratio))
            hysteresis_y = int(round(frame_height * self.hysteresis_ratio))

        if boundary == "inner":
            left = margin_x + hysteresis_x
            top = margin_y + hysteresis_y
            right = frame_width - left
            bottom = frame_height - top
        elif boundary == "outer":
            left = max(0, margin_x - hysteresis_x)
            top = max(0, margin_y - hysteresis_y)
            right = frame_width - left
            bottom = frame_height - top
        else:
            raise ValueError("boundary must be 'inner' or 'outer'.")

        return left, top, right, bottom

    def update(
        self,
        track_id,
        center_x,
        center_y,
        frame_width,
        frame_height,
        frame_number
    ):
        inner_left, inner_top, inner_right, inner_bottom = self.get_bounds(
            frame_width, frame_height, boundary="inner"
        )
        outer_left, outer_top, outer_right, outer_bottom = self.get_bounds(
            frame_width, frame_height, boundary="outer"
        )

        clearly_inside = (
            inner_left <= center_x <= inner_right
            and inner_top <= center_y <= inner_bottom
        )

        clearly_outside = (
            center_x < outer_left
            or center_x > outer_right
            or center_y < outer_top
            or center_y > outer_bottom
        )

        previous = self.track_states.get(track_id)

        # ---------------------------------------------
        # First observation of this track
        # ---------------------------------------------
        if previous is None:
            if clearly_inside:
                inside = True
            elif clearly_outside:
                inside = False
            else:
                inside = None  # grey zone: undecided

            self.track_states[track_id] = {
                "inside": inside,
                "last_seen": frame_number,
                "first_seen": frame_number
            }

            if inside is True and self.report_initial_inside:
                return "INITIAL_INSIDE"

            return None

        # ---------------------------------------------
        # Existing track
        # ---------------------------------------------
        old_state = previous["inside"]
        new_state = old_state

        if clearly_inside:
            new_state = True
        elif clearly_outside:
            new_state = False

        previous["inside"] = new_state
        previous["last_seen"] = frame_number

        # False OR undecided (None) -> inside is an ENTRY.
        if old_state is not True and new_state is True:
            return "ENTRY"

        # Only a track that was known to be inside can EXIT.
        if old_state is True and new_state is False:
            return "EXIT"

        return None

    # --------------------------------------------------
    # ACCESSORS
    # --------------------------------------------------

    def is_inside(self, track_id):
        """True / False / None (unknown or grey zone)."""
        state = self.track_states.get(track_id)
        return None if state is None else state["inside"]

    def last_seen(self, track_id):
        state = self.track_states.get(track_id)
        return None if state is None else state["last_seen"]

    # --------------------------------------------------
    # CLEANUP
    # --------------------------------------------------

    def cleanup(self, frame_number):
        """
        Remove tracks not seen for more than max_track_gap frames.

        Returns [(track_id, state_dict), ...] for the removed tracks so the
        caller can resolve pending events and close visits.
        """
        expired = [
            (track_id, dict(state))
            for track_id, state in self.track_states.items()
            if frame_number - state["last_seen"] > self.max_track_gap
        ]

        for track_id, _ in expired:
            del self.track_states[track_id]

        return expired

    def pop_all(self):
        """Remove and return every track (used at shutdown)."""
        everything = [
            (track_id, dict(state))
            for track_id, state in self.track_states.items()
        ]
        self.track_states.clear()
        return everything
