import cv2
import os
import json
import numpy as np

from datetime import datetime

from src.video_source import open_video_source
from src.tracker import FaceTracker
from src.face_recognition import FaceRecognizer
from src.face_matcher import FaceMatcher
from src.database import Database
from src.boundary_counter import BoundaryCounter
from src.event_logger import EventLogger


# True = no MySQL writes (safe test). False = live mode, writes to MySQL.
DRY_RUN = False

# Optional hook used by the Streamlit dashboard. Unset = old behaviour.
STOP_FILE = os.environ.get("FACE_TRACKER_STOP_FILE")


class VideoProcessor:

    def __init__(self):

        print("Initializing Video Processor...")

        self.tracker = FaceTracker()
        self.recognizer = FaceRecognizer()
        self.database = Database()

        # Load camera-independent settings from the project-root config.json.
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json")
        with open(config_path, "r", encoding="utf-8") as config_file:
            self.config = json.load(config_file)

        boundary_config = self.config.get("boundary", {})

        # Dynamic boundaries are calculated from each input video's dimensions.
        self.boundary_counter = BoundaryCounter(
            margin_x_ratio=float(boundary_config.get("margin_x_ratio", 0.08)),
            margin_y_ratio=float(boundary_config.get("margin_y_ratio", 0.08)),
            hysteresis_ratio=float(boundary_config.get("hysteresis_ratio", 0.02)),
            max_track_gap=int(boundary_config.get("max_track_gap", 90)),
            report_initial_inside=bool(
                boundary_config.get("report_initial_inside", True)
            )
        )

        # Same person under a new ByteTrack ID: tracks seen within this many
        # REAL frames are treated as still present.
        self.duplicate_track_window = int(
            boundary_config.get("duplicate_track_window", 15)
        )

        # Processing options (frame_skip is now actually used).
        processing_config = self.config.get("processing", {})
        self.frame_skip = max(0, int(processing_config.get("frame_skip", 0)))
        self.show_window = bool(processing_config.get("show_window", True))
        self.display_max_width = int(
            processing_config.get("display_max_width", 1280)
        )
        # True: people still inside when the video ends get an EXIT.
        self.close_visits_on_end = bool(
            processing_config.get("close_visits_on_end", True)
        )

        # Identity settings (optional "identity" block in config.json).
        identity_config = self.config.get("identity", {})
        # A NEW person is only registered from a face turned at most this
        # many degrees left/right (profile views make duplicate identities).
        self.max_register_yaw = float(identity_config.get("max_register_yaw", 35))
        # An already-identified track learns a new angle (extra template)
        # when its best match is this similar and this far ahead of #2.
        self.template_min_similarity = float(
            identity_config.get("template_min_similarity", 0.50)
        )
        self.template_min_margin = float(
            identity_config.get("template_min_margin", 0.08)
        )
        # A group stuck in the "uncertain" similarity band (above
        # uncertain_threshold, below the match threshold) is registered as a
        # NEW person once it has this many observations AND its best-matching
        # identity is on screen right now on another track (one person cannot
        # be in two places at once). 0 turns this rule off.
        self.uncertain_resolve_observations = int(
            identity_config.get("uncertain_resolve_observations", 7)
        )
        # SECOND way to resolve an uncertain group as a NEW person, for the
        # common case where the best-matching identity is NOT on screen at
        # the same time. Every condition below must hold (see
        # _sustained_new_person_evidence). 0 turns this rule off.
        self.uncertain_new_min_observations = int(
            identity_config.get("uncertain_new_min_observations", 5)
        )
        # Group average must stay below this against EVERY registered face
        # (the match threshold is 0.45; this keeps a safety gap below it).
        self.uncertain_new_max_similarity = float(
            identity_config.get("uncertain_new_max_similarity", 0.40)
        )
        # Mean similarity of the group's frontal observations to their own
        # average: proves the observations really are one person.
        self.uncertain_new_min_coherence = float(
            identity_config.get("uncertain_new_min_coherence", 0.60)
        )
        # Frontal observations needed (profile views score low on purpose).
        self.uncertain_new_min_frontal = int(
            identity_config.get("uncertain_new_min_frontal", 3)
        )
        # A track that ends with only 2 observations may still register a
        # new person if both are high quality, frontal and agree strongly.
        self.two_observation_min_quality = float(
            identity_config.get("two_observation_min_quality", 0.40)
        )
        self.two_observation_min_agreement = float(
            identity_config.get("two_observation_min_agreement", 0.75)
        )
        # Size InsightFace works at. A 4K frame is shrunk to this before faces
        # are searched for, so small faces disappear at the default 640.
        # Larger finds more faces but is slower (1280 is a good next step).
        self.insightface_det_size = int(
            identity_config.get("insightface_det_size", 640)
        )
        if self.insightface_det_size != 640:
            self.recognizer.app.prepare(
                ctx_id=-1,
                det_size=(self.insightface_det_size, self.insightface_det_size)
            )

        self.dry_run = DRY_RUN
        
        # Initialize event and image logger
        self.event_logger = EventLogger(
            log_folder="logs"
        )
        print(
            "MODE:",
            "DRY RUN - database writes disabled"
            if self.dry_run
            else "LIVE - database writes enabled"
        )

        if not self.database.connect():
            raise RuntimeError("MySQL connection failed.")

        known_faces = self.database.get_all_embeddings()

        self.matcher = FaceMatcher(
            threshold=0.45,
            uncertain_threshold=0.30,
            minimum_margin=0.05,
            max_embeddings_per_face=10,
            duplicate_threshold=0.98
        )

        self.matcher.load_faces(known_faces)

        # Track ID -> confirmed Face ID
        self.track_to_face = {}

        # Face ID -> open visit for this run (permanent identity, not track ID)
        self.session_visits = {}

        # Track ID -> [crossings detected before the Face ID was known]
        self.pending_crossings = {}

        # Track ID -> (latest face crop, time) used for "track lost" exits
        self.track_last_crop = {}

        self.session_faces = set()

        self.stats = {
            "entries": 0,
            "exits": 0,
            "suppressed": 0,
            "unidentified": 0
        }

        # Track ID -> pending identity group ID
        self.track_to_group = {}

        # Pending groups contain observations from one
        # or more Track IDs believed to represent one person.
        self.pending_groups = {}

        self.next_group_id = 1

        self.last_embedding_frame = {}
        self.last_template_frame = {}

        # Track ID -> {rejection reason: count}. Used to explain tracks that
        # never produced an embedding.
        self.track_reject_reasons = {}
        self.track_stats = {}
        self.last_match_reason = None

        self.image_saved_for_track = set()

        # Avoid writing the same TRACKING / RECOGNITION /
        # EMBEDDING_GENERATED log message on every video frame.
        self.logged_track_ids = set()
        self.logged_recognition_tracks = set()
        self.logged_embedding_tracks = set()

        self.event_logger.log_event(
            "SYSTEM_START",
            message=f"Video processor initialized; dry_run={self.dry_run}"
        )

        self.image_folder = "data/registered_faces"
        if not self.dry_run:
            os.makedirs(self.image_folder, exist_ok=True)

        # Candidate confirmation settings
        self.required_observations = 3
        self.minimum_consistency = 0.70

        # Cross-track grouping uses stricter matching
        # than ordinary registered-face recognition.
        self.pending_group_threshold = 0.82
        self.pending_group_margin = 0.08

        # Pending groups expire if no new observation arrives.
        self.pending_group_timeout = 300

        # Embedding processing intervals
        self.pending_embedding_interval = 5
        self.known_embedding_interval = 15
        self.template_interval = 30

        # Face quality settings
        min_face_size = int(identity_config.get("min_face_size", 60))
        self.minimum_face_width = min_face_size
        self.minimum_face_height = min_face_size
        self.minimum_sharpness = 20
        self.minimum_confidence = 0.35

        print(
            f"InsightFace det_size: {self.insightface_det_size} | "
            f"minimum face size: {self.minimum_face_width}px"
        )

        print("Video Processor initialized successfully!")

    # --------------------------------------------------
    # FACE BOX UTILITIES
    # --------------------------------------------------

    @staticmethod
    def calculate_iou(box1, box2):

        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])

        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = (
            max(0, x2 - x1) *
            max(0, y2 - y1)
        )

        area1 = (
            max(0, box1[2] - box1[0]) *
            max(0, box1[3] - box1[1])
        )

        area2 = (
            max(0, box2[2] - box2[0]) *
            max(0, box2[3] - box2[1])
        )

        union = area1 + area2 - intersection

        if union <= 0:
            return 0.0

        return intersection / union

    @staticmethod
    def cosine_similarity(embedding1, embedding2):

        return FaceMatcher.cosine_similarity(
            embedding1,
            embedding2
        )

    def find_insightface_match(
        self,
        tracker_bbox,
        insight_faces
    ):

        best_face = None
        best_iou = 0.0
        self.last_match_reason = None

        if not insight_faces:
            self.last_match_reason = "insightface found no face in frame"
            return None

        for insight_face in insight_faces:

            bbox = insight_face.bbox

            insight_bbox = (
                int(bbox[0]),
                int(bbox[1]),
                int(bbox[2]),
                int(bbox[3])
            )

            iou = self.calculate_iou(
                tracker_bbox,
                insight_bbox
            )

            if iou > best_iou:
                best_iou = iou
                best_face = insight_face

        if best_face is None or best_iou < 0.10:
            self.last_match_reason = (
                f"insightface face not aligned with tracked box "
                f"(best IoU {best_iou:.2f} < 0.10)"
            )
            return None

        # InsightFace pose = [pitch, yaw, roll] in degrees (None if absent).
        pose = getattr(best_face, "pose", None)
        yaw = None

        if pose is not None and len(pose) >= 2:
            yaw = float(pose[1])

        embedding = self.matcher.normalize_embedding(
            best_face.normed_embedding
        )

        return embedding, yaw

    def find_insightface_embedding(
        self,
        tracker_bbox,
        insight_faces
    ):

        match = self.find_insightface_match(
            tracker_bbox,
            insight_faces
        )

        return None if match is None else match[0]

    # --------------------------------------------------
    # FACE QUALITY
    # --------------------------------------------------

    def assess_face_quality(
        self,
        frame,
        bbox,
        detection_confidence
    ):
        """(is_good, quality_score). Same contract as before."""

        is_good, quality_score, _ = self.assess_face_quality_detail(
            frame,
            bbox,
            detection_confidence
        )

        return is_good, quality_score

    def assess_face_quality_detail(
        self,
        frame,
        bbox,
        detection_confidence
    ):
        """(is_good, quality_score, rejection_reason). Filters unchanged."""

        if detection_confidence < self.minimum_confidence:
            return False, 0.0, (
                f"low detection confidence "
                f"({detection_confidence:.2f} < {self.minimum_confidence})"
            )

        x1, y1, x2, y2 = bbox
        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            return False, 0.0, "empty face crop"

        height, width = crop.shape[:2]

        if (
            width < self.minimum_face_width
            or height < self.minimum_face_height
        ):
            return False, 0.0, (
                f"face too small ({width}x{height}px, minimum "
                f"{self.minimum_face_width}x{self.minimum_face_height})"
            )

        gray = cv2.cvtColor(
            crop,
            cv2.COLOR_BGR2GRAY
        )

        sharpness = cv2.Laplacian(
            gray,
            cv2.CV_64F
        ).var()

        if sharpness < self.minimum_sharpness:
            return False, 0.0, (
                f"too blurry (sharpness {sharpness:.1f} < "
                f"{self.minimum_sharpness})"
            )

        brightness = float(np.mean(gray))

        if brightness < 30 or brightness > 225:
            return False, 0.0, (
                f"bad brightness ({brightness:.0f}, allowed 30-225)"
            )

        size_score = min(
            1.0,
            (width * height) / 40000.0
        )

        sharpness_score = min(
            1.0,
            sharpness / 100.0
        )

        confidence_score = float(
            detection_confidence
        )

        brightness_score = max(
            0.0,
            1.0 - abs(brightness - 127.5) / 127.5
        )

        quality_score = (
            0.30 * size_score
            + 0.30 * sharpness_score
            + 0.25 * confidence_score
            + 0.15 * brightness_score
        )

        return True, float(quality_score), ""

    def _note_reject(self, track_id, reason):
        """Count why a track's observation attempt produced no embedding."""

        if not reason:
            return

        counts = self.track_reject_reasons.setdefault(track_id, {})
        key = reason.split(" (")[0]
        counts[key] = counts.get(key, 0) + 1

    # --------------------------------------------------
    # IMAGE STORAGE
    # --------------------------------------------------

    def save_face_image(
        self,
        frame,
        bbox,
        face_id,
        track_id
    ):

        # Dry-run must not write face crops to disk.
        if self.dry_run:
            return False

        x1, y1, x2, y2 = bbox
        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            return False

        folder = os.path.join(
            self.image_folder,
            face_id
        )

        os.makedirs(folder, exist_ok=True)

        path = os.path.join(
            folder,
            f"track_{track_id}.jpg"
        )

        success = cv2.imwrite(path, crop)

        if success:
            print("Face image saved:", path)

        return success

    # --------------------------------------------------
    # EMBEDDING AGGREGATION
    # --------------------------------------------------

    def calculate_weighted_embedding(self, observations):

        if not observations:
            return None

        embeddings = [
            item["embedding"]
            for item in observations
        ]

        weights = [
            max(item["quality"], 0.01)
            for item in observations
        ]

        embeddings = np.stack(embeddings)

        weights = np.asarray(
            weights,
            dtype=np.float32
        )

        average = np.average(
            embeddings,
            axis=0,
            weights=weights
        )

        return self.matcher.normalize_embedding(
            average
        )

    # --------------------------------------------------
    # PENDING GROUP MANAGEMENT
    # --------------------------------------------------

    def create_pending_group(
        self,
        track_id,
        embedding,
        quality_score,
        frame_count,
        yaw=None
    ):

        group_id = self.next_group_id
        self.next_group_id += 1

        self.pending_groups[group_id] = {
            "observations": [{
                "embedding": embedding.copy(),
                "quality": quality_score,
                "yaw": yaw,
                "frame": frame_count
            }],
            "track_ids": {track_id},
            "last_seen": frame_count
        }

        self.track_to_group[track_id] = group_id

        print(
            f"NEW PENDING GROUP: Group {group_id} "
            f"| Track {track_id}"
        )

        return group_id

    def find_matching_pending_group(self, embedding):

        matches = []

        for group_id, group in self.pending_groups.items():

            representative = (
                self.calculate_weighted_embedding(
                    group["observations"]
                )
            )

            if representative is None:
                continue

            similarity = self.cosine_similarity(
                embedding,
                representative
            )

            matches.append(
                (group_id, similarity)
            )

        matches.sort(
            key=lambda item: item[1],
            reverse=True
        )

        if not matches:
            return None

        best_group, best_similarity = matches[0]

        second_similarity = (
            matches[1][1]
            if len(matches) > 1
            else -1.0
        )

        margin = best_similarity - second_similarity

        if best_similarity < self.pending_group_threshold:
            return None

        if margin < self.pending_group_margin:
            print(
                "AMBIGUOUS PENDING GROUP MATCH: "
                f"Similarity={best_similarity:.4f}, "
                f"Margin={margin:.4f}"
            )
            return None

        return best_group

    def cleanup_pending_groups(self, frame_count):

        expired_groups = []

        for group_id, group in self.pending_groups.items():

            age = frame_count - group["last_seen"]

            if age > self.pending_group_timeout:
                expired_groups.append(group_id)

        for group_id in expired_groups:

            group = self.pending_groups.pop(
                group_id
            )

            for track_id in group["track_ids"]:

                self.track_to_group.pop(
                    track_id,
                    None
                )

            print(
                f"PENDING GROUP EXPIRED: Group {group_id}"
            )

    def assign_group_to_face(self, group_id, face_id):

        group = self.pending_groups.get(group_id)

        if group is None:
            return

        for track_id in group["track_ids"]:

            self.track_to_face[track_id] = face_id
            self.track_to_group.pop(track_id, None)

        print(
            f"GROUP ASSIGNED: Group {group_id} "
            f"-> {face_id} | Tracks: "
            f"{sorted(group['track_ids'])}"
        )

        self.pending_groups.pop(group_id, None)

        # Crossings held for ANY track of this group (including tracks that
        # already ended) are replayed together in real-time order, so an
        # ENTRY on one track ID and the EXIT on another stay paired.
        held = []

        for track_id in group["track_ids"]:
            for item in self.pending_crossings.pop(track_id, []):
                held.append((item["time"], track_id, item))

        held.sort(key=lambda entry: entry[0])

        for _, track_id, item in held:
            self.record_line_crossing(
                face_id=face_id,
                crossing=item["kind"],
                crop=item["crop"],
                track_id=track_id,
                event_time=item["time"],
                frame_count=group["last_seen"],
                note=(item["note"] + " | " if item["note"] else "")
                     + "identified later"
            )

    # --------------------------------------------------
    # REGISTER CONFIRMED CANDIDATE
    # --------------------------------------------------

    def _identity_visible_elsewhere(self, face_id, group):
        """True if `face_id` is on screen right now on a track outside `group`."""

        frame_count = group["last_seen"]

        for other_track, other_face in self.track_to_face.items():

            if other_face != face_id:
                continue

            if other_track in group["track_ids"]:
                continue

            last_seen = self.boundary_counter.last_seen(other_track)

            if last_seen is None:
                continue

            if frame_count - last_seen <= self.duplicate_track_window:
                return True

        return False

    def _uncertain_is_new_person(self, group, face_id, similarity):
        """
        Decide whether a group stuck in the uncertain band is really a
        different person from its best-matching registered identity.

        Only the low band is resolved (similarity below the match
        threshold). Two registered identities that score almost the same
        stay uncertain, because registering a third could be a duplicate.
        """

        if self.uncertain_resolve_observations <= 0:
            return False

        if face_id is None:
            return False

        if similarity >= self.matcher.threshold:
            return False

        # Rule A: the best match is on screen right now on another track.
        if (
            len(group["observations"]) >= self.uncertain_resolve_observations
            and self._identity_visible_elsewhere(face_id, group)
        ):
            group["resolution_rule"] = "best match visible on another track"
            return True

        # Rule B: sustained, independent, coherent evidence.
        is_new, reason = self._sustained_new_person_evidence(
            group,
            face_id,
            similarity
        )

        group["last_reason"] = (
            f"uncertain vs {face_id} ({similarity:.2f}): {reason}"
        )

        if is_new:
            group["resolution_rule"] = "sustained independent evidence"

        return is_new

    def _independent_observation_count(self, group):
        """Observations at least one embedding interval apart."""

        frames = sorted(
            item["frame"]
            for item in group["observations"]
            if item.get("frame") is not None
        )

        unknown_frame = sum(
            1 for item in group["observations"]
            if item.get("frame") is None
        )

        count = unknown_frame
        last = None

        for frame in frames:
            if last is None or frame - last >= self.pending_embedding_interval:
                count += 1
                last = frame

        return count

    def _sustained_new_person_evidence(self, group, face_id, similarity):
        """
        Safe "this is a new person" decision when the best-matching
        registered identity is NOT visible elsewhere. ALL must hold:

          1. enough independent observations (spaced frames, not duplicates)
          2. the group average is below uncertain_new_max_similarity
             (a gap under the 0.45 match threshold)
          3. NO single observation reaches the match threshold on its own
             (one good view of the known person vetoes the decision)
          4. enough frontal observations (profile views are unreliable)
          5. the frontal observations agree with each other (one person)

        Returns (decision, human-readable reason).
        """

        if self.uncertain_new_min_observations <= 0:
            return False, "sustained-evidence rule disabled"

        if similarity >= self.uncertain_new_max_similarity:
            return False, (
                f"group similarity {similarity:.2f} not below "
                f"{self.uncertain_new_max_similarity:.2f}"
            )

        observations = group["observations"]
        independent = self._independent_observation_count(group)

        if independent < self.uncertain_new_min_observations:
            return False, (
                f"{independent} independent observations, need "
                f"{self.uncertain_new_min_observations}"
            )

        for item in observations:

            top = self.matcher.find_top_matches(item["embedding"])

            if top and top[0][1] >= self.matcher.threshold:
                return False, (
                    f"one observation matches {top[0][0]} at "
                    f"{top[0][1]:.2f}"
                )

        frontal = [
            item for item in observations
            if item.get("yaw") is None
            or abs(item["yaw"]) <= self.max_register_yaw
        ]

        if len(frontal) < self.uncertain_new_min_frontal:
            return False, (
                f"{len(frontal)} frontal observations, need "
                f"{self.uncertain_new_min_frontal}"
            )

        centre = self.calculate_weighted_embedding(frontal)

        coherence = float(np.mean([
            self.cosine_similarity(item["embedding"], centre)
            for item in frontal
        ]))

        if coherence < self.uncertain_new_min_coherence:
            return False, (
                f"observations disagree (coherence {coherence:.2f} < "
                f"{self.uncertain_new_min_coherence:.2f})"
            )

        return True, (
            f"{independent} independent observations, "
            f"{len(frontal)} frontal, coherence {coherence:.2f}"
        )

    def _few_observations_ok(self, group):
        """
        A track that ENDED with fewer than required_observations. Only two
        observations are ever accepted, and only if both are good quality,
        frontal and strongly agree with each other.
        """

        observations = group["observations"]

        if len(observations) != 2:
            return False, f"{len(observations)} observation(s), need 2+"

        for item in observations:

            if item["quality"] < self.two_observation_min_quality:
                return False, (
                    f"quality {item['quality']:.2f} < "
                    f"{self.two_observation_min_quality:.2f}"
                )

            if (
                item.get("yaw") is not None
                and abs(item["yaw"]) > self.max_register_yaw
            ):
                return False, f"not frontal (yaw {item['yaw']:.0f})"

        agreement = self.cosine_similarity(
            observations[0]["embedding"],
            observations[1]["embedding"]
        )

        if agreement < self.two_observation_min_agreement:
            return False, (
                f"two observations disagree ({agreement:.2f} < "
                f"{self.two_observation_min_agreement:.2f})"
            )

        gap = abs(
            (observations[0].get("frame") or 0)
            - (observations[1].get("frame") or 0)
        )

        if gap < self.pending_embedding_interval:
            return False, "two observations are not independent frames"

        return True, f"2 high-quality frontal observations ({agreement:.2f})"

    def resolve_ended_group(self, group_id, frame_count):
        """
        Called once every track of a pending group has ended, so no more
        observations can arrive. Gives a 2-observation group one safe chance.
        """

        group = self.pending_groups.get(group_id)

        if group is None:
            return None

        observations = group["observations"]

        if len(observations) >= self.required_observations:
            return None

        ok, reason = self._few_observations_ok(group)

        if not ok:
            group["last_reason"] = f"too few observations: {reason}"
            return None

        average = self.calculate_weighted_embedding(observations)
        status, face_id, similarity = self.matcher.classify_match(average)

        if status in ("KNOWN", "UNCERTAIN"):
            # KNOWN is handled by the normal path; UNCERTAIN needs more
            # evidence than two observations can give.
            group["last_reason"] = (
                f"too few observations: 2 observations but "
                f"{status} vs {face_id} ({similarity:.2f})"
            )
            return None

        return self.register_confirmed_candidate(group_id, allow_few=True)

    def register_confirmed_candidate(self, group_id, allow_few=False):

        group = self.pending_groups.get(group_id)

        if group is None:
            return None

        observations = group["observations"]

        if len(observations) < self.required_observations and not allow_few:
            return None

        average_embedding = (
            self.calculate_weighted_embedding(
                observations
            )
        )

        if average_embedding is None:
            return None

        status, face_id, similarity = (
            self.matcher.classify_match(
                average_embedding
            )
        )

        # Candidate may already belong to a registered visitor.
        if status == "KNOWN":

            self.assign_group_to_face(
                group_id,
                face_id
            )

            print(
                f"CANDIDATE MATCHED: Group {group_id} "
                f"-> {face_id} | Similarity: {similarity:.4f}"
            )

            return face_id

        if status == "UNCERTAIN":

            if not self._uncertain_is_new_person(
                group,
                face_id,
                similarity
            ):

                print(
                    f"IDENTITY UNCERTAIN: Group {group_id} "
                    f"| Similarity: {similarity:.4f} "
                    f"| {group.get('last_reason', '')}"
                )

                return None

            # Same moment as the best match, so a different person.
            # Continue below and register a new identity.
            self.event_logger.log_event(
                "UNCERTAIN_RESOLVED",
                message=(
                    f"Group {group_id} resolved as a NEW person: best match "
                    f"{face_id} ({similarity:.4f}); rule: "
                    f"{group.get('resolution_rule', 'unknown')}; "
                    f"{len(observations)} observations"
                )
            )

            print(
                f"UNCERTAIN RESOLVED AS NEW: Group {group_id} "
                f"| {group.get('resolution_rule', '')} "
                f"| Similarity: {similarity:.4f}"
            )

        # In dry-run, do not create even temporary new identities.
        # Existing registered identities can still be recognized above.
        if self.dry_run:
            print(
                f"DRY RUN: unknown Group {group_id} not registered; "
                "no face or embedding files will be written."
            )
            return None

        # Only register a NEW person from front-facing views. Profile views
        # score low against the same person's front view and would create a
        # duplicate identity. The group keeps collecting observations and
        # tries again until a frontal one arrives (or the group expires).
        frontal = [
            item for item in observations
            if item.get("yaw") is None
            or abs(item["yaw"]) <= self.max_register_yaw
        ]

        if not frontal:
            print(
                f"WAITING FOR FRONTAL VIEW: Group {group_id} "
                f"| {len(observations)} observations, smallest angle "
                f"{min(abs(item['yaw']) for item in observations):.0f} deg, "
                f"limit {self.max_register_yaw:.0f} deg"
            )
            return None

        average_embedding = self.calculate_weighted_embedding(frontal)

        # Register one identity for the entire group.
        face_id = self.matcher.register_face(
            average_embedding
        )

        if face_id is None:
            return None

        if not self.dry_run:
            if not self.database.save_face(
                face_id,
                average_embedding
            ):

                self.matcher.known_faces.pop(
                    face_id,
                    None
                )

                self.matcher.next_face_number -= 1

                print(
                    "Database registration failed:",
                    face_id
                )

                return None
        else:
            print(
                f"DRY RUN: {face_id} created in memory only; "
                "not saved to MySQL."
            )

        self.assign_group_to_face(
            group_id,
            face_id
        )

        self.event_logger.log_event(
            "REGISTRATION",
            face_id=face_id,
            message=f"New visitor registered from pending group {group_id}"
        )

        print(
            f"NEW VISITOR CONFIRMED: "
            f"Group {group_id} -> {face_id}"
        )

        return face_id

    # --------------------------------------------------
    # IDENTITY PROCESSING
    # --------------------------------------------------

    def process_identity(
        self,
        track_id,
        embedding,
        quality_score,
        frame_count,
        yaw=None
    ):

        # Already recognized track.
        if track_id in self.track_to_face:
            return self.track_to_face[track_id]

        status, face_id, similarity = (
            self.matcher.classify_match(
                embedding
            )
        )

        # Match against registered identities first.
        if status == "KNOWN":

            self.track_to_face[track_id] = face_id

            old_group = self.track_to_group.pop(
                track_id,
                None
            )

            if old_group is not None:
                group = self.pending_groups.get(old_group)

                if group is not None:
                    self.assign_group_to_face(
                        old_group,
                        face_id
                    )

            print(
                f"RECOGNIZED: Track {track_id} "
                f"-> {face_id} | Similarity: {similarity:.4f}"
            )

            return face_id

        # Reuse an existing pending group if the
        # embedding strongly matches that candidate.
        group_id = self.track_to_group.get(track_id)
        created_now = False

        if group_id is None:

            group_id = self.find_matching_pending_group(
                embedding
            )

            if group_id is None:

                group_id = self.create_pending_group(
                    track_id,
                    embedding,
                    quality_score,
                    frame_count,
                    yaw
                )
                created_now = True

            else:

                group = self.pending_groups[group_id]

                group["track_ids"].add(track_id)
                self.track_to_group[track_id] = group_id

                print(
                    f"CROSS-TRACK MATCH: Track {track_id} "
                    f"joined Group {group_id}"
                )

        group = self.pending_groups.get(group_id)

        if group is None:
            return None

        observations = group["observations"]

        if created_now:
            # create_pending_group() already stored this observation.
            # Appending it again counted ONE embedding as TWO observations
            # and double-weighted it in the group average.
            print(
                f"PENDING: Group {group_id} | "
                f"Track {track_id} | "
                f"Observations: {len(observations)}/"
                f"{self.required_observations}"
            )
            return None

        # Check consistency against the group's
        # existing representative embedding.
        if observations:

            previous_average = (
                self.calculate_weighted_embedding(
                    observations
                )
            )

            if previous_average is not None:

                consistency = self.cosine_similarity(
                    embedding,
                    previous_average
                )

                if consistency < self.minimum_consistency:

                    print(
                        f"INCONSISTENT OBSERVATION: "
                        f"Track {track_id} | {consistency:.4f}"
                    )

                    return None

        observations.append({
            "embedding": embedding.copy(),
            "quality": quality_score,
            "yaw": yaw,
            "frame": frame_count
        })

        group["last_seen"] = frame_count

        print(
            f"PENDING: Group {group_id} | "
            f"Track {track_id} | "
            f"Observations: {len(observations)}/"
            f"{self.required_observations}"
        )

        if len(observations) >= self.required_observations:

            return self.register_confirmed_candidate(
                group_id
            )

        return None

    # --------------------------------------------------
    # ADDITIONAL EMBEDDING STORAGE
    # --------------------------------------------------

    def save_additional_embedding(
        self,
        track_id,
        face_id,
        embedding,
        similarity
    ):

        # Do not alter templates or write embeddings during dry-run.
        if self.dry_run:
            return False

        if similarity is None or similarity < self.template_min_similarity:
            return False

        stored = self.matcher.known_faces.get(
            face_id,
            []
        )

        if len(stored) >= self.matcher.max_embeddings_per_face:
            return False

        if not self.matcher.add_embedding(
            face_id,
            embedding
        ):
            return False

        if not self.dry_run:
            if not self.database.add_embedding(
                face_id,
                embedding
            ):

                self.matcher.known_faces[face_id].pop()

                print(
                    "Could not save additional embedding:",
                    face_id
                )

                return False
        else:
            print(
                f"DRY RUN: additional embedding for {face_id} "
                "kept in memory only."
            )

        print(
            "ADDITIONAL EMBEDDING SAVED:",
            face_id
        )

        return True

    # --------------------------------------------------
    # SESSION VISIT STATE
    #
    # Visits are keyed by the permanent FACE ID, never by the
    # temporary ByteTrack ID. session_visits[face_id] holds the
    # set of track IDs currently attached to that visit.
    # --------------------------------------------------

    def _other_live_tracks(self, visit, track_id, frame_count):
        """Other tracks of the same visit that are inside and recently seen."""

        live = []

        for other in visit["track_ids"]:

            if other == track_id:
                continue

            if self.boundary_counter.is_inside(other) is not True:
                continue

            last_seen = self.boundary_counter.last_seen(other)

            if last_seen is None:
                continue

            if frame_count - last_seen <= self.duplicate_track_window:
                live.append(other)

        return live

    # --------------------------------------------------
    # UNIDENTIFIED CROSSINGS (never silently dropped)
    # --------------------------------------------------

    def record_unidentified_crossing(
        self,
        track_id,
        crossing,
        crop,
        event_time,
        reason=""
    ):
        path = self.event_logger.save_unidentified_image(
            crop,
            track_id,
            crossing,
            event_time
        )

        self.event_logger.log_event(
            f"UNIDENTIFIED_{crossing}",
            track_id=track_id,
            event_time=event_time,
            message=(
                f"Identity never confirmed ({reason}); "
                f"image: {path}; no visit/MySQL record created"
            )
        )

        self.stats["unidentified"] += 1

        print(
            f"UNIDENTIFIED {crossing}: Track {track_id} "
            f"({reason}) | Image: {path}"
        )

        return False

    # --------------------------------------------------
    # ENTRY / EXIT RECORDING
    # --------------------------------------------------

    def record_line_crossing(
        self,
        face_id,
        crossing,
        crop,
        track_id,
        event_time,
        frame_count,
        note=""
    ):
        """
        Record one ENTRY or EXIT for a confirmed Face ID.

        Pairing rules (enforced here, before anything is written):
          * ENTRY while the face already has an active visit
            -> suppressed; the track is attached to that visit
               (same person under a new ByteTrack ID).
          * EXIT without an active visit -> suppressed.
          * EXIT while another track of the same visit is still inside
            -> suppressed (ID switch, person is still in the area).

        Write order (consistency):
          stage image -> MySQL transaction (live) -> move image + events.log.
          If MySQL rejects the event, the staged image is deleted and
          nothing is written to events.log.

        event_time is the real crossing time, even when identification
        finished later.
        """

        if crossing not in ("ENTRY", "EXIT"):
            return False

        if not face_id:
            return self.record_unidentified_crossing(
                track_id, crossing, crop, event_time,
                reason="no face id"
            )

        visit = self.session_visits.get(face_id)

        # ---------------- pairing rules ----------------

        if crossing == "ENTRY":

            if visit is not None:
                visit["track_ids"].add(track_id)
                visit["last_frame"] = frame_count

                self.event_logger.log_event(
                    "ENTRY_SUPPRESSED",
                    face_id=face_id,
                    track_id=track_id,
                    event_time=event_time,
                    message="Face already has an active visit; "
                            "track attached (ByteTrack ID change)"
                )
                self.stats["suppressed"] += 1
                return False

        else:

            if visit is None:
                self.event_logger.log_event(
                    "EXIT_SUPPRESSED",
                    face_id=face_id,
                    track_id=track_id,
                    event_time=event_time,
                    message="No active visit; EXIT would be unpaired"
                )
                self.stats["suppressed"] += 1
                return False

            visit["track_ids"].discard(track_id)

            others = self._other_live_tracks(
                visit, track_id, frame_count
            )

            if others:
                self.event_logger.log_event(
                    "EXIT_SUPPRESSED",
                    face_id=face_id,
                    track_id=track_id,
                    event_time=event_time,
                    message=f"Another live track {others} of the same "
                            "visit is still inside"
                )
                self.stats["suppressed"] += 1
                return False

        # ---------------- phase 1: stage image ----------------

        staged = self.event_logger.stage_event_image(
            crop, face_id, track_id, crossing, event_time
        )

        if staged is None:
            self.event_logger.log_event(
                "EVENT_IMAGE_FAILED",
                face_id=face_id,
                track_id=track_id,
                message=f"{crossing} not recorded: image could not be saved"
            )
            return False

        final_path, staged_path = staged

        # ---------------- phase 2: database ----------------

        db_note = ""

        if not self.dry_run:

            if crossing == "ENTRY":

                visit_id, created = self.database.start_visit(
                    face_id, final_path, event_time
                )

                if visit_id is None:
                    self.event_logger.discard_staged_image(staged_path)
                    self.event_logger.log_event(
                        "DATABASE_ENTRY_FAILED",
                        face_id=face_id,
                        track_id=track_id,
                        message="Visit not created; staged image deleted"
                    )
                    return False

                if not created:
                    # MySQL already has an ACTIVE visit (e.g. left over
                    # from an earlier run). No new ENTRY row exists, so
                    # do not keep an image that no DB event points to.
                    self.event_logger.discard_staged_image(staged_path)
                    self.session_visits[face_id] = {
                        "track_ids": {track_id},
                        "last_crop": crop,
                        "last_time": event_time,
                        "last_frame": frame_count,
                        "entry_time": event_time
                    }
                    self.session_faces.add(face_id)
                    self.event_logger.log_event(
                        "ENTRY_ATTACHED_TO_EXISTING_DB_VISIT",
                        face_id=face_id,
                        track_id=track_id,
                        event_time=event_time,
                        message=f"MySQL visit {visit_id} was already ACTIVE"
                    )
                    self.stats["suppressed"] += 1
                    return False

                db_note = f"visit_id={visit_id}"

            else:

                if not self.database.complete_visit(
                    face_id, final_path, event_time
                ):
                    self.event_logger.discard_staged_image(staged_path)
                    self.session_visits.pop(face_id, None)
                    self.event_logger.log_event(
                        "DATABASE_EXIT_IGNORED",
                        face_id=face_id,
                        track_id=track_id,
                        message="MySQL has no ACTIVE visit; staged image deleted"
                    )
                    self.stats["suppressed"] += 1
                    return False

        # ---------------- phase 3: finalize ----------------

        message = " | ".join(
            part for part in (
                db_note,
                note,
                "DRY RUN" if self.dry_run else ""
            ) if part
        )

        if not self.event_logger.commit_event_image(
            final_path, staged_path, face_id, track_id,
            crossing, event_time, message
        ):
            print(
                f"WARNING: {crossing} stored for {face_id} but "
                "image finalize failed"
            )

        # ---------------- update session state ----------------

        if crossing == "ENTRY":
            self.session_visits[face_id] = {
                "track_ids": {track_id},
                "last_crop": crop,
                "last_time": event_time,
                "last_frame": frame_count,
                "entry_time": event_time
            }
            self.session_faces.add(face_id)
            self.stats["entries"] += 1
        else:
            self.session_visits.pop(face_id, None)
            self.stats["exits"] += 1

        print(
            f"{'DRY RUN ' if self.dry_run else ''}{crossing}: {face_id} "
            f"| Track {track_id} | Image: {final_path}"
        )

        return True

    # --------------------------------------------------
    # PENDING CROSSINGS (identity not known yet)
    # --------------------------------------------------

    def queue_pending_crossing(
        self,
        track_id,
        kind,
        crop,
        event_time,
        frame_count,
        note
    ):
        self.pending_crossings.setdefault(track_id, []).append({
            "kind": kind,
            "crop": crop,
            "time": event_time,
            "frame": frame_count,
            "note": note
        })

        self.event_logger.log_event(
            "CROSSING_PENDING",
            track_id=track_id,
            event_time=event_time,
            message=f"{kind} detected before identity was confirmed; "
                    "held until the Face ID is known"
        )

        print(
            f"PENDING {kind}: Track {track_id} "
            "(waiting for identification)"
        )

    def flush_pending_for_track(self, track_id, face_id, frame_count):
        """Replay held crossings, in order, with their ORIGINAL times."""

        events = self.pending_crossings.pop(track_id, [])

        for item in events:
            self.record_line_crossing(
                face_id=face_id,
                crossing=item["kind"],
                crop=item["crop"],
                track_id=track_id,
                event_time=item["time"],
                frame_count=frame_count,
                note=(item["note"] + " | " if item["note"] else "")
                     + "identified later"
            )

    # --------------------------------------------------
    # ONE TRACK, ONE FRAME
    # --------------------------------------------------

    def process_track_event(
        self,
        track_id,
        face_id,
        crossing,
        crop,
        now,
        frame_count
    ):
        """
        Called for every processed face. Handles crossings and releases
        pending crossings as soon as the track has a Face ID.
        """

        # Keep the newest crop for "track lost" exits.
        if crop is not None:
            self.track_last_crop[track_id] = (crop, now)

        if face_id is not None:

            if track_id in self.pending_crossings:
                self.flush_pending_for_track(
                    track_id, face_id, frame_count
                )

            visit = self.session_visits.get(face_id)

            if (
                visit is not None
                and track_id in visit["track_ids"]
                and crop is not None
            ):
                visit["last_crop"] = crop
                visit["last_time"] = now
                visit["last_frame"] = frame_count

        if crossing is None:
            return

        kind = "ENTRY" if crossing in ("ENTRY", "INITIAL_INSIDE") else "EXIT"

        note = (
            "initial_presence (already inside when first seen)"
            if crossing == "INITIAL_INSIDE"
            else ""
        )

        if face_id is None:
            self.queue_pending_crossing(
                track_id, kind, crop, now, frame_count, note
            )
            return

        self.record_line_crossing(
            face_id=face_id,
            crossing=kind,
            crop=crop,
            track_id=track_id,
            event_time=now,
            frame_count=frame_count,
            note=note
        )

    # --------------------------------------------------
    # TRACK EXPIRY / SHUTDOWN
    # --------------------------------------------------

    def _group_has_live_track(self, group, track_id, frame_count):
        """True if another track of the group is still being seen."""

        for other in group["track_ids"]:

            if other == track_id:
                continue

            # Still known to the boundary counter = not expired yet, so it
            # may still reappear and add observations to the group.
            if self.boundary_counter.last_seen(other) is not None:
                return True

        return False

    def _track_summary(self, track_id):
        """What the detector actually saw for this track."""

        info = self.track_stats.get(track_id)

        if not info:
            return "never seen by the face loop"

        return (
            f"seen on {info['seen']} analysed frame(s), largest face "
            f"{info['max_w']}x{info['max_h']}px, max detector "
            f"confidence {info['max_conf']:.2f}"
        )

    def _unidentified_reason(self, track_id):
        """The ACTUAL reason a track never got a Face ID."""

        group_id = self.track_to_group.get(track_id)
        group = self.pending_groups.get(group_id)

        if group is not None:
            return (
                f"group {group_id}, {len(group['observations'])} "
                f"observations, "
                f"{group.get('last_reason', 'not enough evidence yet')}; "
                f"{self._track_summary(track_id)}"
            )

        rejects = self.track_reject_reasons.get(track_id)

        if rejects:
            summary = ", ".join(
                f"{name} x{count}" for name, count in sorted(
                    rejects.items(), key=lambda pair: -pair[1]
                )
            )
            return (
                f"no embedding: {summary}; "
                f"{self._track_summary(track_id)}"
            )

        return (
            "no embedding: no observation attempt reached the checks; "
            f"{self._track_summary(track_id)}"
        )

    def handle_expired_tracks(
        self,
        expired,
        frame_count,
        emit_lost_exits=True,
        end_of_run=False,
        defer_pending_groups=True
    ):
        """
        For tracks that disappeared:
          * resolve their pending crossings (identified -> record,
            otherwise -> unidentified evidence, never dropped);
          * if the person was inside and no other track of the same
            visit is alive, record an EXIT at the last-seen time.
        """

        for track_id, state in expired:

            face_id = self.track_to_face.get(track_id)
            last = self.track_last_crop.get(track_id)

            # ---- pending crossings ----
            deferred = False

            if face_id is None and defer_pending_groups:

                group_id = self.track_to_group.get(track_id)
                group = self.pending_groups.get(group_id)

                if group is not None:

                    if self._group_has_live_track(
                        group, track_id, frame_count
                    ):
                        # A sibling track of the same person is still being
                        # followed and may yet identify the group.
                        deferred = True

                    else:
                        # Last track of the group: no more observations.
                        self.resolve_ended_group(group_id, frame_count)
                        face_id = self.track_to_face.get(track_id)

            if track_id in self.pending_crossings and not deferred:

                if face_id is not None:
                    self.flush_pending_for_track(
                        track_id, face_id, frame_count
                    )
                else:
                    reason = self._unidentified_reason(track_id)

                    for item in self.pending_crossings.pop(track_id):
                        self.record_unidentified_crossing(
                            track_id,
                            item["kind"],
                            item["crop"],
                            item["time"],
                            reason=reason
                        )

                    # Earlier tracks of the same group were waiting for it.
                    group = self.pending_groups.get(
                        self.track_to_group.get(track_id)
                    )

                    if group is not None:
                        for sibling in sorted(group["track_ids"]):
                            for item in self.pending_crossings.pop(
                                sibling, []
                            ):
                                self.record_unidentified_crossing(
                                    sibling,
                                    item["kind"],
                                    item["crop"],
                                    item["time"],
                                    reason=self._unidentified_reason(sibling)
                                )

            # ---- track lost while inside ----
            if emit_lost_exits and face_id is not None and state.get("inside") is True:

                visit = self.session_visits.get(face_id)

                if visit is not None and track_id in visit["track_ids"]:

                    # Use the most recent sighting of the WHOLE visit (any
                    # attached track), not just this expiring track, so the
                    # EXIT time/image are never older than the latest sighting.
                    crop = visit["last_crop"]
                    seen_time = visit["last_time"]

                    if crop is None and last is not None:
                        crop, seen_time = last

                    self.record_line_crossing(
                        face_id=face_id,
                        crossing="EXIT",
                        crop=crop,
                        track_id=track_id,
                        event_time=seen_time,
                        frame_count=frame_count,
                        note=(
                            "video ended / processing stopped while inside"
                            if end_of_run
                            else "track lost while inside"
                        )
                    )

            # ---- per-track memory cleanup ----
            if not deferred:
                self.track_stats.pop(track_id, None)
                self.track_reject_reasons.pop(track_id, None)
                self.track_last_crop.pop(track_id, None)
                self.last_embedding_frame.pop(track_id, None)
                self.last_template_frame.pop(track_id, None)

    def finalize_session(self, frame_count):
        """Resolve everything still open when processing stops."""

        emit = self.close_visits_on_end

        self.handle_expired_tracks(
            self.boundary_counter.pop_all(),
            frame_count,
            emit_lost_exits=emit,
            end_of_run=True,
            defer_pending_groups=False
        )

        # Pending crossings of tracks that had no boundary state.
        for track_id in list(self.pending_crossings):
            self.handle_expired_tracks(
                [(track_id, {"inside": None})],
                frame_count,
                emit_lost_exits=False,
                defer_pending_groups=False
            )

        # Visits still open: close them so every ENTRY has one EXIT.
        if emit:
            for face_id in list(self.session_visits):

                visit = self.session_visits[face_id]
                visit["track_ids"].clear()

                self.record_line_crossing(
                    face_id=face_id,
                    crossing="EXIT",
                    crop=visit["last_crop"],
                    track_id="END",
                    event_time=visit["last_time"],
                    frame_count=frame_count,
                    note="video ended / processing stopped"
                )

        unique_db = None

        try:
            unique_db = self.database.get_unique_visitor_count()
        except Exception:
            pass

        summary = (
            f"entries={self.stats['entries']} exits={self.stats['exits']} "
            f"suppressed={self.stats['suppressed']} "
            f"unidentified={self.stats['unidentified']} "
            f"unique_faces_this_run={len(self.session_faces)} "
            f"unique_visitors_in_db={unique_db} "
            f"open_visits={len(self.session_visits)}"
        )

        self.event_logger.log_event("SESSION_SUMMARY", message=summary)
        print("\nSESSION SUMMARY:", summary)

    # --------------------------------------------------
    # VIDEO / LIVE STREAM INPUT
    # --------------------------------------------------

    @staticmethod
    def _is_stream(source):
        """True for live network sources (RTSP etc.), False for files."""
        return isinstance(source, str) and source.strip().lower().startswith(
            ("rtsp://", "rtsps://", "rtmp://", "http://", "https://")
        )

    def _open_capture(self, source, is_stream):
        """
        File   -> plain cv2.VideoCapture (unchanged behaviour).
        Stream -> background reader that always keeps the NEWEST frame and
                  reconnects automatically (see src/video_source.py).

        Optional config.json keys (all have defaults):
            processing.reconnect_delay_seconds   seconds between retries (2)
            processing.reconnect_max_seconds     give up after this long (60)
        """

        if not is_stream:
            return cv2.VideoCapture(source)

        processing = self.config.get("processing", {})

        return open_video_source(
            source,
            on_event=lambda name, message: self.event_logger.log_event(
                name, message=message
            ),
            retry_delay=float(processing.get("reconnect_delay_seconds", 2)),
            max_downtime=float(processing.get("reconnect_max_seconds", 60)),
            # Dashboard "Stop" must work even while the camera is down.
            should_stop=lambda: bool(STOP_FILE and os.path.exists(STOP_FILE))
        )

    # --------------------------------------------------
    # VIDEO PROCESSING
    # --------------------------------------------------

    def process_video(self, video_path):

        live_stream = self._is_stream(video_path)
        video = self._open_capture(video_path, live_stream)

        if not video.isOpened():

            print(
                "Could not open video:",
                video_path
            )

            self.database.close()
            return

        frame_count = 0
        processed_frames = 0
        # A live stream already drops stale frames in the reader (it always
        # hands over the newest one), so frame_skip only applies to files.
        step = 1 if live_stream else self.frame_skip + 1

        print(
            "\nStarting video processing... "
            + (
                "(LIVE STREAM: always analysing the newest frame, "
                "auto-reconnect on)"
                if live_stream
                else f"(frame_skip={self.frame_skip}: analysing 1 of every {step} frames)"
            )
        )

        if not self.dry_run:
            stale = self.database.get_all_active_visits()
            if stale:
                self.event_logger.log_event(
                    "STALE_ACTIVE_VISITS",
                    message=f"{len(stale)} ACTIVE visit(s) left in MySQL "
                            f"by an earlier run: {[s[1] for s in stale]}"
                )
                print(
                    "WARNING: ACTIVE visits from an earlier run:",
                    [s[1] for s in stale]
                )

        try:

            while True:

                if STOP_FILE and os.path.exists(STOP_FILE):
                    print("Stop requested from dashboard.")
                    break

                # grab() advances without decoding: skipped frames cost
                # almost nothing, which matters for 4K video.
                if not video.grab():
                    break

                frame_count += 1

                if (frame_count - 1) % step != 0:
                    continue

                ret, frame = video.retrieve()

                if not ret:
                    break

                processed_frames += 1
                now = datetime.now()

                self.cleanup_pending_groups(
                    frame_count
                )

                self.handle_expired_tracks(
                    self.boundary_counter.cleanup(frame_count),
                    frame_count
                )

                tracked_faces = self.tracker.track_faces(
                    frame
                )

                # Full-frame InsightFace is the most expensive call.
                # Run it lazily: only if a face actually needs an embedding.
                insight_faces = None
                insight_ready = False

                height, width = frame.shape[:2]

                for face in tracked_faces:

                    track_id = face["track_id"]

                    if track_id not in self.logged_track_ids:
                        self.logged_track_ids.add(track_id)
                        self.event_logger.log_event(
                            "TRACKING",
                            track_id=track_id,
                            message="New ByteTrack ID observed"
                        )

                    x1, y1, x2, y2 = map(
                        int,
                        face["bbox"]
                    )

                    x1 = max(0, min(x1, width))
                    x2 = max(0, min(x2, width))

                    y1 = max(0, min(y1, height))
                    y2 = max(0, min(y2, height))

                    if x2 <= x1 or y2 <= y1:
                        continue

                    bbox = (
                        x1,
                        y1,
                        x2,
                        y2
                    )

                    info = self.track_stats.setdefault(
                        track_id,
                        {"seen": 0, "max_w": 0, "max_h": 0, "max_conf": 0.0}
                    )
                    info["seen"] += 1
                    info["max_w"] = max(info["max_w"], x2 - x1)
                    info["max_h"] = max(info["max_h"], y2 - y1)
                    info["max_conf"] = max(
                        info["max_conf"], float(face["confidence"])
                    )

                    face_id = self.track_to_face.get(
                        track_id
                    )

                    if face_id is None:
                        interval = self.pending_embedding_interval
                    else:
                        interval = self.known_embedding_interval

                    last_frame = self.last_embedding_frame.get(
                        track_id,
                        -interval
                    )

                    should_process_embedding = (
                        frame_count - last_frame >= interval
                    )

                    if should_process_embedding:

                        is_good, quality_score, reject_reason = (
                            self.assess_face_quality_detail(
                                frame,
                                bbox,
                                face["confidence"]
                            )
                        )

                        if not is_good:
                            # Rejected before InsightFace (cheap): record
                            # why, and retry on the next analysed frame
                            # instead of waiting a whole interval.
                            self._note_reject(track_id, reject_reason)

                        if is_good:

                            self.last_embedding_frame[
                                track_id
                            ] = frame_count

                            if not insight_ready:
                                insight_faces = self.recognizer.app.get(
                                    frame
                                )
                                insight_ready = True

                            match = self.find_insightface_match(
                                bbox,
                                insight_faces
                            )

                            embedding, yaw = (
                                match if match is not None
                                else (None, None)
                            )

                            if match is None:
                                self._note_reject(
                                    track_id,
                                    self.last_match_reason
                                )

                            if embedding is not None:

                                if track_id not in self.logged_embedding_tracks:
                                    self.logged_embedding_tracks.add(track_id)
                                    self.event_logger.log_event(
                                        "EMBEDDING_GENERATED",
                                        face_id=face_id,
                                        track_id=track_id,
                                        message="InsightFace embedding generated"
                                    )

                                if face_id is None:

                                    face_id = self.process_identity(
                                        track_id,
                                        embedding,
                                        quality_score,
                                        frame_count,
                                        yaw
                                    )

                                else:

                                    matches = (
                                        self.matcher.find_top_matches(
                                            embedding
                                        )
                                    )

                                    if matches:

                                        best_face_id, similarity = (
                                            matches[0]
                                        )

                                        second_similarity = (
                                            matches[1][1]
                                            if len(matches) > 1
                                            else -1.0
                                        )

                                        if (
                                            best_face_id == face_id
                                            and similarity - second_similarity
                                            >= self.template_min_margin
                                        ):

                                            last_template = (
                                                self.last_template_frame.get(
                                                    track_id,
                                                    -self.template_interval
                                                )
                                            )

                                            if (
                                                frame_count - last_template
                                                >= self.template_interval
                                            ):

                                                saved = (
                                                    self.save_additional_embedding(
                                                        track_id,
                                                        face_id,
                                                        embedding,
                                                        similarity
                                                    )
                                                )

                                                if saved:

                                                    self.last_template_frame[
                                                        track_id
                                                    ] = frame_count

                                if (
                                    face_id is not None
                                    and track_id not in
                                    self.image_saved_for_track
                                ):

                                    saved = self.save_face_image(
                                        frame,
                                        bbox,
                                        face_id,
                                        track_id
                                    )

                                    if saved:
                                        self.image_saved_for_track.add(
                                            track_id
                                        )

                    # Identity may also have been assigned through a
                    # cross-track group while processing another face.
                    face_id = self.track_to_face.get(
                        track_id,
                        face_id
                    )

                    # ------------------------------------------
                    # FOUR-BOUNDARY ENTRY / EXIT DETECTION
                    # ------------------------------------------

                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2

                    crossing = self.boundary_counter.update(
                        track_id,
                        center_x,
                        center_y,
                        width,
                        height,
                        frame_count
                    )

                    # Log recognition once when a Track ID obtains
                    # a confirmed identity.
                    if (
                        face_id is not None
                        and track_id not in self.logged_recognition_tracks
                    ):
                        self.logged_recognition_tracks.add(track_id)
                        self.event_logger.log_event(
                            "RECOGNITION",
                            face_id=face_id,
                            track_id=track_id,
                            message="Track matched to a registered identity"
                        )

                    crop = self.event_logger.crop_from_frame(
                        frame,
                        bbox
                    )

                    self.process_track_event(
                        track_id,
                        face_id,
                        crossing,
                        crop,
                        now,
                        frame_count
                    )

                    # ---------------- drawing ----------------
                    if not self.show_window:
                        continue

                    color = (
                        (0, 255, 0)
                        if face_id is not None
                        else (0, 165, 255)
                    )

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        color,
                        2
                    )

                    label = f"Track: {track_id}"

                    if face_id is not None:
                        label += f" | {face_id}"

                    else:

                        group_id = self.track_to_group.get(
                            track_id
                        )

                        if group_id is not None:
                            label += f" | Group {group_id}"

                        else:
                            label += " | Pending"

                    cv2.putText(
                        frame,
                        label,
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2
                    )

                if self.show_window:

                    left, top, right, bottom = self.boundary_counter.get_bounds(
                        width, height, boundary="inner"
                    )

                    cv2.rectangle(
                        frame,
                        (left, top),
                        (right, bottom),
                        (0, 255, 255),
                        2
                    )

                    mode_text = "DRY RUN" if self.dry_run else "LIVE"

                    cv2.putText(
                        frame,
                        f"{mode_text} | skip={self.frame_skip} | "
                        f"entries={self.stats['entries']} "
                        f"exits={self.stats['exits']}",
                        (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.48,
                        (0, 255, 255),
                        2
                    )

                    display = frame

                    # Showing a 4K frame is slow; shrink it for display only.
                    if width > self.display_max_width:
                        scale = self.display_max_width / width
                        display = cv2.resize(
                            frame,
                            (int(width * scale), int(height * scale))
                        )

                    cv2.imshow(
                        "Face Recognition and Tracking",
                        display
                    )

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

        finally:

            video.release()
            cv2.destroyAllWindows()

            # Close open visits / resolve pending events BEFORE the DB closes.
            try:
                self.finalize_session(frame_count)
            except Exception as error:
                print("Session finalization failed:", error)
                self.event_logger.log_event(
                    "FINALIZE_FAILED",
                    message=str(error)
                )

            self.event_logger.log_event(
                "SYSTEM_STOP",
                message=(
                    f"Video processing ended after {frame_count} frames "
                    f"({processed_frames} analysed)"
                )
            )

            print("\nVideo processing completed!")

            print(
                "Total frames read:",
                frame_count,
                "| analysed:",
                processed_frames
            )

            print(
                "Registered identities in memory:",
                len(self.matcher.known_faces)
            )

            self.database.close()


if __name__ == "__main__":

    processor = VideoProcessor()

    # Dashboard can override the video and hide the OpenCV window.
    video_source = os.environ.get("VIDEO_PATH") or processor.config.get(
        "camera", {}
    ).get("source", "data/sample.mp4")

    if os.environ.get("FACE_TRACKER_HEADLESS") == "1":
        processor.show_window = False

    processor.process_video(video_source)