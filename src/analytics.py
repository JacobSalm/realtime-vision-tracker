from collections import defaultdict, deque


class Analytics:

    def __init__(
        self,
        min_confirm_frames=5,
        trail_length=30,
        timeout_frames=90
    ):
        self.min_confirm_frames = min_confirm_frames
        self.timeout_frames = timeout_frames

        self.seen_ids = set()

        self.class_counts = defaultdict(int)
        self.track_hits = defaultdict(int)

        self.track_history = defaultdict(
            lambda: deque(maxlen=trail_length)
        )

        self.last_seen_frame = {}
        self.track_side = {}

        self.entered_ids = set()
        self.exited_ids = set()

        self.entered_count = 0
        self.exited_count = 0


    def process_object(
        self,
        track_id,
        class_name,
        confidence,
        box,
        frame_number,
        line_y,
        line_margin
    ):

        self.last_seen_frame[track_id] = frame_number
        self.track_hits[track_id] += 1

        # ---------------------------------
        # Confirm object
        # ---------------------------------

        if (
            self.track_hits[track_id]
            >= self.min_confirm_frames
            and track_id not in self.seen_ids
        ):

            self.seen_ids.add(track_id)

            self.class_counts[class_name] += 1

            print(
                f"CONFIRMED OBJECT -> "
                f"ID: {track_id} | "
                f"Class: {class_name} | "
                f"Confidence: {confidence:.2f}"
            )

        # ---------------------------------
        # Center point
        # ---------------------------------

        x1, y1, x2, y2 = box

        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        self.track_history[track_id].append(
            (center_x, center_y)
        )

        # ---------------------------------
        # Determine side
        # ---------------------------------

        current_side = get_side(
            center_y,
            line_y,
            line_margin
        )

        # ---------------------------------
        # Crossing
        # ---------------------------------

        if track_id in self.seen_ids:

            previous_side = self.track_side.get(
                track_id
            )

            if current_side is not None:

                event = crossing_event(
                    previous_side,
                    current_side,
                    track_id in self.entered_ids,
                    track_id in self.exited_ids
                )

                if event == "entered":

                    self.entered_count += 1
                    self.entered_ids.add(track_id)

                    print(
                        f"ENTERED -> "
                        f"ID: {track_id} | "
                        f"Class: {class_name}"
                    )

                elif event == "exited":

                    self.exited_count += 1
                    self.exited_ids.add(track_id)

                    print(
                        f"EXITED -> "
                        f"ID: {track_id} | "
                        f"Class: {class_name}"
                    )

                self.track_side[track_id] = (
                    current_side
                )


    def cleanup(self, frame_number):

        tracks_to_remove = []

        for track_id, last_frame in self.last_seen_frame.items():

            if (
                frame_number - last_frame
                > self.timeout_frames
            ):

                tracks_to_remove.append(
                    track_id
                )

        for track_id in tracks_to_remove:

            self.track_history.pop(
                track_id,
                None
            )

            self.track_hits.pop(
                track_id,
                None
            )

            self.last_seen_frame.pop(
                track_id,
                None
            )

            self.track_side.pop(
                track_id,
                None
            )

            print(
                f"REMOVED INACTIVE TRACK -> "
                f"ID: {track_id}"
            )


def get_side(center_y, line_y, margin):

    if center_y < line_y - margin:
        return "above"

    if center_y > line_y + margin:
        return "below"

    return None


def crossing_event(
    previous_side,
    current_side,
    already_entered,
    already_exited
):

    if (
        previous_side == "above"
        and current_side == "below"
        and not already_entered
    ):
        return "entered"

    if (
        previous_side == "below"
        and current_side == "above"
        and not already_exited
    ):
        return "exited"

    return None