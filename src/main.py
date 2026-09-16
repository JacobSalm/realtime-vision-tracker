import cv2
import time
from collections import defaultdict, deque
from ultralytics import YOLO


# =========================================================
# CONFIGURATION
# =========================================================

MODEL_NAME = "yolo26s.pt"

CONFIDENCE_THRESHOLD = 0.45
IOU_THRESHOLD = 0.50

# Object must survive this many frames before we trust it
MIN_CONFIRM_FRAMES = 5

# Position of counting line as percentage of screen height
# 0.50 = exact middle
LINE_POSITION = 0.55

# Creates a small dead-zone around the line.
# Helps prevent tiny movements from being counted as crossings.
LINE_MARGIN = 15

# Remove inactive track data after this many frames.
# At ~30 FPS, 90 frames is about 3 seconds.
TRACK_TIMEOUT_FRAMES = 90

# Length of movement trail
TRAIL_LENGTH = 30


# =========================================================
# SETUP
# =========================================================

model = YOLO(MODEL_NAME)

cap = cv2.VideoCapture(0)


# =========================================================
# TRACKING / ANALYTICS MEMORY
# =========================================================

# Confirmed IDs we've seen
seen_ids = set()

# Counts by object class
class_counts = defaultdict(int)

# How many frames each track has appeared
track_hits = defaultdict(int)

# Movement history
track_history = defaultdict(
    lambda: deque(maxlen=TRAIL_LENGTH)
)

# Last frame where each ID was detected
last_seen_frame = {}

# Which side of the line each object was last on
track_side = {}

# Used to prevent multiple counts for the same ID
entered_ids = set()
exited_ids = set()

# Counters
entered_count = 0
exited_count = 0

# Current video frame number
frame_number = 0

# FPS calculation
previous_time = time.perf_counter()


# =========================================================
# MAIN LOOP
# =========================================================

while cap.isOpened():

    success, frame = cap.read()

    if not success:
        print("Could not read frame from camera.")
        break

    frame_number += 1

    frame_height, frame_width = frame.shape[:2]

    # Calculate counting-line position
    line_y = int(frame_height * LINE_POSITION)


    # =====================================================
    # YOLO + TRACKING
    # =====================================================

    results = model.track(
        frame,
        persist=True,
        tracker="tracktrack.yaml",
        conf=CONFIDENCE_THRESHOLD,
        iou=IOU_THRESHOLD,
        imgsz=640,
        verbose=False
    )

    result = results[0]


    # =====================================================
    # PROCESS OBJECTS
    # =====================================================

    if result.boxes.id is not None:

        track_ids = (
            result.boxes.id
            .int()
            .cpu()
            .tolist()
        )

        classes = (
            result.boxes.cls
            .int()
            .cpu()
            .tolist()
        )

        confidences = (
            result.boxes.conf
            .cpu()
            .tolist()
        )

        boxes = (
            result.boxes.xyxy
            .cpu()
            .tolist()
        )


        for track_id, class_id, confidence, box in zip(
            track_ids,
            classes,
            confidences,
            boxes
        ):

            class_name = model.names[class_id]

            # Remember the last frame this ID existed
            last_seen_frame[track_id] = frame_number

            # Count number of frames this ID has survived
            track_hits[track_id] += 1


            # =================================================
            # CONFIRM OBJECT
            # =================================================

            if (
                track_hits[track_id] >= MIN_CONFIRM_FRAMES
                and track_id not in seen_ids
            ):

                seen_ids.add(track_id)

                class_counts[class_name] += 1

                print(
                    f"CONFIRMED OBJECT -> "
                    f"ID: {track_id} | "
                    f"Class: {class_name} | "
                    f"Confidence: {confidence:.2f}"
                )


            # =================================================
            # CENTER POSITION
            # =================================================

            x1, y1, x2, y2 = box

            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            track_history[track_id].append(
                (center_x, center_y)
            )


            # =================================================
            # DETERMINE WHICH SIDE OF LINE
            # =================================================

            current_side = None

            # Above line
            if center_y < line_y - LINE_MARGIN:
                current_side = "above"

            # Below line
            elif center_y > line_y + LINE_MARGIN:
                current_side = "below"


            # =================================================
            # CROSSING DETECTION
            # =================================================

            # Only count trusted/confirmed tracks
            if track_id in seen_ids:

                previous_side = track_side.get(track_id)

                # Only process if object is outside dead-zone
                if current_side is not None:

                    # -----------------------------------------
                    # ENTERED
                    # Above -> Below
                    # -----------------------------------------

                    if (
                        previous_side == "above"
                        and current_side == "below"
                        and track_id not in entered_ids
                    ):

                        entered_count += 1
                        entered_ids.add(track_id)

                        print(
                            f"ENTERED -> "
                            f"ID: {track_id} | "
                            f"Class: {class_name}"
                        )


                    # -----------------------------------------
                    # EXITED
                    # Below -> Above
                    # -----------------------------------------

                    elif (
                        previous_side == "below"
                        and current_side == "above"
                        and track_id not in exited_ids
                    ):

                        exited_count += 1
                        exited_ids.add(track_id)

                        print(
                            f"EXITED -> "
                            f"ID: {track_id} | "
                            f"Class: {class_name}"
                        )


                    # Save current side
                    track_side[track_id] = current_side


    # =========================================================
    # TRACK CLEANUP
    # =========================================================

    tracks_to_remove = []

    for track_id, last_frame in last_seen_frame.items():

        frames_missing = frame_number - last_frame

        if frames_missing > TRACK_TIMEOUT_FRAMES:
            tracks_to_remove.append(track_id)


    for track_id in tracks_to_remove:

        # Remove temporary tracking information
        track_history.pop(track_id, None)
        track_hits.pop(track_id, None)
        last_seen_frame.pop(track_id, None)
        track_side.pop(track_id, None)

        print(
            f"REMOVED INACTIVE TRACK -> ID: {track_id}"
        )


    # =========================================================
    # DRAW YOLO DETECTIONS
    # =========================================================

    annotated_frame = result.plot()


    # =========================================================
    # DRAW MOVEMENT TRAILS
    # =========================================================

    for track_id, points in track_history.items():

        if len(points) < 2:
            continue

        points_list = list(points)

        for i in range(1, len(points_list)):

            cv2.line(
                annotated_frame,
                points_list[i - 1],
                points_list[i],
                (255, 255, 255),
                2
            )


    # =========================================================
    # DRAW COUNTING LINE
    # =========================================================

    cv2.line(
        annotated_frame,
        (0, line_y),
        (frame_width, line_y),
        (0, 255, 255),
        2
    )

    cv2.putText(
        annotated_frame,
        "COUNTING LINE",
        (frame_width - 190, line_y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2
    )


    # =========================================================
    # FPS
    # =========================================================

    current_time = time.perf_counter()

    elapsed_time = current_time - previous_time

    if elapsed_time > 0:
        fps = 1 / elapsed_time
    else:
        fps = 0

    previous_time = current_time


    # =========================================================
    # STATISTICS
    # =========================================================

    cv2.putText(
        annotated_frame,
        f"FPS: {fps:.1f}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        annotated_frame,
        f"Unique Objects: {len(seen_ids)}",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        annotated_frame,
        f"Entered: {entered_count}",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        annotated_frame,
        f"Exited: {exited_count}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    # Class counts
    y_position = 150

    for class_name, count in class_counts.items():

        cv2.putText(
            annotated_frame,
            f"{class_name}: {count}",
            (20, y_position),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        y_position += 30


    # =========================================================
    # DISPLAY
    # =========================================================

    cv2.imshow(
        "Real-Time Vision Tracker",
        annotated_frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# =========================================================
# CLEANUP
# =========================================================

cap.release()
cv2.destroyAllWindows()