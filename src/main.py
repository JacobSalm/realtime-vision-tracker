import cv2
import time
from collections import defaultdict, deque
from ultralytics import YOLO


# -------------------------
# Configuration
# -------------------------

MODEL_NAME = "yolo26s.pt"

CONFIDENCE_THRESHOLD = 0.45
IOU_THRESHOLD = 0.50

MIN_CONFIRM_FRAMES = 5


# -------------------------
# Setup
# -------------------------

# Load YOLO model
model = YOLO(MODEL_NAME)

# Open webcam
cap = cv2.VideoCapture(0)

# IDs that have been officially confirmed/countable
seen_ids = set()

# Number of frames each ID has appeared in
track_hits = defaultdict(int)

# Count confirmed objects by class
class_counts = defaultdict(int)

# Movement history for each ID
track_history = defaultdict(
    lambda: deque(maxlen=30)
)

# FPS timer
previous_time = time.perf_counter()


# -------------------------
# Main loop
# -------------------------

while cap.isOpened():

    success, frame = cap.read()

    if not success:
        print("Could not read frame from camera.")
        break

    # Run YOLO + TrackTrack
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

    # -------------------------
    # Process tracked objects
    # -------------------------

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

            # -------------------------
            # Confirmation system
            # -------------------------

            track_hits[track_id] += 1

            # Only count an object after it has survived
            # for several frames
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

            # -------------------------
            # Calculate center point
            # -------------------------

            x1, y1, x2, y2 = box

            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            track_history[track_id].append(
                (center_x, center_y)
            )

    # -------------------------
    # Draw YOLO detections
    # -------------------------

    annotated_frame = result.plot()

    # -------------------------
    # Draw movement trails
    # -------------------------

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

    # -------------------------
    # FPS calculation
    # -------------------------

    current_time = time.perf_counter()

    elapsed_time = current_time - previous_time

    if elapsed_time > 0:
        fps = 1 / elapsed_time
    else:
        fps = 0

    previous_time = current_time

    # -------------------------
    # Statistics overlay
    # -------------------------

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

    y_position = 90

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

    # -------------------------
    # Display
    # -------------------------

    cv2.imshow(
        "Real-Time Vision Tracker",
        annotated_frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()