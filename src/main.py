import cv2
import time
from collections import defaultdict
from ultralytics import YOLO


# Load YOLO model
model = YOLO("yolo26n.pt")

# Open webcam
cap = cv2.VideoCapture(0)

# Store every tracking ID we have already counted
seen_ids = set()

# Store counts for each class
class_counts = defaultdict(int)

# Used to calculate FPS
previous_time = time.perf_counter()


while cap.isOpened():

    success, frame = cap.read()

    if not success:
        print("Could not read frame from camera.")
        break

    # Run YOLO + ByteTrack
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False
    )

    result = results[0]

    # Check if anything is currently being tracked
    if result.boxes.id is not None:

        track_ids = result.boxes.id.int().cpu().tolist()
        classes = result.boxes.cls.int().cpu().tolist()
        confidences = result.boxes.conf.cpu().tolist()

        for track_id, class_id, confidence in zip(
            track_ids,
            classes,
            confidences
        ):

            class_name = model.names[class_id]

            # Only count this tracking ID once
            if track_id not in seen_ids:

                seen_ids.add(track_id)

                class_counts[class_name] += 1

                print(
                    f"NEW OBJECT -> "
                    f"ID: {track_id} | "
                    f"Class: {class_name} | "
                    f"Confidence: {confidence:.2f}"
                )

    # Let YOLO draw its normal boxes
    annotated_frame = result.plot()

    # -------------------------
    # Calculate FPS
    # -------------------------

    current_time = time.perf_counter()

    elapsed_time = current_time - previous_time

    if elapsed_time > 0:
        fps = 1 / elapsed_time
    else:
        fps = 0

    previous_time = current_time

    # -------------------------
    # Draw statistics
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

    # Starting position for class counters
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

    # Display window
    cv2.imshow(
        "Real-Time Vision Tracker",
        annotated_frame
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()