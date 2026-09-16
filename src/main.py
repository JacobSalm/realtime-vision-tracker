import argparse
import cv2
import time
from collections import defaultdict, deque
from pathlib import Path
from ultralytics import YOLO


# =========================================================
# COMMAND-LINE ARGUMENTS
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Real-Time YOLO Object Detection and Tracking"
    )

    parser.add_argument(
        "--source",
        default="0",
        help="Camera number such as 0, or path to a video file"
    )

    parser.add_argument(
        "--model",
        default="yolo26s.pt",
        help="YOLO model to use"
    )

    parser.add_argument(
        "--tracker",
        default="tracktrack.yaml",
        help="Tracker configuration"
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.45,
        help="Detection confidence threshold"
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=0.50,
        help="IoU threshold"
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="YOLO inference image size"
    )

    parser.add_argument(
        "--save",
        action="store_true",
        help="Save processed video"
    )

    parser.add_argument(
        "--output",
        default="outputs/tracked_output.mp4",
        help="Output video path"
    )

    return parser.parse_args()


args = parse_args()


# =========================================================
# SOURCE HANDLING
# =========================================================

# If source is "0", "1", etc., convert it into an integer
# so OpenCV knows it is a camera.
if args.source.isdigit():
    video_source = int(args.source)
else:
    video_source = args.source


# =========================================================
# CONFIGURATION
# =========================================================

MIN_CONFIRM_FRAMES = 5

LINE_POSITION = 0.55
LINE_MARGIN = 15

TRACK_TIMEOUT_FRAMES = 90

TRAIL_LENGTH = 30


# =========================================================
# SETUP
# =========================================================

model = YOLO(args.model)

cap = cv2.VideoCapture(video_source)

if not cap.isOpened():
    print(f"ERROR: Could not open source: {args.source}")
    raise SystemExit(1)


# =========================================================
# VIDEO OUTPUT SETUP
# =========================================================

video_writer = None

if args.save:

    output_path = Path(args.output)

    # Make outputs folder automatically if needed
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    print(f"Output will be saved to: {output_path}")


# =========================================================
# TRACKING MEMORY
# =========================================================

seen_ids = set()

class_counts = defaultdict(int)

track_hits = defaultdict(int)

track_history = defaultdict(
    lambda: deque(maxlen=TRAIL_LENGTH)
)

last_seen_frame = {}

track_side = {}

entered_ids = set()
exited_ids = set()

entered_count = 0
exited_count = 0

frame_number = 0

previous_time = time.perf_counter()


# =========================================================
# MAIN LOOP
# =========================================================

while cap.isOpened():

    success, frame = cap.read()

    if not success:
        print("End of video or could not read frame.")
        break

    frame_number += 1

    frame_height, frame_width = frame.shape[:2]

    line_y = int(
        frame_height * LINE_POSITION
    )


    # =====================================================
    # YOLO + TRACKING
    # =====================================================

    results = model.track(
        frame,
        persist=True,
        tracker=args.tracker,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        verbose=False
    )

    result = results[0]


    # =====================================================
    # PROCESS TRACKED OBJECTS
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

            last_seen_frame[track_id] = frame_number

            track_hits[track_id] += 1


            # =============================================
            # CONFIRM OBJECT
            # =============================================

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


            # =============================================
            # CENTER POSITION
            # =============================================

            x1, y1, x2, y2 = box

            center_x = int(
                (x1 + x2) / 2
            )

            center_y = int(
                (y1 + y2) / 2
            )

            track_history[track_id].append(
                (center_x, center_y)
            )


            # =============================================
            # SIDE OF COUNTING LINE
            # =============================================

            current_side = None

            if center_y < line_y - LINE_MARGIN:
                current_side = "above"

            elif center_y > line_y + LINE_MARGIN:
                current_side = "below"


            # =============================================
            # CROSSING DETECTION
            # =============================================

            if track_id in seen_ids:

                previous_side = track_side.get(
                    track_id
                )

                if current_side is not None:

                    # ENTERED
                    if (
                        previous_side == "above"
                        and current_side == "below"
                        and track_id not in entered_ids
                    ):

                        entered_count += 1

                        entered_ids.add(
                            track_id
                        )

                        print(
                            f"ENTERED -> "
                            f"ID: {track_id} | "
                            f"Class: {class_name}"
                        )


                    # EXITED
                    elif (
                        previous_side == "below"
                        and current_side == "above"
                        and track_id not in exited_ids
                    ):

                        exited_count += 1

                        exited_ids.add(
                            track_id
                        )

                        print(
                            f"EXITED -> "
                            f"ID: {track_id} | "
                            f"Class: {class_name}"
                        )


                    track_side[track_id] = (
                        current_side
                    )


    # =========================================================
    # CLEAN UP OLD TRACKS
    # =========================================================

    tracks_to_remove = []

    for track_id, last_frame in last_seen_frame.items():

        frames_missing = (
            frame_number - last_frame
        )

        if frames_missing > TRACK_TIMEOUT_FRAMES:

            tracks_to_remove.append(
                track_id
            )


    for track_id in tracks_to_remove:

        track_history.pop(
            track_id,
            None
        )

        track_hits.pop(
            track_id,
            None
        )

        last_seen_frame.pop(
            track_id,
            None
        )

        track_side.pop(
            track_id,
            None
        )

        print(
            f"REMOVED INACTIVE TRACK -> "
            f"ID: {track_id}"
        )


    # =========================================================
    # DRAW YOLO RESULTS
    # =========================================================

    annotated_frame = result.plot()


    # =========================================================
    # MOVEMENT TRAILS
    # =========================================================

    for track_id, points in track_history.items():

        if len(points) < 2:
            continue

        points_list = list(points)

        for i in range(
            1,
            len(points_list)
        ):

            cv2.line(
                annotated_frame,
                points_list[i - 1],
                points_list[i],
                (255, 255, 255),
                2
            )


    # =========================================================
    # COUNTING LINE
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
        (
            frame_width - 190,
            line_y - 10
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2
    )


    # =========================================================
    # FPS
    # =========================================================

    current_time = time.perf_counter()

    elapsed_time = (
        current_time - previous_time
    )

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
    # CREATE VIDEO WRITER
    # =========================================================

    if args.save and video_writer is None:

        source_fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        # Some webcams report FPS incorrectly
        if source_fps <= 0:
            source_fps = 30.0

        fourcc = cv2.VideoWriter_fourcc(
            *"mp4v"
        )

        video_writer = cv2.VideoWriter(
            args.output,
            fourcc,
            source_fps,
            (
                frame_width,
                frame_height
            )
        )

        if not video_writer.isOpened():
            print(
                "ERROR: Could not create "
                "output video."
            )

            video_writer = None


    # =========================================================
    # SAVE FRAME
    # =========================================================

    if video_writer is not None:

        video_writer.write(
            annotated_frame
        )


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
# FINAL CLEANUP
# =========================================================

cap.release()

if video_writer is not None:
    video_writer.release()

cv2.destroyAllWindows()


# =========================================================
# FINAL SUMMARY
# =========================================================

print()
print("===== SESSION SUMMARY =====")
print(
    f"Unique Objects: {len(seen_ids)}"
)
print(
    f"Entered: {entered_count}"
)
print(
    f"Exited: {exited_count}"
)

for class_name, count in class_counts.items():
    print(
        f"{class_name}: {count}"
    )

if args.save:
    print(
        f"Saved video: {args.output}"
    )