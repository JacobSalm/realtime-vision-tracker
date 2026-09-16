import argparse
import cv2
import time

from pathlib import Path
from ultralytics import YOLO

from analytics import Analytics
from visualization import (
    draw_trails,
    draw_counting_line,
    draw_stats
)
from benchmark import Benchmark


def parse_args():

    parser = argparse.ArgumentParser(
        description="Real-Time Vision Tracker"
    )

    parser.add_argument(
        "--source",
        default="0"
    )

    parser.add_argument(
        "--model",
        default="yolo26s.pt"
    )

    parser.add_argument(
        "--tracker",
        default="tracktrack.yaml"
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.45
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=0.50
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640
    )

    parser.add_argument(
        "--save",
        action="store_true"
    )

    parser.add_argument(
        "--output",
        default="outputs/tracked_output.mp4"
    )

    parser.add_argument(
        "--benchmark",
        action="store_true"
    )

    parser.add_argument(
        "--benchmark-output",
        default="outputs/benchmarks.csv"
    )

    return parser.parse_args()


args = parse_args()


# -----------------------------------------
# Source
# -----------------------------------------

if args.source.isdigit():
    source = int(args.source)
else:
    source = args.source


# -----------------------------------------
# Configuration
# -----------------------------------------

LINE_POSITION = 0.55
LINE_MARGIN = 15


# -----------------------------------------
# Setup
# -----------------------------------------

model = YOLO(args.model)

cap = cv2.VideoCapture(source)

if not cap.isOpened():

    print(
        f"ERROR: Could not open "
        f"source {args.source}"
    )

    raise SystemExit(1)


analytics = Analytics()

benchmark = Benchmark()

video_writer = None

previous_time = time.perf_counter()

frame_number = 0


# -----------------------------------------
# Main loop
# -----------------------------------------

while cap.isOpened():

    success, frame = cap.read()

    if not success:
        break

    frame_number += 1

    frame_height, frame_width = (
        frame.shape[:2]
    )

    line_y = int(
        frame_height * LINE_POSITION
    )

    processing_start = (
        time.perf_counter()
    )


    # -------------------------------------
    # YOLO + tracker
    # -------------------------------------

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


    # -------------------------------------
    # Analytics
    # -------------------------------------

    detection_count = 0

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

        detection_count = len(track_ids)

        for (
            track_id,
            class_id,
            confidence,
            box
        ) in zip(
            track_ids,
            classes,
            confidences,
            boxes
        ):

            analytics.process_object(
                track_id,
                model.names[class_id],
                confidence,
                box,
                frame_number,
                line_y,
                LINE_MARGIN
            )


    analytics.cleanup(
        frame_number
    )


    # -------------------------------------
    # Visualization
    # -------------------------------------

    annotated_frame = result.plot()

    draw_trails(
        annotated_frame,
        analytics.track_history
    )

    draw_counting_line(
        annotated_frame,
        line_y
    )


    current_time = (
        time.perf_counter()
    )

    elapsed = (
        current_time
        - previous_time
    )

    fps = (
        1 / elapsed
        if elapsed > 0
        else 0
    )

    previous_time = current_time


    draw_stats(
        annotated_frame,
        fps,
        analytics
    )


    # -------------------------------------
    # Benchmark
    # -------------------------------------

    processing_time = (
        time.perf_counter()
        - processing_start
    )

    benchmark.record_frame(
        processing_time,
        detection_count
    )


    # -------------------------------------
    # Save video
    # -------------------------------------

    if args.save and video_writer is None:

        output_path = Path(
            args.output
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        source_fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        if source_fps <= 0:
            source_fps = 30

        fourcc = (
            cv2.VideoWriter_fourcc(
                *"mp4v"
            )
        )

        video_writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            source_fps,
            (
                frame_width,
                frame_height
            )
        )


    if video_writer is not None:

        video_writer.write(
            annotated_frame
        )


    # -------------------------------------
    # Display
    # -------------------------------------

    cv2.imshow(
        "Real-Time Vision Tracker",
        annotated_frame
    )

    if (
        cv2.waitKey(1) & 0xFF
        == ord("q")
    ):
        break


# -----------------------------------------
# Cleanup
# -----------------------------------------

cap.release()

if video_writer is not None:
    video_writer.release()

cv2.destroyAllWindows()


# -----------------------------------------
# Final results
# -----------------------------------------

print()
print("===== SESSION SUMMARY =====")

print(
    f"Unique Objects: "
    f"{len(analytics.seen_ids)}"
)

print(
    f"Entered: "
    f"{analytics.entered_count}"
)

print(
    f"Exited: "
    f"{analytics.exited_count}"
)


summary = benchmark.summary()

print()
print("===== PERFORMANCE =====")

print(
    f"Frames: "
    f"{summary['frames']}"
)

print(
    f"Average FPS: "
    f"{summary['average_fps']:.2f}"
)

print(
    f"Average frame time: "
    f"{summary['average_frame_ms']:.2f} ms"
)


if args.benchmark:

    benchmark.save_csv(
        args.benchmark_output,
        args.model,
        args.tracker,
        args.conf
    )

    print(
        f"Benchmark saved to: "
        f"{args.benchmark_output}"
    )