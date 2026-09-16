import csv
import time
from pathlib import Path


class Benchmark:

    def __init__(self):

        self.start_time = time.perf_counter()

        self.frame_count = 0

        self.total_processing_time = 0

        self.total_detections = 0


    def record_frame(
        self,
        processing_time,
        detections
    ):

        self.frame_count += 1

        self.total_processing_time += (
            processing_time
        )

        self.total_detections += detections


    def summary(self):

        if self.frame_count == 0:
            return {}

        average_frame_time = (
            self.total_processing_time
            / self.frame_count
        )

        average_fps = (
            1 / average_frame_time
            if average_frame_time > 0
            else 0
        )

        average_detections = (
            self.total_detections
            / self.frame_count
        )

        return {
            "frames": self.frame_count,
            "average_fps": average_fps,
            "average_frame_ms":
                average_frame_time * 1000,
            "average_detections":
                average_detections
        }


    def save_csv(
        self,
        filename,
        model,
        tracker,
        confidence
    ):

        data = self.summary()

        path = Path(filename)

        path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        file_exists = path.exists()

        with open(
            path,
            "a",
            newline=""
        ) as file:

            writer = csv.writer(file)

            if not file_exists:

                writer.writerow([
                    "model",
                    "tracker",
                    "confidence",
                    "frames",
                    "average_fps",
                    "average_frame_ms",
                    "average_detections"
                ])

            writer.writerow([
                model,
                tracker,
                confidence,
                data["frames"],
                f"{data['average_fps']:.2f}",
                f"{data['average_frame_ms']:.2f}",
                f"{data['average_detections']:.2f}"
            ])