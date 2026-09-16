import cv2


def draw_trails(frame, track_history):

    for points in track_history.values():

        if len(points) < 2:
            continue

        points_list = list(points)

        for i in range(1, len(points_list)):

            cv2.line(
                frame,
                points_list[i - 1],
                points_list[i],
                (255, 255, 255),
                2
            )


def draw_counting_line(
    frame,
    line_y
):

    height, width = frame.shape[:2]

    cv2.line(
        frame,
        (0, line_y),
        (width, line_y),
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "COUNTING LINE",
        (max(10, width - 190), line_y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2
    )


def draw_stats(
    frame,
    fps,
    analytics
):

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Unique Objects: {len(analytics.seen_ids)}",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Entered: {analytics.entered_count}",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Exited: {analytics.exited_count}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    y_position = 150

    for class_name, count in analytics.class_counts.items():

        cv2.putText(
            frame,
            f"{class_name}: {count}",
            (20, y_position),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        y_position += 30