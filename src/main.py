import cv2
from ultralytics import YOLO

model = YOLO("yolo26n.pt")

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()

    if not success:
        print("Could not read frame from camera.")
        break

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False
    )

    result = results[0]

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

            print(
                f"ID: {track_id} | "
                f"Class: {class_name} | "
                f"Confidence: {confidence:.2f}"
            )

    annotated_frame = result.plot()

    cv2.imshow("Real-Time Vision Tracker", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()