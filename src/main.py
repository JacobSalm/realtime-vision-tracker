from ultralytics import YOLO

model = YOLO("yolo26n.pt")

# Run YOLO + ByteTrack on the webcam
model.track(
    source=0,
    show=True,
    tracker="custom_bytetrack.yaml"
)