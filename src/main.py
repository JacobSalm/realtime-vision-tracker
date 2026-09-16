from ultralytics import YOLO

model = YOLO("yolo26n.pt")

results = model.track(
    source=0,
    show=True,
    tracker="custom_bytetrack.yaml",
    stream=True
)

for result in results:
    pass