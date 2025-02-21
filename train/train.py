from ultralytics import YOLO

# Load a COCO-pretrained YOLO11n model
model = YOLO("yolov8x-worldv2.pt")
model = YOLO("yolov8l-worldv2.pt")
model = YOLO("yolov8m-worldv2.pt")
model = YOLO("yolov8s-worldv2.pt")
# model = YOLO("yolov8n-worldv2.pt")

# Train the model on the COCO8 example dataset for 100 epochs
# results = model.train(data=r"D:\code\train\ultralytics\ultralytics\cfg\datasets\open-images-v7.yaml", epochs=100, imgsz=640)

# Run inference with the YOLO11n model on the 'bus.jpg' image
# results = model("path/to/bus.jpg")