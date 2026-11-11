from ultralytics import YOLO
from PIL import Image


def sim_ship(image_path,b1,b2,b3,b4):
    class_name=[
        "Aircraft carrier",
        "Amphibious assault ship",
        "Amphibious transport dock",
        "Cruiser",
        "Dry Cargo Ship",
        "Liquid Cargo Ship",
        "Motorboat",
        "Tugboat"
    ]
    query_image = Image.open(image_path)
    query_image = query_image.crop((b1, b2, b3, b4))
    # Load a model
    model = YOLO(r"F:\pythonProject\MTP-main\YOLO\runs\classify\train2\weights\best.pt")  # pretrained YOLOv8n model

    # Run batched inference on a list of images
    results = model(query_image,imgsz=320)  # return a list of Results objects

    # Process results list
    for result in results:
        probs = result.probs  # 分类
        return class_name[probs.top1]