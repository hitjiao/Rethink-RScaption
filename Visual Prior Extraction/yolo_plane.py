from ultralytics import YOLO
from PIL import Image


def sim_plane(image_path,b1,b2,b3,b4):
    class_name=[
        "Anti-submarine aircraft P-3C",
        "AWACS E-3",
        "AWACS E-8",
        "Bomber B-1B",
        "Bomber B-52",
        "Bomber SU-24",
        "Bomber SU-34",
        "Bomber SU-35",
        "Bomber TU-22",
        "Bomber TU-95",
        "Bomber TU-160",
        "Civil airliner ARJ21",
        "Civil airliner Boeing 747",
        "Civil airliner Boeing 777",
        "Fighter Jet F-15",
        "Fighter Jet F-16",
        "Fighter Jet F-22",
        "Fighter Jet FA-18",
        "Stratotanker KC-10",
        "Stratotanker KC-135",
        "Transport aircraft C-5",
        "Transport aircraft C-17",
        "Transport aircraft C-130"
    ]
    query_image = Image.open(image_path)
    query_image = query_image.crop((b1, b2, b3, b4))
    # Load a model
    model = YOLO(r"F:\pythonProject\MTP-main\YOLO\runs\classify\train3\weights\best.pt")  # pretrained YOLOv8n model

    # Run batched inference on a list of images
    results = model(query_image,imgsz=320)  # return a list of Results objects

    # Process results list
    for result in results:
        probs = result.probs  # 分类
        return class_name[probs.top1]