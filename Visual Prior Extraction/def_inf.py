from collections import Counter
from pathlib import Path
from tempfile import NamedTemporaryFile
import io
import os
import warnings

import numpy as np
from PIL import Image
from flask import Flask, request
from mmdet.apis import DetInferencer
from mmengine import Config

from RS_Tasks_Finetune.Horizontal_Detection.yolo_plane import sim_plane
from RS_Tasks_Finetune.Horizontal_Detection.yolo_ship import sim_ship


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore", category=FutureWarning, module="timm.models.layers")
warnings.filterwarnings("ignore", message="Failed to add .*LocalVisBackend.*")
warnings.filterwarnings(
    "ignore",
    message="None of the inputs have requires_grad=True. Gradients will be None",
)
warnings.filterwarnings("ignore", category=UserWarning)


# ----------------------------------------------------------------------
# Model initialization
# ----------------------------------------------------------------------
cfg = Config.fromfile(
    "./configs/mtp/dior/faster_rcnn_rvsa_l_800_mae_mtp_dior.py"
)
checkpoint_path = "dior.pth"
inferencer = DetInferencer(
    model=cfg,
    weights=checkpoint_path,
    device="cuda:0",
)

app = Flask(__name__)

# DIOR category mapping: keep the mapping used by the original project.
CATEGORIES = {
    0: "airplane",
    1: "airport",
    2: "baseball diamond",
    3: "basketball court",
    4: "bridge",
    5: "chimney",
    6: "highway service area",
    7: "highway toll station",
    8: "dam",
    9: "golf course",
    10: "athletics field",
    11: "harbor",
    12: "footbridge",
    13: "ship",
    14: "stadium",
    15: "storage tank",
    16: "tennis court",
    17: "railway station",
    18: "car",
    19: "windmill",
}

SCORE_THRESHOLD = 0.4


def get_grid_region(center_x, center_y, image_width, image_height):
    """Map a bbox center to one of the 3 x 3 image regions."""
    col = min(int(center_x / image_width * 3), 2)
    row = min(int(center_y / image_height * 3), 2)

    vertical = ("upper", "middle", "lower")[row]
    horizontal = ("left", "center", "right")[col]

    if vertical == "middle" and horizontal == "center":
        return "center"
    if vertical == "middle":
        return horizontal
    if horizontal == "center":
        return vertical
    return f"{vertical} {horizontal}"


def get_size_level(bbox, image_width, image_height):
    """
    Paper Eq. (1):
        S_J = bbox_area / image_area * 100 (%)

    Paper thresholds:
        small:  S_J < 0.1
        medium: 0.1 <= S_J <= 0.2
        large:  S_J > 0.2
    """
    x1, y1, x2, y2 = map(float, bbox)
    bbox_width = max(0.0, x2 - x1)
    bbox_height = max(0.0, y2 - y1)

    area_ratio_percent = (
        bbox_width * bbox_height / (image_width * image_height) * 100.0
    )

    if area_ratio_percent < 0.1:
        size_level = "small"
    elif area_ratio_percent <= 0.2:
        size_level = "medium"
    else:
        size_level = "large"

    return size_level, area_ratio_percent


def classify_fine_grained(image_path, category, bbox):
    """Apply the dedicated fine-grained classifier to aircraft and ships."""
    x1, y1, x2, y2 = map(float, bbox)

    try:
        if category == "airplane":
            result = sim_plane(image_path, x1, y1, x2, y2)
            return str(result)
        if category == "ship":
            result = sim_ship(image_path, x1, y1, x2, y2)
            return str(result)
    except Exception as exc:
        # Keep the detection result even if the optional fine-grained
        # classifier fails for a particular crop.
        app.logger.warning("Fine-grained classification failed: %s", exc)

    return None


def build_detection_description(prediction, image_path, image_width, image_height):
    """
    Serialize detection outputs as required by the paper:
    category frequency + per-object category + center coordinates + size.
    """
    scores = prediction.get("scores", [])
    labels = prediction.get("labels", [])
    bboxes = prediction.get("bboxes", [])

    # Filter scores, labels, and boxes together. Do not count scores first
    # and then assume the retained detections occupy the first N indices.
    detections = []
    for score, label, bbox in zip(scores, labels, bboxes):
        score = float(score)
        if score <= SCORE_THRESHOLD:
            continue

        label = int(label)
        category = CATEGORIES.get(label, f"unknown_category_{label}")
        x1, y1, x2, y2 = map(float, bbox)

        # Clamp coordinates to the original image boundary.
        x1 = min(max(x1, 0.0), float(image_width))
        x2 = min(max(x2, 0.0), float(image_width))
        y1 = min(max(y1, 0.0), float(image_height))
        y2 = min(max(y2, 0.0), float(image_height))

        if x2 <= x1 or y2 <= y1:
            continue

        detections.append(
            {
                "category": category,
                "score": score,
                "bbox": (x1, y1, x2, y2),
            }
        )

    if not detections:
        return "No targets found."

    category_counts = Counter(item["category"] for item in detections)

    lines = [
        "Visual Detection Priors",
        f"Image size: width={image_width}, height={image_height}.",
        f"Detection confidence threshold: score > {SCORE_THRESHOLD}.",
        "",
        "Object Category Frequencies:",
    ]

    for category, count in sorted(category_counts.items()):
        lines.append(f"- {category}: {count}")

    lines.extend(["", "Specific Target Attributes:"])

    for index, item in enumerate(detections, start=1):
        x1, y1, x2, y2 = item["bbox"]
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0

        # Use pixel coordinates on the original, unwarped image.
        center_x_int = int(round(center_x))
        center_y_int = int(round(center_y))

        region = get_grid_region(
            center_x, center_y, image_width, image_height
        )
        size_level, area_ratio_percent = get_size_level(
            item["bbox"], image_width, image_height
        )

        line = (
            f"Target {index}: category={item['category']}; "
            f"center=({center_x_int}, {center_y_int}) pixels; "
            f"region={region}; "
            f"size={size_level}; "
            f"bbox_area_ratio={area_ratio_percent:.4f}%."
        )

        # The paper's RAG module invokes fine-grained recognition for
        # generic aircraft and ship detections.
        fine_grained_type = classify_fine_grained(
            image_path, item["category"], item["bbox"]
        )
        if fine_grained_type:
            line += f" Fine-grained category={fine_grained_type}."

        lines.append(line)

    return "\n".join(lines)


@app.route("/det_english", methods=["POST", "GET"])
def run():
    image_bytes = request.get_data()
    if not image_bytes:
        return "No image data received.", 400

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        return f"Invalid image: {exc}", 400

    image_width, image_height = image.size

    # Keep the original aspect ratio and resolution. The inferencer handles
    # model-specific resizing internally; this also keeps bbox coordinates
    # and the fine-grained classifier crop in the same coordinate system.
    temp_path = None
    try:
        with NamedTemporaryFile(
            suffix=".jpg", prefix="rs_detection_", delete=False
        ) as temp_file:
            temp_path = temp_file.name
            image.save(temp_path, format="JPEG", quality=95)

        result = inferencer(
            temp_path,
            no_save_vis=True,
            no_save_pred=True,
        )

        predictions = result.get("predictions", [])
        if not predictions:
            return "No targets found."

        prediction = predictions[0]
        description = build_detection_description(
            prediction,
            temp_path,
            image_width,
            image_height,
        )
        return description

    except Exception as exc:
        app.logger.exception("Detection inference failed")
        return f"Detection inference failed: {exc}", 500

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=1212)
