import io
import os
import warnings

import cv2
import numpy as np
from PIL import Image
from flask import Flask, request
from mmengine import Config
from mmsegmentation.mmseg.apis import init_model, inference_model

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="timm.models.layers",
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="torch.functional",
)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# ---------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------
CONFIG_PATH = (
    "./configs/mtp/loveda/"
    "rvsa-l-upernet-512-mae-mtp-loveda.py"
)
CHECKPOINT_PATH = "loveda-rvsa-l-mae-mtp-iter_80000.pth"
DEVICE = "cuda:0"

# LoveDA semantic label IDs:
# 0 background, 1 building, 2 road, 3 water,
# 4 barren, 5 forest, 6 agriculture.
# Background is excluded from the textual land-cover description.
CLASS_NAMES = {
    1: "building",
    2: "road",
    3: "water",
    4: "barren land",
    5: "forest",
    6: "agricultural land",
}

# Paper: divide the mask into a 3 x 3 grid.
REGION_NAMES = [
    ["upper-left", "top", "upper-right"],
    ["left", "center", "right"],
    ["lower-left", "bottom", "lower-right"],
]

# Paper's empirical threshold: a category is present in a grid cell
# only when its occupancy ratio is strictly greater than 0.1.
REGION_OCCUPANCY_THRESHOLD = 0.1


cfg = Config.fromfile(CONFIG_PATH)
model = init_model(cfg, CHECKPOINT_PATH, DEVICE)

app = Flask(__name__)


def decode_request_image(raw_bytes):
    """Decode uploaded image bytes into an OpenCV BGR image."""
    if not raw_bytes:
        raise ValueError("Request body is empty; send image bytes in the POST body.")

    image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    image_rgb = np.asarray(image)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    return image_bgr


def get_segmentation_mask(image_bgr):
    """Run MMSeg inference and return a 2-D integer label mask."""
    result = inference_model(model, image_bgr)
    pred_mask = result.pred_sem_seg.data[0].detach().cpu().numpy()
    pred_mask = np.asarray(pred_mask).squeeze()

    if pred_mask.ndim != 2:
        raise ValueError(
            f"Expected a 2-D semantic mask, got shape {pred_mask.shape}."
        )

    return pred_mask.astype(np.int32, copy=False)


def serialize_segmentation_prior(pred_mask):
    """
    Serialize the segmentation mask according to the paper.

    For each land-cover category c and grid cell G(i,j):
        P_c(i,j) = N_c(i,j) / |G(i,j)|

    A category is reported in a cell only if P_c(i,j) > 0.1.
    """
    height, width = pred_mask.shape
    total_pixels = height * width

    # np.array_split handles image dimensions that are not divisible by 3.
    row_splits = np.array_split(np.arange(height), 3)
    col_splits = np.array_split(np.arange(width), 3)

    lines = [
        "Semantic Segmentation Priors",
        f"Segmentation mask size: {width} x {height}",
        "Regional composition (3 x 3 grid):",
    ]

    found_land_cover = False

    for class_id, class_name in CLASS_NAMES.items():
        class_mask = (pred_mask == class_id)
        class_pixel_count = int(np.count_nonzero(class_mask))

        if class_pixel_count == 0:
            continue

        found_land_cover = True
        global_ratio = class_pixel_count / total_pixels

        # Overall class proportion is supplementary descriptive information.
        # The paper's regional-presence decision is based on per-cell ratios.
        lines.append(
            f"- {class_name}: {global_ratio:.1%} of the full image"
        )

        present_regions = []

        for row_idx, row_indices in enumerate(row_splits):
            for col_idx, col_indices in enumerate(col_splits):
                cell_mask = class_mask[
                    np.ix_(row_indices, col_indices)
                ]
                cell_pixel_count = int(np.count_nonzero(cell_mask))
                cell_pixel_total = int(cell_mask.size)

                if cell_pixel_total == 0:
                    continue

                occupancy_ratio = cell_pixel_count / cell_pixel_total

                if occupancy_ratio > REGION_OCCUPANCY_THRESHOLD:
                    region_name = REGION_NAMES[row_idx][col_idx]
                    present_regions.append(
                        f"{region_name} ({occupancy_ratio:.1%})"
                    )

        if present_regions:
            lines.append(
                "  Dominant regions (>10% cell occupancy): "
                + ", ".join(present_regions)
            )
        else:
            lines.append(
                "  No grid cell exceeds the 10% occupancy threshold."
            )

    if not found_land_cover:
        lines.append("No foreground land-cover classes were detected.")

    return "\n".join(lines)


@app.route("/seg_english", methods=["POST"])
def run_segmentation():
    try:
        image_bgr = decode_request_image(request.get_data())
        pred_mask = get_segmentation_mask(image_bgr)
        response_text = serialize_segmentation_prior(pred_mask)
        return response_text, 200, {"Content-Type": "text/plain; charset=utf-8"}

    except Exception as exc:
        app.logger.exception("Segmentation request failed")
        return f"Segmentation failed: {exc}", 400, {
            "Content-Type": "text/plain; charset=utf-8"
        }


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=1214)
