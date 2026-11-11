from mmpretrain import ImageClassificationInferencer
from mmengine import Config
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="timm.models.layers")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.functional")
from flask import Flask, request, jsonify,send_file
from PIL import Image
import os,io
import numpy as np
import cv2

cfg = Config.fromfile(r'./configs/mtp/resisc45/vit-rvsa-l-224-mae-mtp_nwpu28.py')
checkpoint_path = 'resisc45-rvsa-l-mae-mtp-epoch_200.pth'
inferencer = ImageClassificationInferencer(model=cfg,pretrained=checkpoint_path)
app = Flask(__name__)
@app.route('/class_english', methods=['POST','GET'])
def run():
    image = request.data
    img_data = Image.open(io.BytesIO(image))
    image_np = np.array(img_data)

    if image_np.shape[2] == 4:
        image_np = image_np[:, :, :3]

    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    #image = '00321.jpg'


    result = inferencer(image_bgr)[0]
    #print(result)
    class_name = {
        0: 'airport',
        1: 'airport',
        2: 'sports field',
        3: 'sports field',
        4: 'beach',
        5: 'bridge and river',
        6: 'shrubland',
        7: 'church',
        8: 'farmland',
        9: 'cloud',
        10: 'commercial area',
        11: 'dense residential area',
        12: 'desert',
        13: 'forest',
        14: 'highway',
        15: 'golf course',
        16: 'sports field',
        17: 'harbor',
        18: 'industrial area',
        19: 'intersection',
        20: 'island',
        21: 'lake',
        22: 'meadow',
        23: 'residential area with houses',
        24: 'residential area with houses',
        25: 'mountain',
        26: 'overpass',
        27: 'residential area with houses',
        28: 'parking lot',
        29: 'railway',
        30: 'railway station',
        31: 'farmland',
        32: 'river',
        33: 'intersection',
        34: 'runway',
        35: 'sea ice',
        36: 'ship',
        37: 'snowy hill',
        38: 'residential area with houses',
        39: 'sports field',
        40: 'tank farm',
        41: 'stadium',
        42: 'terraced field',
        43: 'nuclear power plant',
        44: 'wetland'
    }

    index=result['pred_label']
    #print(result['pred_scores'][index])
    #print(class_name[index])
    content='This is an image of '+class_name[index]
    return content

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=1203)