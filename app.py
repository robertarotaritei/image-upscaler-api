from flask import Flask, request
from realesrgan import RealESRGANer
from gfpgan import GFPGANer
import cv2
import base64
import numpy as np
from flask_cors import CORS

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
CORS(app)
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file present'
    bg_upsampler = RealESRGANer(
        scale=2,
        model_path='models/RealESRGAN_x2plus.pth',
        tile=400,
        tile_pad=10,
        pre_pad=0,
        half=False)
    face_enhancer = GFPGANer(
        model_path='models/GFPGANv1.3.pth',
        upscale=2,
        arch='clean',
        channel_multiplier=2,
        bg_upsampler=bg_upsampler)
    img_str = request.files['file'].read()
    request.files['file'].close()

    nparr = np.frombuffer(img_str, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    _, _, img = face_enhancer.enhance(img_np, has_aligned=False, only_center_face=False, paste_back=True)
    _, buffer = cv2.imencode('.jpg', img)
    jpg_as_text = base64.b64encode(buffer)
    return jpg_as_text

if __name__ == '__main__':
    app.run(host="0.0.0.0")