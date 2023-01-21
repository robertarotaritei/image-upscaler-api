from flask import Flask, request
from flask_cors import CORS
import replicate
import os
from dotenv import load_dotenv

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

load_dotenv()
app = Flask(__name__)
CORS(app)
os.environ['REPLICATE_API_TOKEN'] = os.getenv('REPLICATE_API_TOKEN')
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file present'
    model = replicate.models.get("xinntao/realesrgan")
    version = model.versions.get("1b976a4d456ed9e4d1a846597b7614e79eadad3032e9124fa63859db0fd59b56")
    f = request.files['file']
    f.save(f.filename)
    inputs = {
        'img': open(f.filename, 'rb'),
        'version': "General - RealESRGANplus",
        'scale': 2,
        'face_enhance': False,
        'tile': 16,
    }
    output = version.predict(**inputs)
    print(output)
    return output

if __name__ == '__main__':
    app.run(host="0.0.0.0")