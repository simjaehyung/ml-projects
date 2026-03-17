import cv2
import numpy as np
import requests
import json
import os
import shutil
import time

# Generate a proper valid test image using imencode
img = np.zeros((400,600,3), dtype=np.uint8)
cv2.rectangle(img, (100,100), (200,200), (0,255,0), -1)

# Directory settings
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
upload_dir = os.path.join(base_dir, 'data', 'uploads')
os.makedirs(upload_dir, exist_ok=True)
filepath = os.path.join(upload_dir, 'test.png')

is_success, im_buf_arr = cv2.imencode('.png', img)
if is_success: im_buf_arr.tofile(filepath)

url = 'http://127.0.0.1:5000/save'
payload = {
    "filename": "test.png",
    "annotations": [
        {"label": "Green Box", "x": 100, "y": 100, "w": 100, "h": 100}
    ],
    "augmentation": {
        "rotation": True,
        "brightness": True,
        "noise": True
    }
}

headers = {'Content-Type': 'application/json'}
response = requests.post(url, data=json.dumps(payload), headers=headers)

print("Status Code:", response.status_code)
print("Response JSON:", response.json())

# Check output directory
output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'output')
print("Files in output dir:", os.listdir(output_dir))
