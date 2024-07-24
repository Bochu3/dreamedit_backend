from PIL import Image
from transparent_background import Remover
from ultralytics import YOLO
import io
import base64
import numpy as np
from ultralytics.utils.ops import scale_image
remover = Remover()
yolo = YOLO("yolov8x-seg.pt")
def pilToBase64(img) -> str:
  image_bytes = io.BytesIO()
  img.save(image_bytes, format="webp")
  image_bytes = image_bytes.getvalue()
  base64_string = base64.b64encode(image_bytes).decode("utf-8")
  return base64_string

def rembg(img: Image) -> str:
  result = remover.process(img, type='map', threshold=0.7)
  result = result.convert('L')
  return pilToBase64(result)

def segment(img: Image):
    data = []
    results = yolo(img)
    result = results[0]
    boxes = result.boxes
    masks = result.masks.data.cpu().numpy()
    masks = np.moveaxis(masks, 0, -1)
    masks = scale_image(masks, result.masks.orig_shape)
    masks = np.moveaxis(masks, -1, 0)
    for mask, box in zip(masks, boxes):
        class_id = int(box.cls[0])
        label = yolo.names[class_id]
        x1, y1, x2, y2 = box.xyxy[0]
        colored_mask = (mask*255).astype(np.uint8)
        colored_mask = Image.fromarray(colored_mask, 'L')
        cropped_mask = colored_mask.crop((int(x1), int(y1), int(x2), int(y2)))
        base64_string = pilToBase64(cropped_mask)
        data.append({
            "confident": float(box.conf),
            "box":[
                int(x1),
                int(y1),
                int(x2),
                int(y2)
            ],
            "mask": base64_string,
            "label": label
        })
    return data