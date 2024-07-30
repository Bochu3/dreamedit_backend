from PIL import Image
from transparent_background import Remover
from ultralytics import YOLO
import numpy as np
from ultralytics.utils.ops import scale_image
from spandrel import ModelLoader
import torch
from utils import mask_floor, pilToBase64, resize_square, tensor_to_pil, to_torch, undo_resize_square
remover = Remover()
yolo = YOLO("yolov8x-seg.pt")
lama_file = "./models/big-lama.pt"
if lama_file.endswith(".pt"):
    sd = torch.jit.load(lama_file, map_location="cuda").state_dict()
lama = ModelLoader().load_from_state_dict(sd)

def rembg(img: Image) -> str:
  result = remover.process(img, type='map', threshold=0.7)
  result = result.convert('L')
  return pilToBase64(result)

def segment(img: Image):
    data = []
    results = yolo(img)
    result = results[0]
    boxes = result.boxes
    if(len(boxes) > 0):
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

def remove_object(image:Image, mask:Image):
    image = Image.open('/content/NWFSrLW9PsHSyQ3687uJ_0.png').convert("RGB")
    image = np.array(image).astype(np.float32) / 255.0
    image = torch.from_numpy(image)[None,]
    mask = Image.open('/content/ComfyUI_temp_hahxz_00001_.png').convert("L")
    mask = np.array(mask).astype(np.float32) / 255.0
    mask = torch.from_numpy(mask)[None,]
    if lama.architecture.id == "LaMa":
        required_size = 256
    image, mask = to_torch(image, mask)
    batch_size = 1
    if mask.shape[0] != batch_size:
        mask = mask[0].unsqueeze(0).repeat(batch_size, 1, 1, 1)

    image_device = image.device
    device = "cuda"
    lama.to(device)

    i=0
    work_image, work_mask = image[i].unsqueeze(0), mask[i].unsqueeze(0)
    work_image, work_mask, original_size = resize_square(
        work_image, work_mask, required_size
    )
    work_mask = mask_floor(work_mask)

    torch.manual_seed(0)
    work_image = lama(work_image.to(device), work_mask.to(device))


    work_image.to(image_device)
    work_image = undo_resize_square(work_image.to(image_device), original_size)
    work_image = image[i] + (work_image - image[i]) * mask_floor(mask[i])
    work_image = work_image.permute(0, 2, 3, 1)
    pil_image = tensor_to_pil(work_image)
    return pilToBase64(pil_image)