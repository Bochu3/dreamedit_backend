from PIL import Image, ImageDraw, ImageOps, ImageFilter, ImageFile
import requests
from torch import Tensor
import torch.nn.functional as F
import numpy as np
import os
import io
import base64
def download_model(file_url:str):
    # Define the local file path where the file will be saved
    file_name = file_url.split("/")[-1]
    local_file_path = f'./{file_name}'
    # Check if the file already exists in the local directory
    if not os.path.exists(local_file_path):
        # Download the file
        response = requests.get(file_url)
        # Save the file to the local directory
        with open(local_file_path, "wb") as file:
            file.write(response.content)
    return local_file_path
def download_image(url, width=None, height=None)->Image:
    print(f"downloading image {url}")
    image = Image.open(requests.get(url, stream=True, timeout=20).raw,)
    image = image.convert("RGB")
    if(width is None or height is None):
        return image
    width = int(width)
    height = int(height)
    return image.resize((width,height), Image.Resampling.LANCZOS)
def pilToBase64(img) -> str:
  image_bytes = io.BytesIO()
  img.save(image_bytes, format='WebP', optimize=True, quality=90, lossless=False)
  image_bytes = image_bytes.getvalue()
  base64_string = base64.b64encode(image_bytes).decode("utf-8")
  return base64_string
def mask_unsqueeze(mask: Tensor):
    if len(mask.shape) == 3:  # BHW -> B1HW
        mask = mask.unsqueeze(1)
    elif len(mask.shape) == 2:  # HW -> B1HW
        mask = mask.unsqueeze(0).unsqueeze(0)
    return mask

def mask_floor(mask: Tensor, threshold: float = 0.99):
    return (mask >= threshold).to(mask.dtype)

def to_torch(image: Tensor, mask: Tensor | None = None):
    if len(image.shape) == 3:
        image = image.unsqueeze(0)
    image = image.permute(0, 3, 1, 2)  # BHWC -> BCHW
    if mask is not None:
        mask = mask_unsqueeze(mask)
    if image.shape[2:] != mask.shape[2:]:
        raise ValueError(
            f"Image and mask must be the same size. {image.shape[2:]} != {mask.shape[2:]}"
        )
    return image, mask

def resize_square(image: Tensor, mask: Tensor, size: int):
    _, _, h, w = image.shape
    pad_w, pad_h, prev_size = 0, 0, w
    if w == size and h == size:
        return image, mask, (pad_w, pad_h, prev_size)

    if w < h:
        pad_w = h - w
        prev_size = h
    elif h < w:
        pad_h = w - h
        prev_size = w
    image = pad_reflect_once(image, (0, pad_w, 0, pad_h))
    mask = pad_reflect_once(mask, (0, pad_w, 0, pad_h))

    if image.shape[-1] != size:
        image = F.interpolate(image, size=size, mode="nearest-exact")
        mask = F.interpolate(mask, size=size, mode="nearest-exact")

    return image, mask, (pad_w, pad_h, prev_size)


def undo_resize_square(image: Tensor, original_size: tuple[int, int, int]):
    _, _, h, w = image.shape
    pad_w, pad_h, prev_size = original_size
    if prev_size != w or prev_size != h:
        image = F.interpolate(image, size=prev_size, mode="bilinear")
    return image[:, :, 0 : prev_size - pad_h, 0 : prev_size - pad_w]

# torch pad does not support padding greater than image size with "reflect" mode
def pad_reflect_once(x: Tensor, original_padding: tuple[int, int, int, int]):
    _, _, h, w = x.shape
    padding = np.array(original_padding)
    size = np.array([w, w, h, h])

    initial_padding = np.minimum(padding, size - 1)
    additional_padding = padding - initial_padding

    x = F.pad(x, tuple(initial_padding), mode="reflect")
    if np.any(additional_padding > 0):
        x = F.pad(x, tuple(additional_padding), mode="constant")
    return x
def tensor_to_pil(img_tensor, batch_index=0):
  # Convert tensor of shape [batch_size, channels, height, width] at the batch_index to PIL Image
  img_tensor = img_tensor[batch_index].unsqueeze(0)
  i = 255. * img_tensor.detach().numpy()
  img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8).squeeze())
  return img