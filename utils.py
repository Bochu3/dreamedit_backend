from PIL import Image, ImageDraw, ImageOps, ImageFilter, ImageFile
import requests
def download_image(url, width=None, height=None)->Image:
    print(f"downloading image {url}")
    # image = load_image(
    #    url
    # )
    image = Image.open(requests.get(url, stream=True, timeout=20).raw,)
    image = image.convert("RGB")
    if(width is None or height is None):
        return image
    width = int(width)
    height = int(height)
    return image.resize((width,height), Image.Resampling.LANCZOS)