import base64
import io
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Body, Depends, FastAPI, File, Form, Request, UploadFile
from PIL import Image
from app import rembg, remove_object, segment
# from utils import download_image
import json

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/hello")
def get_hello():
    return {"message": "Hello, World!"}


@app.post("/rembg")
async def rembg_function(image:str = Body(..., embed=True)):
    image_bytes = base64.b64decode(image)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    data = {
        "rembg": rembg(image),
    }
    return data

@app.post("/segment")
async def segment_function(image:str = Body(..., embed=True)):
    image_bytes = base64.b64decode(image)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    data = {
        "segment": segment(image)
    }
    return data

@app.post("/remove_object")
async def remove_object_function(image: str = Body(..., embed=True), mask: str = Body(..., embed=True)):
    image_bytes = base64.b64decode(image)
    mask_bytes = base64.b64decode(mask)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    mask = Image.open(io.BytesIO(mask_bytes))
    image.save("image.jpg")
    mask.save("mask.jpg")
    data = {
        "remove_object": remove_object(image, mask)
    }
    return data

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8081)