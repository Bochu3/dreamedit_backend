from typing import List
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Depends, FastAPI, File, Form, Request, UploadFile
from PIL import Image
from app import rembg, remove_object, segment
from utils import download_image
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
async def rembg(file: UploadFile):
    image = Image.open(file.file).convert("RGB")
    data = {
        "rembg": rembg(image),
    }
    return data

@app.post("/segment")
async def segment(file: UploadFile):
    image = Image.open(file.file).convert("RGB")
    data = {
        "segment": segment(image)
    }
    return data

@app.post("/remove_object")
async def update_style(files: List[UploadFile]):
    image = files[0]
    mask = files[1]
    image = Image.open(image.file).convert("RGB")
    mask = Image.open(mask.file).convert("L")
    data = {
        "remove_object": remove_object(image, mask)
    }
    return data

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8081)