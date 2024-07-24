from fastapi.middleware.cors import CORSMiddleware
from fastapi import Depends, FastAPI, File, Form, Request, UploadFile

from app import rembg, segment
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

@app.get("/preprocess")
async def update_style(url: str):
    image = download_image(url)
    data = {
        "rembg": rembg(image),
        "segment": segment(image)
    }
    json_data = json.dumps(data)
    return json_data


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)