from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from src import predict_in_tfl

import io

app = FastAPI()
origins = [
    "http://localhost:5500",
    "http://127.0.0.1:5500",
    "https://rginanjar.dev"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"Hello":"World"}

@app.post("/upload-image")
async def upload_images(file: UploadFile = File(...)):
    file_bytes = await file.read()

    image = Image.open(io.BytesIO(file_bytes))

    response_in_json = predict_in_tfl(image)

    return response_in_json