from fastapi import FastAPI, UploadFile, File
from PIL import Image
from src import predict_in_tfl

import io

app = FastAPI()

@app.get("/")
def root():
    return {"Hello":"World"}

@app.post("/upload-image")
async def upload_images(file: UploadFile = File(...)):
    file_bytes = await file.read()

    image = Image.open(io.BytesIO(file_bytes))

    response_in_json = predict_in_tfl(image)

    return response_in_json