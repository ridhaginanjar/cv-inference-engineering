from fastapi import FastAPI, UploadFile, File
from PIL import Image

import io

app = FastAPI()

@app.get("/")
def root():
    return {"Hello":"World"}

@app.post("/upload-image")
async def upload_images(file: UploadFile = File(...)):
    file_bytes = await file.read()

    image = Image.open(io.BytesIO(file_bytes))

    return {
        "filename": file.filename,
        "format": image.format,
        "size": image.size  # (width, height)
    }