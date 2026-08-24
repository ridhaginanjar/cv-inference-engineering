# Computer Vision Inference Backend

This repository is the backend part of my exploration into serving deep learning
models for computer vision inference.

The broader project is split into a few stages:

1. Explore the model and preprocessing flow in Google Colab.
2. Package the inference flow behind a FastAPI backend.
3. Build a frontend client that can upload images and display predictions.

This repository focuses on stage 2: exposing a trained computer vision model as
an HTTP API.

## Project Overview

The backend accepts an uploaded image, preprocesses it, runs inference with a
TensorFlow Lite model, and returns the top 5 ImageNet predictions.

Current model:

- Model: MobileNetV4 convolution medium
- Format: TensorFlow Lite (`.tflite`)
- Input task: image classification
- Labels: ImageNet classes
- API framework: FastAPI

## Repository Structure

```text
.
├── app/
│   ├── main.py              # FastAPI application and routes
│   └── src/
│       ├── __init__.py
│       └── preprocess.py    # Image preprocessing and TFLite inference
├── model/
│   ├── imagenet_classes.txt
│   └── mobilenetv4_conv_medium.e500_r256_in1k_float32.tflite
├── requirements.txt
└── README.md
```

## API Endpoints

### `GET /`

Simple root endpoint used to check whether the API is running.

Example response:

```json
{
  "Hello": "World"
}
```

### `POST /upload-image`

Uploads an image and returns the top 5 predicted ImageNet classes.

Request:

- Content type: `multipart/form-data`
- Field name: `file`
- Value: image file

Example response:

```json
[
  "tabby:42.15",
  "tiger cat:21.8",
  "Egyptian cat:10.44",
  "lynx:4.27",
  "Siamese cat:3.91"
]
```

Each item follows this format:

```text
class_name:probability_percentage
```

## How Inference Works

The inference flow is implemented in `app/src/preprocess.py`:

1. Read the uploaded image.
2. Resize the image to `256x256`.
3. Convert the image into a NumPy array.
4. Normalize the image using the pretrained model mean and standard deviation.
5. Load the TensorFlow Lite model.
6. Run inference with `tf.lite.Interpreter`.
7. Apply softmax to convert logits into probabilities.
8. Return the top 5 ImageNet class predictions.

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running The API

Run the FastAPI development server:

```bash
fastapi dev app/main.py
```

The API should be available at:

```text
http://127.0.0.1:8000
```

FastAPI also provides interactive API documentation at:

```text
http://127.0.0.1:8000/docs
```

## Testing With cURL

Use any local image file:

```bash
curl -X POST "http://127.0.0.1:8000/upload-image" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/image.jpg"
```