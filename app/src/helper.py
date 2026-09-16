import numpy as np
import tensorflow as tf
from PIL import Image

MODEL_PATH ="/home/dicoding/learning/fastapi/model/mobilenetv4_conv_medium.e500_r256_in1k_float32.tflite"
IMAGENET_CLASS_PATH = "/home/dicoding/learning/fastapi/model/imagenet_classes.txt"

def read_imagenet_classes():
    with open(IMAGENET_CLASS_PATH, 'r') as f:
        labels = f.readlines()

    return labels

def image_preprocess(image):
    img_dimension = (256,256)

    image = image.convert("RGB")
    image = image.resize(img_dimension, Image.BICUBIC)

    image = np.array(image, dtype=np.float32)

    # Reposition
    image = np.expand_dims(image, axis=-1)
    image = np.rollaxis(image, 3)

    # Standarization
    mean_pretrained_cfg = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std_pretrained_cfg = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image_stn = np.array((image / 255.0) - mean_pretrained_cfg) / std_pretrained_cfg

    return image_stn

def predict_in_tfl(image):
    # Init
    image_stn = image_preprocess(image)

    # Interpreter and IO Initializations
    tfl_interpret = tf.lite.Interpreter(model_path=MODEL_PATH)
    tfl_interpret.allocate_tensors()

    # Get the model contract
    inpt_cntrct = tfl_interpret.get_input_details()[0]
    outpt_cntrct = tfl_interpret.get_output_details()[0]

    # set to tensor
    tfl_interpret.set_tensor(inpt_cntrct['index'], image_stn)
    tfl_interpret.invoke() # returning numpy

    # get tensor
    tfl_output = tfl_interpret.get_tensor(outpt_cntrct['index'])
    tfl_tensor = tf.convert_to_tensor(tfl_output)

    # standarization -> the tensor has variety range of tensor (Check on Colab)
    tfl_tensor_softmax = tf.nn.softmax(tfl_tensor, axis=1)

    # get the top k=5 class
    top_5_probabilities, top_5_indices = tf.math.top_k(tfl_tensor_softmax, k=5)
    top_5_probabilities_list = top_5_probabilities.numpy().tolist()
    top_5_indices_list = top_5_indices.numpy().tolist()

    labels = read_imagenet_classes()
    index_to_labels = {index: cls_.strip() for index, cls_ in enumerate(labels)}

    response_in_json = [f"{index_to_labels[idx]}:{round(probs*100, 2)}" 
                        for idx, probs in 
                        zip(top_5_indices_list[0], top_5_probabilities_list[0])]

    return response_in_json



    


