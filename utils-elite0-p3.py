#  Copyright (C) 2020 Xilinx, Inc
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import time
from pynq_dpu import DpuOverlay
import os
import numpy as np
import cv2
from tqdm import tqdm

BASE_DIR = "."
RESULT_DIR = BASE_DIR
RESULT_FILE = RESULT_DIR + '/image.list.result'
IMAGE_FOLDER = 'val'
# model = "dpu_tf_inceptionv1_0.elf"
# model = "dpu_resnet50_0.elf"
model = "dpu_elit0_p3.elf"

# user added initialization
overlay = DpuOverlay("dpu.bit")
overlay.set_runtime("vart")
overlay.load_model(model)

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94

MEANS = [_B_MEAN,_G_MEAN,_R_MEAN]

def resize_shortest_edge(image, size):
    H, W = image.shape[:2]
    if H >= W:
        nW = size
        nH = int(float(H)/W * size)
    else:
        nH = size
        nW = int(float(W)/H * size)
    return cv2.resize(image,(nW,nH))

def mean_image_subtraction(image, means):
    B, G, R = cv2.split(image)
    B = B - means[0]
    G = G - means[1]
    R = R - means[2]
    image = cv2.merge([R, G, B])
    return image

def BGR2RGB(image):
    B, G, R = cv2.split(image)
    image = cv2.merge([R, G, B])
    return image

def central_crop(image, crop_height, crop_width):
    image_height = image.shape[0]
    image_width = image.shape[1]
    offset_height = (image_height - crop_height) // 2
    offset_width = (image_width - crop_width) // 2
    return image[offset_height:offset_height + crop_height, offset_width:
                 offset_width + crop_width, :]

def normalize(image):
    image = image - [_R_MEAN, _G_MEAN, _B_MEAN]
    image = image / [58.395, 57.12 , 57.375]
    return image

def preprocess_fn(image, crop_height = 224, crop_width = 224):
    image = cv2.imread(image)
    image = BGR2RGB(image)
    image = resize_shortest_edge(image, 256)
    image = central_crop(image, crop_height, crop_width)
    image = normalize(image)
    return image

def predict_label(data):
    # return np.argmax(data)-1
    return np.argmax(data)

def top5_label(data):
    return data.argsort()[-5:][::-1]-1

class Processor:
    def __init__(self):
        pass

    def run(self):
        # The val folder containes all the val images
        original_images = [i for i in os.listdir(IMAGE_FOLDER) if i.endswith("JPEG")]
        total_images = len(original_images)

        #user VART to do image classification
        dpu = overlay.runner
        inputTensors = dpu.get_input_tensors()
        outputTensors = dpu.get_output_tensors()
        tensorformat = dpu.get_tensor_format()
        if tensorformat == dpu.TensorFormat.NCHW:
            outputHeight = outputTensors[0].dims[2]
            outputWidth = outputTensors[0].dims[3]
            outputChannel = outputTensors[0].dims[1]
        elif tensorformat == dpu.TensorFormat.NHWC:
            outputHeight = outputTensors[0].dims[1]
            outputWidth = outputTensors[0].dims[2]
            outputChannel = outputTensors[0].dims[3]
        else:
            raise ValueError("Image format error.")
        outputSize = outputHeight*outputWidth*outputChannel

        # define buffers to store input and output
        # these buffers will be reused
        shape_in = (1,) + tuple(
            [inputTensors[0].dims[i] for i in range(inputTensors[0].ndims)][1:])
        shape_out = (1, outputHeight, outputWidth, outputChannel)
        input_data = []
        output_data = []
        input_data.append(np.empty((shape_in),
                                    dtype = np.float32, order = 'C'))
        output_data.append(np.empty((shape_out),
                                    dtype = np.float32, order = 'C'))
        image = input_data[0]

        # result output
        fwa = open(RESULT_FILE, "w")

        for image_index in tqdm(range(total_images)):
            preprocessed = preprocess_fn(os.path.join(IMAGE_FOLDER, original_images[image_index]))
            image[0,...] = preprocessed.reshape(
                inputTensors[0].dims[1],
                inputTensors[0].dims[2],
                inputTensors[0].dims[3])
            job_id = dpu.execute_async(input_data, output_data)
            dpu.wait(job_id)
            temp = [j.reshape(1, outputSize) for j in output_data]
            #y_hat = predict_label(temp[0][0])
            y_hat_top_5=top5_label(temp[0][0])
            # write output result
            for y_hat in y_hat_top_5:
                fwa.writelines(f'{original_images[image_index]} {y_hat}\n')

        fwa.close()
        return
