import torch
import torchvision
from PIL import Image
from torchvision import transforms
import torchvision.models as models
import matplotlib.pyplot as plt
import time
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import pdb
import os
import numpy as np
import cv2

# This logger is required to build an engine
TRT_LOGGER = trt.Logger()
filename = "/home/rcclub/MVSDK/demo/python_demo/2.jpg"
# filename = "D:/biancheng/opencv python/reference/bigBinaryImageOfArmor/1/0.png"
# engine_file_path = "D:/biancheng/opencv python/reference/Binary_classification.engine"
engine_file_path = "/home/rcclub/MVSDK/demo/python_demo/8Binary_classification.engine"

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        """Within this context, host_mom means the cpu memory and device means the GPU memory
        """
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))

    return inputs, outputs, bindings, stream


def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer data from CPU to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]

    # Run inference.
    t_model = time.perf_counter()
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    print(f'only one line cost:{time.perf_counter() - t_model:.8f}s')

    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]

    # Synchronize the stream
    stream.synchronize()

    # Return only the host outputs.
    return [out.host for out in outputs]


print("Reading engine from file {}".format(engine_file_path))
with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())

# create the context for this engine
context = engine.create_execution_context()

# allocate buffers for input and output
inputs, outputs, bindings, stream = allocate_buffers(engine)  # input, output: host # bindings

image_size = (32, 32)
transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(image_size)])

# 读图
img = Image.open(filename)
print("img.size", img.size)

img_p = transform(img)
print("img_p.shape", img_p.shape)

# 增加一个维度
img_normalize = np.expand_dims(img_p, 0)
print("img_normalize.shape", img_normalize.shape)

# covert to numpy
# img_normalize_np = img_normalize.cpu().data.numpy()
img_normalize_np = img_normalize
# Load data to the buffer
inputs[0].host = img_normalize_np
print("inputs[0].host.shape", inputs[0].host.shape)

# Do Inference
t_model = time.perf_counter()
trt_outputs = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)  # numpy data
print(f'do inference cost:{time.perf_counter() - t_model:.8f}s')
print("trt_outputs", trt_outputs)
print("len(trt_outputs)", len(trt_outputs))

# 转化为torch tensor
out = torch.from_numpy(trt_outputs[0])
print("out.shape", out.shape)

# 拓展一个纬度
out = torch.unsqueeze(out, 0)
print("out.shape", out.shape)

with open('imagenet_classes.txt') as f:
    classes = [line.strip() for line in f.readlines()]

_, indices = torch.sort(out, descending=True)
print("indices", indices)
percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
prediction = [[classes[idx], percentage[idx].item()] for idx in indices[0][:1]]
print("prediction", prediction)

