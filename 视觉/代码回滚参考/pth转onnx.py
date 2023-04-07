import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import torch
import numpy as np
import torch.nn as nn

learning_rate = 1e-3
epochs = 100
batchsize = 32
num_workers = 2
device = torch.device('cuda:0')
image_size = (32, 32)
idx2name = {0: "1", 1: "2", 2: "3", 3: "4", 4: "5", 5: "base", 6: "outpost", 7: "sentry", 8: "wrongpic"}


class QBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels * expansion, kernel_size=3, stride=1,
                      padding=1, bias=False),
            nn.BatchNorm2d(out_channels * expansion),
            nn.Conv2d(in_channels=out_channels * expansion, out_channels=out_channels, kernel_size=3, stride=1,
                      padding=1, bias=False),
            nn.ReLU6()
        )

    def forward(self, x):
        return self.conv(x.clone()) + x


class QNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.inblock = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.block1 = QBasicBlock(in_channels=16, out_channels=16, expansion=4)
        self.down_sample1 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.block2 = QBasicBlock(in_channels=32, out_channels=32, expansion=4)
        self.down_sample2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.down_sample3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=4, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.down_sample4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.outblock = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.inblock(x)
        x = self.block1(x)
        x = self.down_sample1(x)
        x = self.block2(x)
        x = self.down_sample2(x)
        x = self.down_sample3(x)
        x = self.down_sample4(x)
        x = x.view(-1, 128)
        x = self.outblock(x)
        return x


if __name__ == '__main__':
    dnn = QNet()
    dnn.to(device)
    dnn.eval()
    input_shape = (1, 32, 32)
    batch_size = 1
    x = torch.randn(batch_size, *input_shape).to(device)  # 生成张量
    # 加载模型
    dnn.load_state_dict(torch.load("/content/Binary_classification.pth", map_location=device))
    # 模型转换
    input_name = "inputs"
    output_name = "outputs"
    export_onnx_file = "./Binary_classification.onnx"  # 目的ONNX文件名
    torch.onnx.export(dnn,
                      x,
                      export_onnx_file,  # 设置onnx模型输出路径, 例如：c: / xxx.onnx
                      opset_version=10,
                      do_constant_folding=True,  # 是否执行常量折叠优化
                      export_params=True,
                      verbose=True,
                      input_names=[input_name],
                      output_names=[output_name],
                      dynamic_axes={
                          input_name: {0: 'batch_size'},
                          output_name: {0: 'batch_size'}})

    print("-----pth to onnx trans successed.")

