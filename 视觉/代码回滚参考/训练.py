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
batchsize = 1024
num_workers = 2
device = torch.device('cuda:0')
image_size = (32, 32)
idx2name = {0: "1", 1: "2", 2: "3", 3: "4", 4: "5", 5: "base", 6: "outpost", 7: "sentry", 8: "wrongpic"}


class ImageDataset(Dataset):
    def __init__(self, mode="train"):
        self.root = "/content/BinaryImageOfArmor"
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((32, 32)),
            # transforms.Normalize((0.1307,), (0.3081,))
        ])
        clazs = os.listdir(self.root)
        self.picpaths = []
        self.images = []
        for claz in clazs:
            paths = os.listdir(os.path.join(self.root, claz))
            for path in paths:
                self.picpaths.append(os.path.join(self.root, claz, path))
        for path in self.picpaths:
            with open(path, "rb") as fp:
                tmp = Image.open(fp)
                self.images.append(np.array(tmp))

    def __getitem__(self, idx):
        picpath = self.picpaths[idx]
        img = self.transform(self.images[idx])
        label = picpath.split("/")[-2]
        if label == "wrongpic":
            label = 0
        elif 1 <= int(label) <= 8:
            label = 1

        return img, label

    def __len__(self):
        return len(self.picpaths)


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
    '''
        Designed by QPC, aimed at a binary (32*32) input, and 2 classes classification.
        This is extraodinarily small so that it's necessary to design a special net.
    '''

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


# %%
if __name__ == '__main__':
    dataset = ImageDataset()  # 实例
    dataloader = DataLoader(dataset=dataset, batch_size=batchsize, num_workers=num_workers, shuffle=True)
    for img, label in dataset:
        print(img.shape)  # [bs, channels, height, width]
        break
    dnn = QNet()
    dnn.to(device)
    # 定义损失函数
    loss_function = nn.CrossEntropyLoss()
    loss_function.to(device)
    # 定义优化器
    optimizer = torch.optim.Adam(dnn.parameters(), lr=learning_rate)
    log_train_acc = []
    # 进行迭代
    print("start training...")
    for i in range(50):
        # 训练
        dnn.train()
        total_train_acc = 0.0
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = dnn(imgs)
            print(outputs)
            loss = loss_function(outputs, labels)
            total_train_acc += ((outputs.argmax(1) == labels).sum()).item()
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("{}th epoch. train_acc:{:.4f}".format(i, total_train_acc / len(dataset)))
        log_train_acc.append(total_train_acc / len(dataset))
    print(len(dataset))
    torch.save(dnn.state_dict(), "./" + "Binary_classification" + ".pth")

    plt.plot(log_train_acc)
    plt.title('Accuracy')
    plt.show()

