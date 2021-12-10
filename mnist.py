import random

import cv2
import torch
import torch.nn.functional as F
import torch.optim as optim
import os
import numpy as np
from torch.nn import functional as F

from ConvNet import ConvNet

torch.__version__

BATCH_SIZE = 512  # 大概需要2G的显存
EPOCHS = 5  # 总共训练批次
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 让torch判断是否使用GPU，建议使用GPU环境，因为会快很多

# 下载训练集
# train_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('datasets', train=True, download=True,
#                    transform=transforms.Compose([
#                        transforms.ToTensor(),
#                        transforms.Normalize((0.1307,), (0.3081,))])
#                    ),
#     batch_size=BATCH_SIZE, shuffle=True)


# a = datasets.MNIST('datasets', train=True).data
# b = a[0].numpy()

# 下载测试集
# test_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('datasets', train=False, transform=transforms.Compose([
#         transforms.ToTensor(),
#         # transforms.Normalize((0.1307,), (0.3081,))
#     ])),
#     batch_size=BATCH_SIZE, shuffle=True)

data = []
targets = []
for root, dir, files in os.walk("datasets/cross&tick"):
    for file in files:
        img = cv2.imread(root + "/" + file, 0)  # 以灰度图的方式读取要预测的图片
        ret, img = cv2.threshold(img, 100, 255, 1)
        # kernel = np.ones((5, 5), np.uint8)
        # img = cv2.dilate(img, kernel, 1)
        img = cv2.resize(img, (28, 28))
        if root == "datasets/cross&tick\\cross" or root == "datasets/cross&tick\\crossaug":
        # if root == "datasets/cross&tick\\crossaug":
            data.append(img)
            targets.append(1)
        else:
        # elif root == "datasets/cross&tick\\tickaug":
            data.append(img)
            targets.append(0)
randnum = random.randint(0, 100)
random.seed(randnum)
random.shuffle(data)
random.seed(randnum)
random.shuffle(targets)

split_index = 8000
data = data[split_index:] + data[:split_index]
targets = targets[split_index:] + targets[:split_index]
print(len(data))
data_torch = torch.from_numpy(np.array(data).reshape((len(targets), 1, 28, 28)))
datasets = torch.utils.data.TensorDataset(data_torch.to(torch.float32), torch.from_numpy(np.array(targets)).to(torch.long) )
train_set, test_set = torch.utils.data.random_split(datasets, [split_index,len(datasets) - split_index])
train_loader = torch.utils.data.DataLoader(train_set,
    batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set,
    batch_size=BATCH_SIZE, shuffle=True)


# 定义卷积神经网络


# 训练


# 测试
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # 将一批的损失相加
            pred = output.max(1, keepdim=True)[1]  # 找到概率最大的下标
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if (batch_idx + 1) % 30 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


if __name__ == '__main__':
    model = ConvNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters())

    for epoch in range(1, EPOCHS + 1):
        train(model, DEVICE, train_loader, optimizer, epoch)
        test(model, DEVICE, test_loader)

    # 保存训练完成后的模型
    torch.save(model, './CHECK_with_box.pth')
