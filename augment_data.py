import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import random
import os
import numpy as np

BATCH_SIZE = 512  # 大概需要2G的显存
EPOCHS = 20  # 总共训练批次
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 让torch判断是否使用GPU，建议使用GPU环境，因为会快很多

data = []
targets = []
for root, dir, files in os.walk("datasets/cross&tick/tick"):
    for file in files:
        img = cv2.imread(root + "/" + file, 0)  # 以灰度图的方式读取要预测的图片
        # ret, img = cv2.threshold(img, 100, 255, 1)
        img = img[random.randint(1,10):img.shape[0] - 1 -random.randint(1,10), random.randint(1,10):img.shape[1] - 1 -random.randint(1,10)]
        img = cv2.rectangle(img, (int(img.shape[0] * random.random() * 0.1), int(img.shape[1] * random.random() * 0.1)), (int(img.shape[0] * (random.random() * 0.1 + 0.9)), int(img.shape[1] * (random.random() * 0.1 + 0.9))), (0, 0, 0), 1)
        # img = cv2.rectangle(img, (2, 2), (int(img.shape[0] - 3), int(img.shape[1] - 3)), (0, 0, 0), 1)
        # cv2.imshow(root + "aug/1" + file, img)
        cv2.imwrite(root + "aug/15" + file, img)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        # kernel = np.ones((5, 5), np.uint8)
        # img = cv2.dilate(img, kernel, 1)
        # img = cv2.resize(img, (28, 28))
        # if root == "datasets/cross&tick\\cross":
        #     data.append(img)
        #     targets.append(1)
        # else:
        #     data.append(img)
        #     targets.append(0)

# data = data[3000:] + data[:3000]
# targets = targets[3000:] + targets[:3000]
# print(len(data))
# data_torch = torch.from_numpy(np.array(data).reshape((len(targets), 1, 28, 28)))
# datasets = torch.utils.data.TensorDataset(data_torch.to(torch.float32), torch.from_numpy(np.array(targets)).to(torch.long) )
# train_set, test_set = torch.utils.data.random_split(datasets, [3000,1344])
# train_loader = torch.utils.data.DataLoader(train_set,
#     batch_size=BATCH_SIZE, shuffle=True)
# test_loader = torch.utils.data.DataLoader(test_set,
#     batch_size=BATCH_SIZE, shuffle=True)


# 定义卷积神经网络
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        # batch*1*28*28（每次会送入batch个样本，输入通道数1（黑白图像），图像分辨率是28x28）
        # 下面的卷积层Conv2d的第一个参数指输入通道数，第二个参数指输出通道数，第三个参数指卷积核的大小
        self.conv1 = nn.Conv2d(1, 10, 5)  # 输入通道数1，输出通道数10，核的大小5
        self.conv2 = nn.Conv2d(10, 20, 3)  # 输入通道数10，输出通道数20，核的大小3
        # 下面的全连接层Linear的第一个参数指输入通道数，第二个参数指输出通道数
        self.fc1 = nn.Linear(20 * 10 * 10, 500)  # 输入通道数是2000，输出通道数是500
        self.fc2 = nn.Linear(500, 10)  # 输入通道数是500，输出通道数是10，即10分类

    def forward(self, x):
        in_size = x.size(0)  # 在本例中in_size=512，也就是BATCH_SIZE的值。输入的x可以看成是512*1*28*28的张量。
        out = self.conv1(x)  # batch*1*28*28 -> batch*10*24*24（28x28的图像经过一次核为5x5的卷积，输出变为24x24）
        out = F.relu(out)  # batch*10*24*24（激活函数ReLU不改变形状））
        out = F.max_pool2d(out, 2, 2)  # batch*10*24*24 -> batch*10*12*12（2*2的池化层会减半）
        out = self.conv2(out)  # batch*10*12*12 -> batch*20*10*10（再卷积一次，核的大小是3）
        out = F.relu(out)  # batch*20*10*10
        out = out.view(in_size, -1)  # batch*20*10*10 -> batch*2000（out的第二维是-1，说明是自动推算，本例中第二维是20*10*10）
        out = self.fc1(out)  # batch*2000 -> batch*500
        out = F.relu(out)  # batch*500
        out = self.fc2(out)  # batch*500 -> batch*10
        out = F.log_softmax(out, dim=1)  # 计算log(softmax(x))
        return out


# 训练
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


if __name__ == '__main__':
    model = ConvNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters())

    # for epoch in range(1, EPOCHS + 1):
    #     train(model, DEVICE, train_loader, optimizer, epoch)
    #     test(model, DEVICE, test_loader)
    #
    # # 保存训练完成后的模型
    # torch.save(model, './CHECK.pth')
