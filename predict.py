import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize
import ConvNet
from constants import *

def load_model(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(model_path)  # 加载模型
    model = model.to(device)
    return model

def predict_digital_image(filename):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load('./MNIST.pth')  # 加载模型
    model = model.to(device)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()  # 把模型转为test模式
    img = cv2.imread(filename, 0)  # 以灰度图的方式读取要预测的图片
    img = cv2.resize(img, (28, 28 - 2 * 1))
    # img = 255 - img
    dst = np.zeros((1, 28), np.uint8)
    img = np.insert(img, 0, dst, 0)
    img = np.insert(img, 28 - 1, dst, 0)
    # img = cv2.resize(img, (28, 28))
    img = np.array(img).astype(np.float32)
    # cv2.imshow("i", img)
    img = np.expand_dims(img, 0)
    img = np.expand_dims(img, 0)  # 扩展后，为[1，1，28，28]
    img = torch.from_numpy(img)
    trans = Normalize((0.1307,), (0.3081,))
    img = trans(img)
    img = img.to(device)
    output = model(Variable(img))
    prob = F.softmax(output, dim=1)
    prob = Variable(prob)
    prob = prob.cpu().numpy()  # 用GPU的数据训练的模型保存的参数都是gpu形式的，要显示则先要转回cpu，再转回numpy模式
    # print(prob)  # prob是10个分类的概率
    pred = np.argmax(prob)  # 选出概率最大的一个
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return pred.item()

def predict_box_image(dataset, model):
    for i in range(len(dataset)):
        img = cv2.resize(dataset[i], (28, 28))
        dataset[i] = np.array(img).astype(np.float32)
    dataset = np.array(dataset)
    dataset = np.expand_dims(dataset, 1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # transform = transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.1307,), (0.3081,))])
    # dataset = transform(dataset)
    test_loader = DataLoader(
        dataset, batch_size=512, shuffle=False,)
    if model == None:
        print("require a pre-trained model")
        return -1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    pred = []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            output = model(data)
            pred = output.max(1, keepdim=True)[1]  # 找到概率最大的下标
    ret = None
    try:
        ret = pred.cpu().numpy()
    except:
        ret = pred.numpy()
    return ret

def predict_images(dataset, model):
    for i in range(len(dataset)):
        # if sum(dataset[i][0]) == 0:
        img = dataset[i]
        w = img.shape[1] * 24 // img.shape[0]
        if w % 2: w += 1
        a = (24 - w) // 2 + 2
        if w > 24:
            w = 24
            a = 2
        try:
            img = cv2.resize(img, (w, 24))
            # img = 255 - img
            dst = np.zeros((24, a), np.uint8)
            dst1 = np.zeros((2, 28), np.uint8)
            img = np.insert(img, 0, dst.T, 1)
            img = np.insert(img, 28 - a, dst.T, 1)
            img = np.insert(img, 0, dst1, 0)
            img = np.insert(img, 26, dst1, 0)
        except:
            cv2.imshow(str(i), img)
        # else:
        #     img = cv2.resize(dataset[i], (28 - 2 * a, 28 - 2 * b))
        #     dst = np.zeros((b, 28 - 2 * a), np.uint8)
        #     dst1 = np.zeros((28, a), np.uint8)
        #     img = np.insert(img, 0, dst, 0)
        #     img = np.insert(img, 28 - b, dst, 0)
        #     img = np.insert(img, 0, dst1.T, 1)
        #     img = np.insert(img, 28 - a, dst1.T, 1)
        # cv2.imshow(str(i), img)
        dataset[i] = np.array(img).astype(np.float32)
    dataset = np.array(dataset)
    dataset = np.expand_dims(dataset, 1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # transform = transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.1307,), (0.3081,))])
    # dataset = transform(dataset)
    test_loader = DataLoader(
        dataset, batch_size=512, shuffle=False,)
    if model == None:
        print("require a pre-trained model")
        return -1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    pred = []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            output = model(data)
            pred = output.max(1, keepdim=True)[1]  # 找到概率最大的下标
    ret = None
    try:
        ret = pred.cpu().numpy()
    except:
        ret = pred.numpy()
    return ret


if __name__ == '__main__':
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = torch.load('./MNIST.pth')  # 加载模型
    # model = model.to(device)
    # model.eval()  # 把模型转为test模式
    #
    # img = cv2.imread('3.png', 0)  # 以灰度图的方式读取要预测的图片
    # img = cv2.resize(img, (28, 28))
    #
    # height, width = img.shape
    # dst = np.zeros((height, width), np.uint8)
    # for i in range(height):
    #     for j in range(width):
    #         dst[i, j] = 255 - img[i, j]
    #
    # img = dst
    #
    # img = np.array(img).astype(np.float32)
    # img = np.expand_dims(img, 0)
    # img = np.expand_dims(img, 0)  # 扩展后，为[1，1，28，28]
    # img = torch.from_numpy(img)
    # img = img.to(device)
    # output = model(Variable(img))
    # prob = F.softmax(output, dim=1)
    # prob = Variable(prob)
    # prob = prob.cpu().numpy()  # 用GPU的数据训练的模型保存的参数都是gpu形式的，要显示则先要转回cpu，再转回numpy模式
    # print(prob)  # prob是10个分类的概率
    # pred = np.argmax(prob)  # 选出概率最大的一个
    # print(pred.item())
    model = load_model()
    print(predict_digital_image('gou.png', model))
