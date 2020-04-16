#from torchvision import models
from torch import nn
import torch
from torch.nn import init

#自定义的模型
#自定义模型
# class FirstCNN(nn.Module):
#     def __init__(self, num_classes=101):
#         super(FirstCNN, self).__init__()
#         # 保证输出的大小和原图像大小一致为448
#         self.features = nn.Sequential(
#             nn.Conv2d(3, 64, 5, stride=2),  # 卷积核大小：5×5，stride=2  padding=226       #222
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64, 64, 3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),  # 111
#             nn.Conv2d(64, 128, 3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(128, 128, 3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),  # 55
#             nn.Conv2d(128, 128, 3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(128, 256, 3, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 256, 3, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),  # 27
#             nn.Conv2d(256, 256, 3, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 256, 3, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 256, 3, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),  # 13
#             nn.Conv2d(256, 512, 3, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(512, 512, 3, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(512, 512, 3, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),  # 6
#         )
#         # 自适应平均池化
#         self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
#         self.classifier = nn.Sequential(
#             nn.Dropout(p=0.25),
#             nn.Linear(512 * 6 * 6, 4096),
#             nn.ReLU(True),
#             nn.Dropout(p=0.25),
#             nn.Linear(4096, 1024),
#             nn.ReLU(True),
#             nn.Dropout(p=0.25),
#             nn.Linear(1024, num_classes)
#         )
#
#     def forward(self, x):
#         out = self.features(x)
#         # out=self.avgpool(out)
#         out = out.view(x.size(0), -1)
#         out = self.classifier(out)
#         return out
#
#
# # xavier初始化
# def init_weights(m):
#     if type(m) == nn.Linear or type(m) == nn.Conv2d:
#         torch.nn.init.xavier_uniform(m.weight)
#         m.bias.data.fill_(0.01)


# model.apply(init_weights)
# test
# model=FirstCNN(101)            #model size:45.56509M
# a=torch.randn((2,3,448,448))
# model(a)
# print(model)
# print('model size: {:.5f}M'.format(sum(p.numel() for p in model.parameters()) / 1e6))


#加载模型



# def model_process(vgg16_bn):
#     # 加载模型权重
#     state_dict = torch.load(r"F:\AI\classifierModelpth\vgg16_bn.pth\vgg16_bn.pth")
#     # 将权重赋给我们的模型
#     vgg16_bn.load_state_dict(state_dict)
#     # print(vgg16_bn.classifier[6].out_features)      #1000
#     # 修改全连接层最后一层的out_features值
#     num_in_features = vgg16_bn.classifier[6].in_features  # 4096
#     vgg16_bn.classifier[6] = nn.Linear(in_features=num_in_features, out_features=101)
#     # 修改第一层卷积层
#     first_conv = nn.Conv2d(3, 64, 5, 2, padding=1)
#     vgg16_bn.features[0] = first_conv
#
#     # 冻结所有未改变层的权重
#     for param in vgg16_bn.features[3:].parameters():
#         param.require_grad = False
#
#     # 第一层卷积层、全连阶层采用xavier初始化
#     def init_weights(m):
#         if type(m) == nn.Linear or m == vgg16_bn.features[0]:
#             torch.nn.init.xavier_uniform_(m.weight)
#             # m.bias.data.fill_(0.01)  预训练模型中卷积层没有，全连阶层有
#
#     vgg16_bn.apply(init_weights)
#     # init.xavier_uniform_(vgg16_bn.features[0].weight)
#     # init.constant_(vgg16_bn.features[0].bias, 0.1)
#     return vgg16_bn
#
# vgg16_bn=models.vgg16_bn()
# vgg16_bn=model_process(vgg16_bn)
# print('model size: {:.5f}M'.format(sum(p.numel() for p in vgg16_bn.parameters()) / 1e6))
# print(vgg16_bn)


#input = torch.randn(2, 3)
from torch.nn import functional as F
import os

from PIL import Image
from torchvision import transforms as T


if __name__=='__main__':
   print('cen'+str(2.22))