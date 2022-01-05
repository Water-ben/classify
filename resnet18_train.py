import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models
# from effnetv2 import effnetv2_s
from torch.autograd import Variable
from icecream import ic
# 设置超参数

BATCH_SIZE = 128
EPOCHS = 80
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据预处理

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    # transforms.RandomVerticalFlip(),
    # transforms.RandomCrop(50),
    # transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

])
transform_test = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

#本地训练数据路径： /Volumes/software/project/cv_dataset/classify/dogs-vs-cats/dataset_new/train
# 读取数据
# dataset_train = datasets.ImageFolder('/Volumes/software/project/cv_dataset/classify/dogs-vs-cats/dataset_new/train', transform)
# dataset_test = datasets.ImageFolder('/Volumes/software/project/cv_dataset/classify/dogs-vs-cats/dataset_new/test', transform_test)

#服务器训练数据路径：
dataset_train = datasets.ImageFolder('/home/yangdongben/project/deepLearning/pytorch/classify/dogs-vs-cats/dataset_new/train', transform)
dataset_test = datasets.ImageFolder('/home/yangdongben/project/deepLearning/pytorch/classify/dogs-vs-cats/dataset_new/test', transform_test)


ic(dataset_train.imgs)
'''
[('/Volumes/software/project/cv_dataset/classify/dogs-vs-cats/dataset_new/train/cat/cat.9996.jpg',0),
 ('/Volumes/software/project/cv_dataset/classify/dogs-vs-cats/dataset_new/train/cat/cat.9997.jpg',0)]
'''
# 对应文件夹的label
ic(dataset_train.class_to_idx)  # {'cat': 0, 'dog': 1}
# 对应文件夹的label
ic(dataset_test.class_to_idx) # {'cat': 0, 'dog': 1}
ic(dataset_test.classes)  # ['cat', 'dog']

# 导入数据
train_loader = torch.utils.data.DataLoader(dataset_train, num_workers=4, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset_test, num_workers=4, batch_size=BATCH_SIZE, shuffle=False)
modellr = 1e-4

# 实例化模型并且移动到GPU
criterion = nn.CrossEntropyLoss()
# model = effnetv2_s()
# num_ftrs = model.classifier.in_features
# model.classifier = nn.Linear(num_ftrs, 2)
model = torchvision.models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model.to(DEVICE)
# 选择简单暴力的Adam优化器，学习率调低
optimizer = optim.Adam(model.parameters(), lr=modellr)


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    modellrnew = modellr * (0.1 ** (epoch // 50))
    ic("lr:", modellrnew)
    for param_group in optimizer.param_groups:
        param_group['lr'] = modellrnew


# 定义训练过程

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    sum_loss = 0
    total_num = len(train_loader.dataset)
    # ic(total_num, len(train_loader))
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data).to(device), Variable(target).to(device)
        output = model(data)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ic_loss = loss.data.item()
        sum_loss += ic_loss
        if (batch_idx + 1) % 50 == 0:
            ic('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                       100. * (batch_idx + 1) / len(train_loader), loss.item()))
    ave_loss = sum_loss / len(train_loader)
    ic('epoch:{},ave_loss:{}'.format(epoch, ave_loss))

def val(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    total_num = len(test_loader.dataset)
    # ic(total_num, len(test_loader))
    with torch.no_grad():
        for data, target in test_loader:
            data, target = Variable(data).to(device), Variable(target).to(device)
            output = model(data)
            loss = criterion(output, target)
            _, pred = torch.max(output.data, 1)
            correct += torch.sum(pred == target)
            ic_loss = loss.data.item()
            test_loss += ic_loss
        correct = correct.data.item()
        acc = correct / total_num
        avgloss = test_loss / len(test_loader)
        ic('\nVal set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            avgloss, correct, len(test_loader.dataset), 100 * acc))


# 训练

for epoch in range(1, EPOCHS + 1):
    adjust_learning_rate(optimizer, epoch)
    train(model, DEVICE, train_loader, optimizer, epoch)
    val(model, DEVICE, test_loader)
torch.save(model, 'model.pth')