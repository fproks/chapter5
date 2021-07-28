import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transform


# 用于ResNet18和34的残差块，用的是2个3x3的卷积
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        # 经过处理后的x要与x的维度相同(尺寸和深度)
        # 如果不相同，需要添加卷积+BN来变换为同一维度
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# 用于ResNet50,101和152的残差块，用的是1x1+3x3+1x1的卷积
class Bottleneck(nn.Module):
    # 前面1x1和3x3卷积的filter个数相等，最后1x1卷积是其expansion倍
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, linear_size=32):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(int((linear_size ** 2) / 2) * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(linear_size=32):
    return ResNet(BasicBlock, [2, 2, 2, 2], linear_size=linear_size)


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


def test():
    net = ResNet18(64)
    a = np.random.random((64, 64, 3))
    b = transform.ToTensor()(a).to(torch.float32)
    print(b.shape)
    y = net(torch.unsqueeze(b, 0))
    print(y.size())
    print(y)


"""
https://blog.csdn.net/qq_36370187/article/details/103103382

"""
test()


def PretreatmentData():
    import torch
    from torch.utils.data import random_split
    import torch.utils.data as Data
    train_x = torch.randn(10, 8)
    train_y = torch.randn(10, )
    import torchvision
    import torchvision.transforms as transforms
    import torch.optim as optim

    '''
    transforms.Compose 这个类的主要作用是串联多个图片变换的操作
    transforms.RandomCrop  裁剪大小
    RandomHorizontalFlip   水平翻转
    Normalize  标准化
    '''
    transfrom_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transfrom_train)

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    print(type(trainset))
    print(type(testset))
    train = np.transpose(np.concatenate((trainset.data, testset.data)), [0, 3, 1, 2])
    target = trainset.targets + testset.targets
    print(train.shape)
    print(len(target))
    # tr=transforms.ToTensor()(train)
    # print(tr.shape)
    dataset = Data.TensorDataset(torch.tensor(train).float(), torch.tensor(target).long())
    print(len(dataset))
    train_data, eval_data = random_split(dataset, [round(0.05 * train.shape[0]), round(0.95 * train.shape[0])],
                                         generator=torch.Generator().manual_seed(42))  # 把数据机随机切分训练集和验证集
    print(len(train_data))
    train_loader = Data.DataLoader(dataset=train_data, batch_size=100, shuffle=True, num_workers=0, drop_last=False)
    test_loader = Data.DataLoader(dataset=eval_data, batch_size=100, shuffle=False, num_workers=2)
    # for step,(train_x,train_y) in enumerate(loader):
    #     print(step,':',(train_x,train_y))
    net = ResNet18()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    for epoch in range(0, 100):
        print(f'\n EPOCH: {epoch + 1}')
        net.train()
        sum_loss = 0.0
        correct = 0.0
        total = 0.0
        for i, data in enumerate(train_loader, 0):
            length = len(train_loader)
            inputs, labeles = data
            print(inputs.shape)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labeles)
            loss.backward()
            optimizer.step()

            sum_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labeles.size(0)
            correct += predicted.eq(labeles.data).cpu().sum()
            print(
                f'[epoch:{epoch + 1},iter:{i + 1 + epoch * length}] Loss:{sum_loss / (i + 1):.3f} '
                f'|Acc:{100. * correct / total:.3f}')
            if sum_loss<=.01 or correct==total:
                print(f"train stop when loss == {sum_loss} ,epoch:{epoch+1},iter:{i+1+epoch*length}")
                break
        if sum_loss<=.01 or correct==total:
            break

    print('Waiting Test...')
    with torch.no_grad():
        correct = 0
        total = 0
        for i,data in enumerate(test_loader,0):
            net.eval()
            images, labeles = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labeles.size(0)
            tmp_corr=(predicted==labeles).sum()
            print(f'Test {i} ac is ------->{100*tmp_corr/labeles.size(0):.3f}')
            correct += tmp_corr
        print(f'Test \‘ ac is:{100 * correct / total:.3f}')


if __name__ == '__main__':
    PretreatmentData()
