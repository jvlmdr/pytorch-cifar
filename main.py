'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import wandb

import os
import argparse

import models
from utils import progress_bar


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--mom', default=0.9, type=float)
parser.add_argument('--wd', default=5e-4, type=float)
parser.add_argument('--epochs', default=200, type=int)
# parser.add_argument('--resume', '-r', action='store_true',
#                     help='resume from checkpoint')
parser.add_argument('--arch', default='PreActResNet18', help='Model')
parser.add_argument('--dataset', default='CIFAR10', help='Dataset')
parser.add_argument('--train_aug', default='cifar')
parser.add_argument('--seed', default=0, type=int)
args = parser.parse_args()

model_fn = getattr(models, args.arch)
dataset_fn = getattr(torchvision.datasets, args.dataset)
data_root = './data/' + args.dataset.lower()
torch.manual_seed(args.seed)

wandb.init(project='deepopt-pytorch', config=args)
# config = wandb.config  # Why?

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    *([transforms.RandomCrop(32, padding=4),
       transforms.RandomHorizontalFlip()] if args.train_aug == 'cifar' else []),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = dataset_fn(
    root=data_root, train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = dataset_fn(
    root=data_root, train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

# classes = ('plane', 'car', 'bird', 'cat', 'deer',
#            'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
net = model_fn()
# net = models.VGG('VGG19')
# net = models.ResNet18()
# net = models.PreActResNet18()
# net = models.PreActResNet50()
# net = models.GoogLeNet()
# net = models.DenseNet121()
# net = models.ResNeXt29_2x64d()
# net = models.MobileNet()
# net = models.MobileNetV2()
# net = models.DPN92()
# net = models.ShuffleNetG2()
# net = models.SENet18()
# net = models.ShuffleNetV2(1)
# net = models.EfficientNetB0()
# net = models.RegNetX_200MF()
# net = models.SimpleDLA()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

# if args.resume:
#     # Load checkpoint.
#     print('==> Resuming from checkpoint..')
#     assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
#     checkpoint = torch.load('./checkpoint/ckpt.pth')
#     net.load_state_dict(checkpoint['net'])
#     best_acc = checkpoint['acc']
#     start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=args.mom, weight_decay=args.wd)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return {
        'loss': train_loss/(batch_idx+1),
        'acc': correct/total,
    }


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item() 
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return {
        'loss': test_loss/(batch_idx+1),
        'acc': correct/total,
    }

    # # Save checkpoint.
    # acc = 100.*correct/total
    # if acc > best_acc:
    #     print('Saving..')
    #     state = {
    #         'net': net.state_dict(),
    #         'acc': acc,
    #         'epoch': epoch,
    #     }
    #     if not os.path.isdir('checkpoint'):
    #         os.mkdir('checkpoint')
    #     torch.save(state, './checkpoint/ckpt.pth')
    #     best_acc = acc


for epoch in range(start_epoch, start_epoch+args.epochs):
    train_metrics = train(epoch)
    test_metrics = test(epoch)
    metrics = {
        **{'train_' + k: v for k, v in train_metrics.items()},
        **{'test_' + k: v for k, v in test_metrics.items()},
    }
    wandb.log(metrics, step=epoch + 1)
    scheduler.step()
