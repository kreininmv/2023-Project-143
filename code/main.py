'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import sys

from adamwh import MyAdamW

from resnet import ResNet18

from tqdm import tqdm
#from torch.utils import progress_bar
from torch.utils.tensorboard import SummaryWriter

def test(epoch):
    global best_acc
    global test_step
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        loop = tqdm(enumerate(testloader), total=len(testloader), leave=False)
        for batch_idx, (inputs, targets) in loop:
            test_step += 1
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            writer.add_scalar('Testing loss', test_loss/(batch_idx+1),global_step=test_step)
            writer.add_scalar('Testing accuracy', 100.*correct/total, global_step=test_step)
            
            loop.set_description(f"{full_name}Epoch (Test)[{epoch}/{num_epoch}]")
            loop.set_postfix(loss= test_loss/(batch_idx+1), acc=100.*correct/total,correct=correct, total=total)

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, f'./checkpoint/ckpt.pth')
        best_acc = acc
    return test_loss, correct, total

# Training
def train(epoch):
    #print('\nEpoch: %d' % epoch)
    global train_step
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    loop = tqdm(enumerate(trainloader), total=len(trainloader), leave=False)
    
    for batch_idx, (inputs, targets) in loop:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_step += 1

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        writer.add_scalar('Training loss', train_loss/(batch_idx+1),global_step=train_step)
        writer.add_scalar('Training accuracy', 100.*correct/total, global_step=train_step)

        loop.set_description(f"{full_name} Epoch (Train)[{epoch}/{num_epoch}]")
        loop.set_postfix(loss= train_loss/(batch_idx+1), acc=100.*correct/total, correct=correct, total=total)

    return train_loss, correct, total
if __name__ == '__main__': # protect your program's entry point
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch


    criterion = nn.CrossEntropyLoss() 
    num_epoch = 100

    betas = (0.9, 0.999)
    learning_rates = [0.01, 0.005, 0.0005]
    weight_decayes = [0.005, 0.0005, 0.00005]
    optimizers = [optim.SGD, optim.AdamW, MyAdamW, optim.Adam]
    names = ['SGD + L2', 'AdamW', 'MyAdamW', 'Adam+L2']

    for lr in learning_rates:
        for wd in weight_decayes:
            for opt, name in zip(optimizers, names):
                # Data
                print('==> Preparing data..')
                transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])

                transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])

                trainset = torchvision.datasets.CIFAR10(
                    root='./data', train=True, download=True, transform=transform_train)
                trainloader = torch.utils.data.DataLoader(
                    trainset, batch_size=128, shuffle=True, num_workers=2)

                testset = torchvision.datasets.CIFAR10(
                    root='./data', train=False, download=True, transform=transform_test)
                testloader = torch.utils.data.DataLoader(
                    testset, batch_size=100, shuffle=False, num_workers=2)

                classes = ('plane', 'car', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck')

                # Model
                print('==> Building model..')
                net = ResNet18()

                net = net.to(device)

                if device == 'cuda':
                    net = torch.nn.DataParallel(net)
                    cudnn.benchmark = True

                if name == 'SGD + L2':
                    optimizer = opt(net.parameters(), lr=lr, weight_decay=wd)
                else:
                    optimizer = opt(net.parameters(), lr=lr, betas=betas, eps=1e-8, weight_decay=wd)

                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=0.00001) 

                full_name = f'{name} - lr({lr}), weight_decay({wd})'
                path = f'runs/cifar10/{full_name}'
                writer = SummaryWriter(path)
                train_step = 0
                test_step = 0

                for epoch in range(start_epoch, start_epoch + num_epoch):
                    loss_train, correct_train, total_train = train(epoch)
                    loss_test, correct_test, total_test = test(epoch)

                    writer.add_scalar('Testing loss (epoch)', loss_test,epoch)
                    writer.add_scalar('Testing accuracy (epoch)', 100.*correct_test/total_test, epoch)

                    writer.add_scalar('Training loss (epoch)', loss_train, epoch)
                    writer.add_scalar('Training accuracy (epoch)', 100.*correct_train/total_train, epoch)

                    scheduler.step()