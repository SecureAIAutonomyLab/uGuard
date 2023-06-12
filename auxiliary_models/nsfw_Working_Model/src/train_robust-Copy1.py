import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import argparse

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np
import itertools
from collections import Counter
import matplotlib.pyplot as plt
import tqdm
from utils.modules import *



device = 'cuda' if torch.cuda.is_available() else 'cpu'


class fully_connected(nn.Module):
    def __init__(self):
        super(fully_connected, self).__init__()
        self.fc = nn.Linear(2048, 2, bias=True)
    def forward(self, x, output_features=False, train_robust=False):
        if output_features:
            return x
        if train_robust:
            return [self.fc(x), x]
        return self.fc(x)

def adjust_weight_names(weights_path):
    weights = torch.load(weights_path)
    weights["fc.fc.weight"] = weights["fc.weight"].clone()
    weights["fc.fc.bias"] = weights["fc.bias"].clone()
    print (weights["fc.fc.bias"])
    print (weights["fc.bias"])
    
    weights.pop("fc.weight")
    weights.pop("fc.bias")
    return weights

def train(epoch, net, trainloader, optimizer, scheduler, criterion, safe_distributions, unsafe_distributions):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    batch_size = args.batch_size
    safe_dist = torch.from_numpy(safe_distributions).cuda()
    unsafe_dist = torch.from_numpy(unsafe_distributions).cuda()
    mse = nn.MSELoss().cuda()
    for batch_idx, (inputs, targets) in tqdm.tqdm(enumerate(trainloader)):
        batch_dist_gt = safe_dist.unsqueeze(0).repeat(batch_size, 1,1)
        inputs, targets = inputs, targets.to(device)
        for i in targets.nonzero():
            batch_dist_gt[i] = unsafe_dist
        if(np.random.uniform(0, 1)<0.3):
            inputs = inputs + torch.rand(inputs.shape)
            inputs = torch.clamp(inputs, max=1, min=0)
        inputs = inputs.to(device)
        outputs, features = net(inputs, train_robust=True)
        dist_loss = mse(features.mean(dim=0), batch_dist_gt[0][0]) + mse(features.std(dim=0), batch_dist_gt[0][1])
        loss = criterion(outputs,targets) + 0.3*dist_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

def test(epoch, net, testloader):
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
            acc = 100.*correct/total
    return acc

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Head pose estimation using the Hopenet network.')
    parser.add_argument('--gpu', dest='gpu', help='GPU device id to use', nargs='+',
            default=[0, 1], type=int)
    parser.add_argument('--num_epochs', dest='num_epochs', help='Maximum number of training epochs.',
          default=100, type=int)
    parser.add_argument('--batch_size', dest='batch_size', help='Batch size.',
          default=32, type=int)
    parser.add_argument('--lr', dest='lr', help='Base learning rate.',
          default=0.01, type=float)
    parser.add_argument('--trainning_data_dir', dest='trainning_data_dir', help='Directory path for trainning data.',
          default='./data/train', type=str)
    parser.add_argument('--validation_data_dir', dest='validation_data_dir', help='Directory path for validation data.',
          default='./data/test', type=str)
    parser.add_argument('--save_path', dest='save_path', help='Path of model snapshot for save.',
          default='./models', type=str)
    parser.add_argument('--saved_model', help='Path of model snapshot for continue training.',
          default='./models/resnet50-19c8e357.pth', type=str)
    parser.add_argument('--extract_distributions', help='Call model in eval mode to extract feature distributions',
          default='None', type=str)
    parser.add_argument('--save_dist_path', help='Path where to save the distributions',
          default='/workspace/adv_robustness/cyberbullying_purification/samuel_src/distributions_clean_model/aux', type=str)
    parser.add_argument('--safe_clean_distributions', help='Path of safe clean distributions.',
          default='../../distributions_clean_model/nsfw/safe_samples_dist.npy', type=str)
    parser.add_argument('--unsafe_clean_distributions', help='Path of unsafe clean distributions',
          default='../../distributions_clean_model/nsfw/unsafe_samples_dist.npy', type=str)

    args = parser.parse_args()
    return args


def main(args):
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    batch_size = args.batch_size
    transform_train = transforms.Compose([transforms.Resize(320),
                                          transforms.RandomCrop(299), transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                               std=[0.229, 0.224, 0.225])])


    transform_test = transforms.Compose([transforms.Resize(320),
                                         transforms.RandomCrop(299), transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                              std=[0.229, 0.224, 0.225])])
    train_path = args.trainning_data_dir
    trainset = torchvision.datasets.ImageFolder(train_path, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_path = args.validation_data_dir
    testset = torchvision.datasets.ImageFolder(test_path, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    net = resnet50()
    # net = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=False)
    net.fc = fully_connected()
    net.load_state_dict(torch.load(args.saved_model), strict=True)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), 
                                lr=0.00001, 
                                weight_decay=2e-4,
                                momentum=0.9) 
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40000, 80000], gamma=0.1)#4000
    criterion = nn.CrossEntropyLoss().to(device)
    safe_distributions = np.load(args.safe_clean_distributions, allow_pickle=True)
    print(safe_distributions.shape)
    unsafe_distributions = np.load(args.unsafe_clean_distributions, allow_pickle=True)
    print(unsafe_distributions.shape)
    
    best_test_acc = 0        
    for epoch in range(start_epoch, start_epoch+25):
        train(epoch, net, trainloader, optimizer, scheduler, criterion, safe_distributions, unsafe_distributions)
        acc = test(epoch, net, testloader)
        if acc>=best_test_acc:
            #print('==> Saving model to: ', args.save_path)
            torch.save(net.state_dict(), args.save_path + 'resnet_'+str(epoch).zfill(3)+'.pth')
            print(acc)
            best_test_acc = acc

if __name__ == '__main__':
    args = parse_args()
    main(args)
