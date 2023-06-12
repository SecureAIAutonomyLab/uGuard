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
from utils.extract_distributions import extract_distributions as extract_distributions_fn

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

def extract_distributions(args):
    transform_train = transforms.Compose([transforms.Resize(320),
                                          transforms.RandomCrop(299), transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                               std=[0.229, 0.224, 0.225])])

    train_path = args.trainning_data_dir
    trainset = torchvision.datasets.ImageFolder(train_path, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True, num_workers=2)

    
    net = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=False)
    net.fc = fully_connected()
    net.load_state_dict(torch.load(args.saved_model), strict=True)
    net.to(device)
    
    step = 0
    run_dataset(train_loader, net, step, args)
    sys.exit()

def run_dataset(loader, model, step, args):
    safe_samples = []
    unsafe_samples = []
    for i, (images, labels, names) in tqdm.tqdm(enumerate(loader), total=len(loader.dataset)):
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()
        feature = model(images, output_features=True)
        if labels.item() == 0:
            safe_samples.append(feature.detach().cpu().numpy())
        if labels.item() == 1:
            unsafe_samples.append(feature.detach().cpu().numpy())
        if i%1000==0 and i!=0:
            print("Unsafe Samples shape: ", np.array(unsafe_samples).reshape(-1,2048).mean(axis=1).shape)
            print("Safe Samples shape: ", np.array(safe_samples).reshape(-1,2048).mean(axis=1).shape)
            print("Dist Shape: ", np.array([np.array(unsafe_samples).reshape(-1,2048).mean(axis=0), np.array(unsafe_samples).reshape(-1,2048).std(axis=0)]).shape)
            np.save(os.path.join(args.save_dist_path, 'unsafe_samples_dist.npy'), np.array([np.array(unsafe_samples).reshape(-1,2048).mean(axis=0), np.array(unsafe_samples).reshape(-1,2048).std(axis=0)]), allow_pickle=True)
            np.save(os.path.join(args.save_dist_path, 'safe_samples_dist.npy'), np.array([np.array(safe_samples).reshape(-1,2048).mean(axis=0), np.array(safe_samples).reshape(-1,2048).std(axis=0)]),  allow_pickle=True)

        

        

