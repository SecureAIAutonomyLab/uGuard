import argparse
import time

from model import resnet
import torch
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms
import torch.nn as nn
from torch.autograd import Variable


from model.utils import load_filtered_state_dict, SaveBestModel, AverageMeter, accuracy
from data_wrapper import get_dataset, DataWrapper
import sys, tqdm, os
import numpy as np

def extract_distributions(args):
    transformations = transforms.Compose([transforms.Resize(320),
                                          transforms.RandomCrop(299), transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    train_x, train_y, classes_names = get_dataset(args.trainning_data_dir)
    trainning_dataset = DataWrapper(train_x, train_y, transformations)
    num_classes = len(classes_names)
    train_loader = torch.utils.data.DataLoader(dataset=trainning_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=False,
                                               num_workers=1)
    n = trainning_dataset.__len__()
    print("---> Number of samples in the distribution: ", n)
    model = resnet.ResNet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], num_classes)
    saved_state_dict = torch.load(args.saved_model)
    load_filtered_state_dict(model, saved_state_dict, ignore_layer=[], reverse=False)
    # multi-gpu
    # model = nn.DataParallel(model, device_ids=args.gpu)
    model.cuda()
    print(model)
    
    step = 0
    run_dataset(train_loader, model, step, args)
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
        
        
        
        

