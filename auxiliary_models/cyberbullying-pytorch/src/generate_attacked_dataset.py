# -*- coding: utf-8 -*-

import argparse
import time
import numpy as np

#from utils.modules import *

import torch
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms
import torch.nn as nn
from torch.autograd import Variable
import tqdm
import torchattacks

from tensorboardX import SummaryWriter


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Head pose estimation using the Hopenet network.')
    parser.add_argument('--gpu', dest='gpu', help='GPU device id to use', nargs='+',
            default=[0, 1], type=int)
    parser.add_argument('--num_epochs', dest='num_epochs', help='Maximum number of training epochs.',
          default=100, type=int)
    parser.add_argument('--batch_size', dest='batch_size', help='Batch size.',
          default=1, type=int)
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

    args = parser.parse_args()
    return args

def evaluate(eval_loader, model, args):
        
    for i, (images, labels) in tqdm.tqdm(enumerate(eval_loader)):
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()
        images = get_random_attack_simple(model, images, labels)
        save_name = "image_"+str(i).zfill(5)+".png"
        if labels == 0:
            print("../../datasets/cyberbullying_attacked_2/test/cyberbullying/"+ save_name)
            torchvision.utils.save_image(images, "../../datasets/cyberbullying_attacked_2/train/cyberbullying/"+ save_name)
        else:
            print('../../datasets/cyberbullying_attacked_2/test/non_cyberbullying/'+ save_name)
            torchvision.utils.save_image(images, '../../datasets/cyberbullying_attacked_2/train/non_cyberbullying/'+ save_name)


def get_random_attack(model, images, labels):
    atk_list = ['Clean', 'PGD', 'CW', 'DeepFool', 'Jitter', 'Pixl', 'SparseFool']
    probs = [0.250, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]
    attack_name = np.random.choice(atk_list, 1, p=probs)
    print("Attacked BY: ", attack_name)
    if attack_name == 'Clean':
        return images
    elif attack_name == 'PGD':
        atk = torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=4)
        return atk(images, labels)
    elif attack_name == 'CW':
        atk = torchattacks.CW(model, c=1e-4, kappa=0, steps=100, lr=0.01)
        return atk(images, labels)
    elif attack_name == 'AutoAttack':
        atk = torchattacks.AutoAttack(model, norm='Linf', eps=.3, version='standard', n_classes=2, seed=None, verbose=False)
        return atk(images, labels)
    elif attack_name == 'DeepFool':
        atk = torchattacks.DeepFool(model, steps=50, overshoot=0.02)
        return atk(images, labels)
    elif attack_name == 'Jitter':
        atk = torchattacks.Jitter(model, eps=0.3, alpha=2/255, steps=40, scale=10, std=0.1, random_start=True)
        return atk(images, labels)
    elif attack_name == 'Pixl':
        atk = torchattacks.Pixle(model, x_dimensions=(0.1, 0.2), restarts=100, max_iterations=50)
        return atk(images, labels)
    elif attack_name == 'SparseFool':
        atk = torchattacks.SparseFool(model, steps=20, lam=3, overshoot=0.02)
        return atk(images, labels)

def get_random_attack_simple(model, images, labels):
    atk_list = ['Clean', 'PGD', 'BIM', 'Square']
    probs = [0.25, 0.25, 0.25, 0.25]
    attack_name = np.random.choice(atk_list, 1, p=probs)
    print("Attacked BY: ", attack_name)
    if attack_name == 'Clean':
        return images
    elif attack_name == 'PGD':
        atk = torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=10)
        return atk(images, labels)
    elif attack_name == 'BIM':
        atk = attack = torchattacks.BIM(model, eps=4/255, alpha=1/255, steps=10)
        return atk(images, labels)
    elif attack_name == 'Square':
        atk = torchattacks.Square(model, norm='Linf', n_queries=500, n_restarts=1, eps=16/255, p_init=.8, seed=0, verbose=False, loss='margin', resc_schedule=True)
        return atk(images, labels)
    # elif attack_name == 'AutoAttack':
    #     atk = torchattacks.AutoAttack(model, norm='Linf', eps=.3, version='standard', n_classes=2, seed=None, verbose=False)
    #     return atk(images, labels)
    # elif attack_name == 'DeepFool':
    #     atk = torchattacks.DeepFool(model, steps=50, overshoot=0.02)
    #     return atk(images, labels)
    # elif attack_name == 'Jitter':
    #     atk = torchattacks.Jitter(model, eps=0.3, alpha=2/255, steps=40, scale=10, std=0.1, random_start=True)
    #     return atk(images, labels)
    # elif attack_name == 'Pixl':
    #     atk = torchattacks.Pixle(model, x_dimensions=(0.1, 0.2), restarts=100, max_iterations=50)
    #     return atk(images, labels)
    # elif attack_name == 'SparseFool':
    #     atk = torchattacks.SparseFool(model, steps=20, lam=3, overshoot=0.02)
    #     return atk(images, labels)


def main(args):
    cudnn.enabled = True    
    if args.extract_distributions!="None":
        extract_distributions_fn(args);
    print('Loading data.')

    transform_test = transforms.Compose([transforms.Resize(320),
                                         transforms.RandomCrop(299), transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                              std=[0.229, 0.224, 0.225])])

    test_path = args.validation_data_dir
    testset = torchvision.datasets.ImageFolder(test_path, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)

    net = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
    net.fc = nn.Linear(2048, 2, bias=True)
    
    #net = resnet50()
    net.load_state_dict(torch.load(args.saved_model), strict=False)
    net.cuda()
    
    crossEntropyLoss = nn.CrossEntropyLoss().cuda()    
       
    evaluate(testloader, net, args)



if __name__ == '__main__':
    args = parse_args()

    main(args)



