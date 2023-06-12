# -*- coding: utf-8 -*-

import argparse
import time
import numpy as np

from model import resnet
from model.dpn import dpn92

import torch
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms
import torch.nn as nn
from torch.autograd import Variable
from utils.extract_distributions import extract_distributions as extract_distributions_fn
import tqdm
import torchattacks

from model.utils import load_filtered_state_dict, SaveBestModel, AverageMeter, accuracy
from data_wrapper import get_dataset, DataWrapper
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



def evaluate(eval_loader, model, writer, step, Save_model, epoch):
        
    top_prec = AverageMeter()    
    softmax = nn.Softmax().cuda()
    num_correct = 0
    total = 0
    for i, (images, labels, names) in tqdm.tqdm(enumerate(eval_loader)):
        total+=images.shape[0]
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()
        images = get_random_attack(model, images, labels)
        save_name = str(names[0]).split('/')[-1]
        if labels == 0:
            torchvision.utils.save_image(images, "../../datasets/nsfw_attacked/test/neutral/"+ save_name)
        else:
            torchvision.utils.save_image(images, '../../datasets/nsfw_attacked/test/porn/'+ save_name.replace('jpg','png'))
        

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

    

def train(train_loader, model, criterion, optimizer, writer, batch_size, epoch, step, n):
    last_time = time.time()

    for param_group in optimizer.param_groups:
        lr = param_group['lr']
        writer.add_scalar('learning_rate', lr, step)
        break 
    softmax = nn.Softmax().cuda()

    for i, (images, labels, name) in enumerate(train_loader):
        
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()
        images = get_random_attack(model, images, labels)
        save_name = str(names[0]).split('/')[-1]
        if labels == 0:
            torchvision.utils.save_image(images, "../../datasets/nsfw_attacked/train/neutral/"+ save_name)
        else:
            torchvision.utils.save_image(images, '../../datasets/nsfw_attacked/train/porn/'+ save_name.replace('jpg','png'))

    return step

def main(args):
    cudnn.enabled = True    
    if args.extract_distributions!="None":
        extract_distributions_fn(args);
    print('Loading data.')

    transformations = transforms.Compose([transforms.Resize(320),
        transforms.RandomCrop(299), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    train_x, train_y, classes_names = get_dataset(args.trainning_data_dir)
    test_x, test_y, _ = get_dataset(args.validation_data_dir)
    num_classes = len(classes_names)
    print("classes : {}".format(classes_names))

    trainning_dataset = DataWrapper(train_x, train_y, transformations)
    eval_dataset = DataWrapper(test_x, test_y, transformations)

    train_loader = torch.utils.data.DataLoader(dataset=trainning_dataset,
                                               batch_size=1,
                                               shuffle=True,
                                               num_workers=1)######################
    
    eval_loader = torch.utils.data.DataLoader(dataset=eval_dataset,
                                               batch_size=1,
                                               shuffle=True,
                                               num_workers=1)
    n = trainning_dataset.__len__()
    print(n)

    # ResNet50 structure
    model = resnet.ResNet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], num_classes)
    # dpn 92
    #model = dpn92(num_classes=num_classes)

    if args.saved_model:
        print('Loading model.')
        saved_state_dict = torch.load(args.saved_model)

        # 'origin model from pytorch'
        if 'resnet' in args.saved_model:
            load_filtered_state_dict(model, saved_state_dict, ignore_layer=[], reverse=False)
        else:
            load_filtered_state_dict(model, saved_state_dict, ignore_layer=[], reverse=True)

    
    crossEntropyLoss = nn.CrossEntropyLoss().cuda()    
    # optimizer = torch.optim.Adam(model.parameters(), lr = args.lr )
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40, 60], gamma=0.1)    
    

    # multi-gpu
    model = nn.DataParallel(model, device_ids=args.gpu)
    model.cuda()

    Save_model = SaveBestModel(save_dir=args.save_path)
    Writer = SummaryWriter()
    step = 0
    
    epoch = 0    
    evaluate(eval_loader, model, Writer, step, Save_model, epoch)
    step = train(train_loader, model, crossEntropyLoss, optimizer, Writer, args.batch_size, epoch, step, n)        


if __name__ == '__main__':
    args = parse_args()

    main(args)



