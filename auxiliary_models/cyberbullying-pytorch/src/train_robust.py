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

from model.utils import load_filtered_state_dict, SaveBestModel, AverageMeter, accuracy
from data_wrapper import get_dataset, DataWrapper
from tensorboardX import SummaryWriter


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Head pose estimation using the Hopenet network.')
    parser.add_argument('--gpu', dest='gpu', help='GPU device id to use', nargs='+',
            default=[0, 1], type=int)
    parser.add_argument('--num_epochs', dest='num_epochs', help='Maximum number of training epochs.',
          default=50, type=int)
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
    parser.add_argument('--safe_clean_distributions', help='Path of safe clean distributions.',
          default='../../distributions_clean_model/nsfw/safe_samples_dist.npy', type=str)
    parser.add_argument('--unsafe_clean_distributions', help='Path of unsafe clean distributions',
          default='../../distributions_clean_model/nsfw/unsafe_samples_dist.npy', type=str)
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
        label_pred = model(images)
        y_pred_tag = torch.log_softmax(label_pred, dim = 1)
        _, y_pred_tags = torch.max(y_pred_tag, dim = 1)
        
        # print("Label true: {}, Label pred: {}".format(labels, y_pred_tags))
        # label_pred = softmax(label_pred)
        correct_results_sum = (y_pred_tags == labels).sum().float()
        num_correct += correct_results_sum
        acc = correct_results_sum/labels.shape[0]
        acc = torch.round(acc * 100)
        # print("Batch Acc: ", acc)
#         prec = accuracy(y_pred_tags, labels, topk=(1,))
       
        top_prec.update(acc/100)
        

    print('evaluate * Prec@1 {top:.3f}'.format(top=top_prec.avg))
    # print("My Eval: ", num_correct/total)
    writer.add_scalar('eval_prec', top_prec.avg, step)
        
    Save_model.save(model, top_prec.avg, epoch)
    

def train(train_loader, model, criterion, optimizer, writer, batch_size, epoch, step, n, safe_distributions, unsafe_distributions):
    last_time = time.time()
    safe_dist = torch.from_numpy(safe_distributions).cuda()
    unsafe_dist = torch.from_numpy(unsafe_distributions).cuda()
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
        writer.add_scalar('learning_rate', lr, step)
        break 
    softmax = nn.Softmax().cuda()
    mse = nn.MSELoss().cuda()
    
    for i, (images, labels, name) in enumerate(train_loader):
        batch_dist_gt = safe_dist.unsqueeze(0).repeat(batch_size, 1,1)
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()
        label_pred, features = model(images, train_robust=True)
        for i in labels.nonzero():
            batch_dist_gt[i] = unsafe_dist
        label_loss = criterion(label_pred, labels)
        dist_loss = mse(features.mean(dim=0), batch_dist_gt[0][0]) + mse(features.std(dim=0), batch_dist_gt[0][1])
        loss = label_loss + 0.3*dist_loss
        print("Labels loss: {}; Dist Loss: {};".format(label_loss.item(), dist_loss.item()))
        writer.add_scalar('loss', loss, step)            
        optimizer.zero_grad()
        loss.backward()            
        optimizer.step()

        if i % 10 == 0:
            curr_time = time.time()
            sps = 10.0 / (curr_time - last_time) * batch_size
            print("Epoch [{}], Iter [{}/{}]  {} samples/sec, Losses: {}".format(epoch+1, 
                i+1, n//batch_size, sps, loss.item()))
            
            last_time = curr_time

        step += 1

    # evaluate
    
    label_pred = softmax(label_pred)
    prec = accuracy(label_pred, labels, topk=(1,))
    print('training * Prec@1 {top:.3f}'.format(top=prec[0].item()))
    writer.add_scalar('training_prec', prec[0].item(), step)

    return step

def main(args):
    cudnn.enabled = True    
    if args.extract_distributions!="None":
        extract_distributions_fn(args);
    print('Loading data.')

    transformations = transforms.Compose([transforms.ToTensor()])

    train_x, train_y, classes_names = get_dataset(args.trainning_data_dir)
    test_x, test_y, _ = get_dataset(args.validation_data_dir)
    num_classes = len(classes_names)
    print("classes : {}".format(classes_names))
    
    trainning_dataset = DataWrapper(train_x, train_y, transformations)
    eval_dataset = DataWrapper(test_x, test_y, transformations)
        
    
    train_loader = torch.utils.data.DataLoader(dataset=trainning_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=4)######################
    
    eval_loader = torch.utils.data.DataLoader(dataset=eval_dataset,
                                               batch_size=1,
                                               shuffle=True,
                                               num_workers=4)
    n = trainning_dataset.__len__()
    print("Size of Training Set: ", n)
    model = resnet.ResNet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], num_classes)


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
    safe_distributions = np.load(args.safe_clean_distributions, allow_pickle=True)
    print(safe_distributions.shape)
    unsafe_distributions = np.load(args.unsafe_clean_distributions, allow_pickle=True)
    print(unsafe_distributions.shape)
    i = 0
    for epoch in range(args.num_epochs):
        scheduler.step()
        # evaluate(eval_loader, model, Writer, step, Save_model, epoch)
        step = train(train_loader, model, crossEntropyLoss, optimizer, Writer, args.batch_size, epoch, step, n, safe_distributions, unsafe_distributions)        
        torch.save(model.state_dict(), str(args.save_path)+'/epoch_'+str(i)+'.pth')
        i+=1

if __name__ == '__main__':
    args = parse_args()

    main(args)



