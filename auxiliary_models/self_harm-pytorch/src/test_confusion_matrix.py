
import argparse
import time

from model import resnet

import torch
import torch.backends.cudnn as cudnn
import torchvision
from torch.autograd import Variable
from torchvision import transforms
import torch.nn as nn
from model.utils import load_filtered_state_dict, SaveBestModel, AverageMeter, accuracy
from data_wrapper import get_dataset, DataWrapper
from PIL import Image
import glob
import os
from sklearn.metrics import confusion_matrix
import numpy as np

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Head pose estimation using the Hopenet network.')
    parser.add_argument('--gpu', dest='gpu', help='GPU device id to use', nargs='+', 
        default=[0], type=int)
    parser.add_argument('--batch_size', dest='batch_size', help='Batch size.',
        default=32, type=int)
    parser.add_argument('--image_size', dest='image_size', help='Image size.',
        default=224, type=int)
        
    parser.add_argument('--test_data_dir', dest='test_data_dir', help='Directory path for validation data.',
        default='/workspace/adv_robustness/cyberbullying_purification/samuel_src/datasets/nsfw/val_balanced/', type=str)
        
    parser.add_argument('--saved_model', help='Path of model snapshot for continue training.',
        default='/workspace/adv_robustness/cyberbullying_purification/samuel_src/auxiliary_models/nsfw-pytorch/models/nsfw/epoch_75.pkl', type=str)

    parser.add_argument('--num_classes', help='num of classes.', default=2, type=int)

    args = parser.parse_args()
    return args


def main(args):    
    
    model = resnet.ResNet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], args.num_classes)
    saved_state_dict = torch.load(args.saved_model)    

    transformations = transforms.Compose([transforms.Resize((args.image_size, args.image_size)),transforms.ToTensor()])
    
    if args.gpu[0] >=0:        
        cudnn.enabled = True 
        softmax = nn.Softmax().cuda()
        model.cuda()
    else:
        softmax = nn.Softmax()        

    load_filtered_state_dict(model, saved_state_dict, ignore_layer=[], reverse=True)

    test_x, test_y, classes_names = get_dataset(args.test_data_dir)
    test_dataset = DataWrapper(test_x, test_y, transformations, augumentation=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=1)    

    classes, indices = np.unique(test_y, return_index=True)        
    
    #n = (test_dataset.__len__() + args.batch_size - 1) / args.batch_size * args.batch_size
    n = test_dataset.__len__()
        
    y_pred = np.zeros((n))
    y = np.zeros((n))
    count = 0
    import tqdm
    for i, (images, labels, names) in tqdm.tqdm(enumerate(test_loader)):
        images = Variable(images)
        labels = Variable(labels)
        if args.gpu[0] >=0: 
            images = images.cuda()
            labels = labels.cuda()
        # print("True Label: ", labels)
        label_pred = model(images)
        # print("Label Pred: ", label_pred)
        # label_pred = softmax(label_pred)
        # print("Label Pred Soft: ", label_pred)


        n = images.size()[0]
        y_pred_tag = torch.log_softmax(label_pred, dim = 1)
        _, y_pred_tags = torch.max(y_pred_tag, dim = 1)
        correct_results_sum = (y_pred_tags == labels).sum().float()
        acc = correct_results_sum/labels.shape[0]
        acc = torch.round(acc * 100)
        print("Current ACC: ", acc)
        # _, label_pred = label_pred.topk(1, 1, True, True)
        y_pred[count:count+n] = y_pred_tags.cpu().numpy()
        y[count:count+n] = labels.data.cpu().numpy()
           
        # import sys
        # sys.exit()
        count += n
        
        
        
        
        # print("Label true: {}, Label pred: {}".format(labels, y_pred_tags))
        # label_pred = softmax(label_pred)
        # correct_results_sum = (y_pred_tags == labels).sum().float()

    plot(y, y_pred, classes_names)

def plot(y_test, y_pred, classes_names):    
    matrix = confusion_matrix(y_test, y_pred)
    print(matrix)

    """
    import matplotlib.pyplot as plt
    import itertools

    plt.figure()
    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.get_cmap('Blues'))
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes_names))
    plt.xticks(tick_marks, classes_names, rotation=45)
    plt.yticks(tick_marks, classes_names)

    fmt = '.2f' if normalize else 'd'
    thresh = matrix.max() / 2.
    for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
        plt.text(j, i, format(matrix[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if matrix[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    plt.show()
    '"""

if __name__ == '__main__':
    args = parse_args()

    main(args)