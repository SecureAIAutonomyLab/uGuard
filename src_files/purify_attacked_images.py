import os
import numpy as np
import cv2
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import numpy as np

from data_wrapper import get_dataset, DataWrapper

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from sporco import util
from sporco import signal
from sporco import plot
from sporco.metric import psnr
from sporco.admm import cbpdn
from multiprocessing import Pool

import os
import argparse
import sys
import numpy as np
import pickle

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset, Dataset
from model import resnet
from model.utils import load_filtered_state_dict
import tqdm

from PIL import Image
from matplotlib import cm


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Reconstruction Dictionaries Creation.')
    parser.add_argument('--gpu', dest='gpu', help='GPU device id to use', nargs='+', default=[0, 1], type=int)
    parser.add_argument('--eval_data_dir', dest='eval_data_dir', help='Directory path for data that needs to be purified.', default='../datasets/nsfw/val_balanced/', type=str)
    parser.add_argument('--distribution_files', dest='distributions', help='Path for class distributions.', default='../distributions_clean_model/nsfw/', type=str)
    parser.add_argument('--dictionary_files', dest='dictionaries', help='Path for reconstruction dictionaries.', default='../clean_dictionaries/', type=str)
    parser.add_argument('--robust_model', help='Path of robust model.',
          default='../auxiliary_models/nsfw-pytorch/models/nsfw_robust/epoch_11.pkl', type=str)
    parser.add_argument('--save_path', dest='purified_images_folder', help='Directory path to save dictionaries.', default='../purified_images/nsfw/', type=str)
    args = parser.parse_args()
    return args

def load_images(path):
    all_images = []
    for folder in sorted(os.listdir(path)):
        images = []
        full_path = os.path.join(path, folder)
        img_list = os.listdir(full_path)
        for img in tqdm.tqdm(img_list):
            img_path = os.path.join(full_path, img)
            images.append(np.array(Image.open(img_path)))
        all_images.append(images)
    return all_images

def reconstruct_from_my_D(img,D):
    
    npd = 16
    fltlmbd = 5
    sl, sh = signal.tikhonov_filter(img, fltlmbd, npd)
    lmbda = 0.2
    opt = cbpdn.ConvBPDN.Options({'Verbose': True, 'MaxMainIter': 250,
                              'RelStopTol': 1e-3, 'AuxVarObj': False})
    b = cbpdn.ConvBPDN(D[:,:,:,0,:], sh, lmbda, opt)
    X = b.solve()
    shr = b.reconstruct().squeeze()
    imgr = sl + shr
    return imgr

def purify(loader, model, distributions, dictionaries, args):
    mse = nn.MSELoss().cuda()
    model = model.cuda()
    for i, (images, labels, names) in tqdm.tqdm(enumerate(loader)):
        feature = model(images.cuda(), output_features=True)
        mse_dist_0 = mse(distributions[0][0].cuda(), feature.squeeze(0))
        mse_dist_1 = mse(distributions[1][0].cuda(), feature.squeeze(0))
        if mse_dist_0<mse_dist_1:
            img = reconstruct_from_my_D(images[0].permute(1,2,0).detach().cpu().numpy(), dictionaries[0])
        else:
            img = reconstruct_from_my_D(images[0].permute(1,2,0).detach().cpu().numpy(), dictionaries[1])
        save_path = os.path.join(args.purified_images_folder, os.path.join(names[0].split('/')[-2],names[0].split('/')[-1]))
        torchvision.utils.save_image(torch.tensor(img).permute(2,0,1), save_path)

def main(args):
    
    test_x, test_y, classes_names = get_dataset(args.eval_data_dir)
    transformations = transforms.Compose([transforms.ToTensor()])
    num_classes = len(classes_names)
    distributions = [torch.from_numpy(np.load(os.path.join(args.distributions,'safe_samples_dist.npy'))), torch.from_numpy(np.load(os.path.join(args.distributions,'unsafe_samples_dist.npy')))]
    dictionaries = [torch.from_numpy(np.load(os.path.join(args.dictionaries,'dictionary_class_0.npy'))), torch.from_numpy(np.load(os.path.join(args.dictionaries,'dictionary_class_1.npy')))]
    
    print("classes : {}".format(classes_names))
    eval_dataset = DataWrapper(test_x, test_y, transformations)
    eval_loader = torch.utils.data.DataLoader(dataset=eval_dataset,
                                               batch_size=1,
                                               shuffle=True,
                                               num_workers=4)
    n = eval_dataset.__len__()
    print("Size of Training Set: ", n)
    model = resnet.ResNet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], num_classes)
    if args.robust_model:
        print('Loading model.')
        saved_state_dict = torch.load(args.robust_model)
        if 'resnet' in args.robust_model:
            load_filtered_state_dict(model, saved_state_dict, ignore_layer=[], reverse=False)
        else:
            load_filtered_state_dict(model, saved_state_dict, ignore_layer=[], reverse=True)
    purify(eval_loader, model, distributions, dictionaries, args)
    

if __name__ == '__main__':
    args = parse_args()
    main(args)

    

        
        
    
# for batch_idx, (inputs, targets) in enumerate(testloader):
#     inputs, targets = inputs, targets.to(device)
#     # inputs = atk(inputs, targets).to(device)
#     imgs_in_batch = []
#     _, latent = net(inputs.to(device))
#     latent_pca = pca.transform(np.reshape(latent.detach().cpu().clone().numpy(),
#                                           [latent.detach().cpu().clone().numpy().shape[0], -1]))
#     latent_cluster = kmeans.predict(latent_pca)
#     for img_idx,img in enumerate(inputs):
#         a = torch.tensor(reconstruct_from_my_D((img.permute(1,2,0)).detach().cpu().numpy(), D[latent_cluster[img_idx]])).permute(2,0,1)
#         imgs_in_batch.append(a)
#     aux = torch.stack(imgs_in_batch)
#     imgs_in_batch = (aux.to(device))#.permute(0,3,1,2)
#     outputs, _ = net(imgs_in_batch)
#     loss = criterion(outputs, targets)
#     test_loss += loss.item()
#     _, predicted = outputs.max(1)
#     total += targets.size(0)
#     correct += predicted.eq(targets).sum().item()
#     print("\r{} completed: {:.2%}".format(
#             'Testing', batch_idx / len(testloader)), end="")
#     sys.stdout.flush()
#     accuracy = 100.*correct/total
# print('Accuracy', 100.*correct/total)
