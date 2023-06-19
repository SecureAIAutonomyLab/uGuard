#Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from multiprocessing import Pool


import torchvision
from torchvision import models as tvmodels
from torchsummary import summary

import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset, Dataset

import torchvision.models as torchvisionmodels

import os
import numpy as np
import cv2
import argparse
import sys

import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
import itertools
import more_itertools

import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt

from captum.attr import LayerGradCam
from captum.attr import visualization
from PIL import Image
import shutil

import numpy as np
from dask_image.imread import imread
from dask_image import ndfilters, ndmorph, ndmeasure
import matplotlib.pyplot as plt
from dask_image import ndmeasure

from operator import itemgetter


import argparse

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='CSRA Image Obfuscation')
    parser.add_argument('--gpu', dest='gpu', help='GPU device id to use', nargs='+', default=[0, 1], type=int)
    parser.add_argument('--classifier_model', help='Path of classifier model.',
          default='../auxiliary_models/nsfw-pytorch/models/cyberbullying_robust_2/epoch_41.pth', type=str)
    parser.add_argument('--save_path', dest='masked_images_folder', help='Directory path to save masked images.', default='../masked_images/nsfw/', type=str)
    parser.add_argument('--test_data_dir', dest='test_data_dir', help='Directory path for data to be obfuscated.', default='../datasets/nsfw/test/', type=str)
    args = parser.parse_args()
    return args



# Model load
from model import resnet
from model.utils import load_filtered_state_dict, SaveBestModel, AverageMeter, accuracy
model = resnet.ResNet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 2)
checkpoint = torch.load(args.classifier_model)
load_filtered_state_dict(model, checkpoint, ignore_layer=[], reverse=True)
transform_test = transforms.Compose([transforms.Resize(320),
        transforms.RandomCrop(299), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

good_img_transform = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

device = 'cuda' if torch.cuda.is_available() else 'cpu'

classes = {0:'cyberbullying', 1:'non_cyberbullying'}

test_path = args.test_data_dir
testset = torchvision.datasets.ImageFolder(test_path, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=16, shuffle=True, num_workers=2)

model.to(device)
model.eval()



invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),
                               ])
                               




from skimage import segmentation
from pytorch_grad_cam import XGradCAM, GradCAM, FullGrad, GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io


def get_grayscale_grad_cam(image):
    input_tensor = image.to(device)
    targets = [ClassifierOutputTarget(0)]
    #target_layers = [model.layer4[-1]]
    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    
    return(grayscale_cam)
    
    def segmentation_info(image, num_segments, compactness):
    img_np = image.detach().cpu().squeeze().permute(1, 2, 0).numpy()
    segments_slic = slic(img_np, n_segments = num_segments, compactness=compactness,
                     start_label=1)
    num_segments = len(np.unique(segments_slic))
    list_unique_regions = np.unique(segments_slic)
    segment_pixel_num_list = []
    total_pixels = 0
    for i in (list_unique_regions):
        num_pixels = np.count_nonzero(segments_slic == i)
        segment_pixel_num_list.append(num_pixels)
        total_pixels += num_pixels
    
    
    information_for_each_segment = []
    for i in (list_unique_regions):
        image_list = []
        image_list.append(i)
        image_list.append(segment_pixel_num_list[i-1])
        image_list.append(total_pixels)
        information_for_each_segment.append(image_list)

    return(information_for_each_segment, segments_slic, num_segments)


# I want to get the average attribution score for each segment
def cam_processor_for_segments(grayscale_cam_output, segments_slic):
    
    
    
    list_unique_regions = np.unique(segments_slic)
    region_attr_score = []
    final_region_attr_score = []
    num_pixels_in_region_list = []
    
    for i in (list_unique_regions):
        row_counter = 0
        column_counter = 0
        region_attr_score = []
        num_pixels_in_region = 0
        for row in grayscale_cam_output:
            for cell in row:
                current_score = grayscale_cam_output[row_counter, column_counter]
                current_region = segments_slic[row_counter, column_counter]
                if current_region == i:
                    region_attr_score.append(current_score)
                    num_pixels_in_region += 1
                column_counter +=1
            row_counter += 1
            column_counter = 0
        avg_score = np.mean(region_attr_score)
        final_region_attr_score.append(avg_score)
        num_pixels_in_region_list.append(num_pixels_in_region)
    
    unique_region_info = []
    for i in (list_unique_regions):
        image_list = []
        image_list.append(i)
        image_list.append(final_region_attr_score[i-1])
        image_list.append(num_pixels_in_region_list[i-1])
        image_list.append(np.sum(num_pixels_in_region_list))
        unique_region_info.append(image_list)
    
    return(unique_region_info)
    
    
def get_feature_masks(image, attributions, segments_slic):
    segments_slic_1 = segments_slic
    features = []
    for i in attributions:
        feature = np.where(i==segments_slic_1, 1, 0)
        features.append(feature)
        
    return(features)


def attribution_ranker(cam_processor_for_segments_output, num_top_attr):
    ranked_images = sorted(cam_processor_for_segments_output, key=itemgetter(1), reverse=True)
    top_ranked_features = []
    for i in range(num_top_attr):
        top_ranked_features.append(ranked_images[i][0])
        
    return top_ranked_features
    
    def get_image_versions(image, features_list, model, SMU_class_index):
    image_versions = []
    num_pixels_changed = []
    total_attr_list = []

    powerset_list = list(more_itertools.powerset(features_list))
    powerset_list = [list(ele) for ele in powerset_list]
    num_versions = len(powerset_list)
    
    original_image = invTrans(image)
    image_versions.append(original_image)
    num_pixels_changed.append(0)
    total_attr_list.append(np.zeros((299, 299)))
    
    for version in range(num_versions - 1):
        obfuscated_image = image
        total_attribution = np.zeros((299, 299))
        total_num_pixels = total_attribution.size
        for mask in range(len(powerset_list[version + 1])):
            total_attribution += powerset_list[version + 1][mask]
            #print(np.max(powerset_list[version + 1][mask]))
        #print(np.max(total_attribution))
        num_changes = np.count_nonzero(total_attribution)
        num_pixels_changed.append(num_changes)
        total_attr_list.append(total_attribution)
        #print(num_changes)
        obfuscated_image = blur_image_from_attribution(image = obfuscated_image,
                                                       attribution_map = total_attribution)
        obfuscated_image = obfuscated_image.to(device)
        obfuscated_image = invTrans(obfuscated_image)
        
        image_versions.append(obfuscated_image)
    
    scores = []
    for i in range(num_versions):
        current_image = image_versions[i].to(device)
        #current_image = good_img_transform(current_image).to(device)
        score = SMU_cost_function(num_total_pixels = total_num_pixels,
                                  num_obf_pixels = num_pixels_changed[i],
                                  model = model,
                                  image = current_image,
                                  SMU_class_index = SMU_class_index)
        scores.append(score)
    
    
    
    unique_image_info = []
    for i in range(num_versions):
        image_list = []
        image_list.append(image_versions[i])
        image_list.append(num_pixels_changed[i])
        image_list.append(total_num_pixels)
        image_list.append(scores[i])
        image_list.append(total_attr_list[i])
        unique_image_info.append(image_list)
    
    
    return(unique_image_info)


def image_rankings(get_image_versions):
    #for idx in iterative_Grad_CAM_counterfactual_masking_output
    ranked_images = sorted(get_image_versions, key=itemgetter(3))
    
    return ranked_images


def blur_image_from_attribution(image, attribution_map):
    # attribution map is the attributions after being passed through the attribution processor
    # image is a tensor
    # will output the blurred image based on the attribution map
    
    
    #average_img = image.squeeze().cpu().permute(1, 2, 0).numpy()
    #avg = np.average(average_img)
    #blurred_img = cv2.GaussianBlur(image.squeeze().cpu().permute(1, 2, 0).numpy(), (181, 181), 0)
    avg = np.float32(-2.1179039478302)
    #avg_img = np.where(average_img > 9999, average_img, avg)
    
    #attribution_map = attribution_map.detach().squeeze().cpu().numpy()
    
    mask = [attribution_map, attribution_map, attribution_map]
    mask = np.array(mask)
    mask = mask.transpose(1,2,0)
    
    out = np.where(mask==np.array([0, 0, 0]), image.squeeze().cpu().permute(1, 2, 0).numpy(), avg)
    #out = np.where(mask==np.array([0, 0, 0]), image.squeeze().cpu().permute(1, 2, 0).numpy(), blurred_img)
    
    out = torch.tensor(out)
    out = out.permute(2,0,1)
    out = out.unsqueeze(0)
    
    return out

def SMU_cost_function(num_total_pixels, num_obf_pixels, model, image, SMU_class_index):
    image = good_img_transform(image)
    logits = model(image).cpu()
    probs = F.softmax(logits, dim=1)
    probs = probs.detach().cpu()
    probs = probs.tolist()[0]
    probs = probs[SMU_class_index]
    
    if num_obf_pixels == 0:
        score = 1000
    
    else:
        # Punish for prob too far from 0.5. We want to obfuscate just enough to change model prediction
        score = probs + (num_obf_pixels / num_total_pixels)
    
    if probs > 0.50:
        score += 1
    if probs > 0.75:
        score += 1
    if probs > 0.80:
        score += 10**(1+probs)
    #print(probs)
    return score

def full_obfuscation_function(image, num_segments, num_top_attr, compactness, model, SMU_class_index):
    example_10 = get_grayscale_grad_cam(image = image)
    seg = segmentation_info(image = image, num_segments = num_segments, compactness = compactness)
    avg_attr_scores = cam_processor_for_segments(grayscale_cam_output = example_10, segments_slic = seg[1])
    top_attrs = attribution_ranker(cam_processor_for_segments_output = avg_attr_scores, num_top_attr = num_top_attr)
    features_1 = get_feature_masks(image = image, attributions = top_attrs, segments_slic = seg[1])
    ex_1 = get_image_versions(image = image, features_list = features_1, model = model, SMU_class_index = SMU_class_index)
    ranked = image_rankings(get_image_versions = ex_1)
    
    return ranked




# To test our algorithm with no region dilation (8 regions)
i = 0
n = 0
image_info_list = []
while i < 150:
    torch.manual_seed(0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True, num_workers=2)
    images, labels = next(itertools.islice(testloader, n, None))
    just_label = labels.item()
    
    outputs = model(images.to(device))
    _, predicted = outputs.max(1)
    predicted = predicted.cpu().item()
    #n += 1
    #print()
    #print(predicted)
    if just_label == 0 and predicted ==0:
        print('index:', n)
        i += 1
        
        example_10 = get_grayscale_grad_cam(image = images)
        seg = segmentation_info(image = images, num_segments = 30, compactness = 50)
        avg_attr_scores = cam_processor_for_segments(grayscale_cam_output = example_10, segments_slic = seg[1])
        top_attrs = attribution_ranker(cam_processor_for_segments_output = avg_attr_scores, num_top_attr = 8)
        features_1 = get_feature_masks(image = images, attributions = top_attrs, segments_slic = seg[1])
        ex_1 = get_image_versions(image = images, features_list = features_1, model = model, SMU_class_index = 0)
        ranked = image_rankings(get_image_versions = ex_1)
        
        
        image_info = []
        
        example = ranked[0][0]
        exam_img = good_img_transform(example)
        logits = model(exam_img).cpu()
        probs = F.softmax(logits, dim=1)
        probs = probs.detach().cpu()
        probs = probs.tolist()[0]
        # Change probs[int] to int = SMU class index
        probs = probs[0]
        #print(probs)
        image_info.append(probs)
        
        num_pixels_obf = ranked[0][1]
        image_info.append(num_pixels_obf)
        
        image_info_list.append(image_info)
    n += 1


success = 0
total = len(image_info_list)
total_pix = total * 299 * 299
total_obf = 0
for i in range(len(image_info_list)):
    if image_info_list[i][0] < 0.5:
        success += 1
    total_obf += image_info_list[i][1] 

print("Successful obfuscation proportion", success / total)
print("Average obfuscation proportion", total_obf / total_pix)




