#conda activate '../../../../cyberbullying_purification/uGuard/env'
#pip install torchvision==0.12.0
import numpy as np
import json, os, warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils
from torchvision import models
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torchattacks
from scipy.ndimage import gaussian_filter
import cv2
from PIL import Image
from torchattacks import *
import itertools
import argparse


class adversarial_attack:
    '''
    Launch adversarial attacks on images in input path and save attacked images to output path.
    
    Parameters: 
        input_path: path of input images to be attacked
        output_path: path where output images will be stored
    Raises:
        AttackExceptions
    '''
    
    def __init__(self, input_path: str, output_path: str, eps: float = 32/255, use_cuda: bool = True) -> None:
        '''
        Launch adversarial attacks on images in input path and save attacked images to output path.

        Parameters: 
            input_path: path of input images to be attacked
            output_path: path where output images will be stored
            eps: epsilon for the adversarial attack strength. Default is 32/255
            use_cuda: only enable if the system uses cuda.
            
        Raises:
            AttackExceptions
        '''
        def image_folder_custom_label(root, transform, idx2label) :
            # custom_label
            # type : List
            # index -> label
            # ex) ['tench', 'goldfish', 'great_white_shark', 'tiger_shark']
            old_data = dsets.ImageFolder(root=root, transform=transform)
            old_classes = old_data.classes
            label2idx = {}
            for i, item in enumerate(idx2label) :
                label2idx[item] = i
            new_data = dsets.ImageFolder(root=root, transform=transform, 
                                         target_transform=lambda x : idx2label.index(old_classes[x]))
            new_data.classes = idx2label
            new_data.class_to_idx = label2idx
            return new_data
        # load Inception V3 model
        class Normalize(nn.Module):
            def __init__(self, mean, std) :
                super(Normalize, self).__init__()
                self.register_buffer('mean', torch.Tensor(mean))
                self.register_buffer('std', torch.Tensor(std))
            
            def forward(self, input):
                # Broadcasting
                mean = self.mean.reshape(1, 3, 1, 1)
                std = self.std.reshape(1, 3, 1, 1)
                return (input - mean) / std
        
        class_idx = json.load(open("data/imagenet_class_index.json"))
        idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
        transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(), # ToTensor : [0, 255] -> [0, 1]
        ])
        imagnet_data = image_folder_custom_label(root='data/imagenet', transform=transform, idx2label=idx2label)
        data_loader = torch.utils.data.DataLoader(imagnet_data, batch_size=1, shuffle=False)    
        images, labels = iter(data_loader).next()
            
        device = torch.device("cuda" if use_cuda else "cpu")
        norm_layer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        model = nn.Sequential(
            norm_layer,
            models.resnet18(pretrained=True)
        ).to(device)
        model = model.eval()
        # launch adversarial attacks
        if not isinstance(input_path, str) or not input_path:
            raise AttackException(
                f'Input path is not valid.')
        atks = [
            FGSM(model, eps=eps),
            PGD(model, eps=eps, alpha=2/225, steps=100, random_start=True),
            VANILA(model),
            Square(model, eps=eps, n_queries=10000, n_restarts=1, loss='ce'),
            AutoAttack(model, eps=eps, n_classes=10, version='standard'),
            DeepFool(model, steps=500),
        ]
        

        input_files = os.listdir(input_path)
        input_files = [file_name for file_name in input_files if (file_name.endswith('.jpg') or file_name.endswith('.jpeg') or file_name.endswith('.png'))]
        if len(input_files) == 0:
            raise AttackException(
                f'There is no valid file to attack in the input path.')
            
        print("Adversarial Image & Predicted Label")
        for atk in atks :
            print("-"*70)
            print(atk)
            # save attack name. if path for saving attacked image does not exist, make it
            atk_name = str(atk).split('(')[0]
            save_path = os.path.join(output_path, atk_name)
            pert_path = os.path.join(output_path, 'perturbation', atk_name)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            if not os.path.exists(pert_path):
                os.makedirs(pert_path)
            # iterate through images
            for img_name in input_files:
                # load the image
                image = Image.open(os.path.join(input_path, img_name))
                x = TF.to_tensor(image)
                x.unsqueeze_(0)
                # put image to device
                x = x.to(device)
                # get correct label for image
                labels[0] = int(torch.topk(model(x), k = 1)[1][0][0])
                # attack the image
                adv_image = atk(x, labels)
                labels = labels.to(device)
                outputs = model(adv_image)
                # isolate perturbation
                perturbation = adv_image - x
                # save perturbation as png
                torchvision.utils.save_image(perturbation, os.path.join(pert_path, img_name.split('.')[0] + '_pert.png'))
                # save the attacked image
                adv_path = os.path.join(save_path, img_name.split('.')[0] + '.png')
                torchvision.utils.save_image(adv_image, adv_path)
                # gaussian blur
                attacked_img = cv2.imread(adv_path)
                ensemble_img = cv2.GaussianBlur(attacked_img, (5, 5), cv2.BORDER_DEFAULT)    
                cv2.imwrite(os.path.join(output_path, atk_name + '_GB', img_name.split('.')[0]) + '.png', ensemble_img)
        # ensemble attacks
        atk_names = [str(name).split('(')[0] for name in atks]
        for pair in combinations(atks, 2):
            ensemble_path = os.path.join(output_path, pair[0] + '_' + pair[1])
            if not os.path.exists(ensemble_path):
                os.makedirs(ensemble_path)
            for img_name in input_files:
                clean_image = TF.to_tensor(Image.open(os.path.join(input_path, img_name)))
                p1_path = os.path.join(output_path, 'perturbation', pair[0], img_name.split('.')[0] + '_pert.png')
                p2_path = os.path.join(output_path, 'perturbation', pair[1], img_name.split('.')[0] + '_pert.png')
                p1 = TF.to_tensor(Image.open(p1_path))
                p2 = TF.to_tensor(Image.open(p2_path))
                ensemble_img = clean_img + (p1+p2)
                torchvision.utils.save_image(ensemble_img, os.path.join(ensemble_path, img_name.split('.')[0] + '.png'))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example script with input and output paths.")
    parser.add_argument("input_path", help="Path to the input file.")
    parser.add_argument("output_path", help="Path to the output file.")
    args = parser.parse_args()
    adversarial_attack(args.input_path, args.output_path)
