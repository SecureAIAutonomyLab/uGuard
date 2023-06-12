#!/bin/bash

cd ../auxiliary_models/nsfw-pytorch/

python src/train.py --saved_model ../nsfw_Working_Model/models/nsfw-clean/ResNet_NSFW_with_feature_output.pth --trainning_data_dir ../../datasets/nsfw/train_balanced/ --batch_size 1 --extract_distributions True --gpu 0 --save_dist_path '../../distributions_clean_model/nsfw/'

