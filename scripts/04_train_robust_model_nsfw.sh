#!/bin/bash

cd ../auxiliary_models/nsfw_Working_Model/

python src/train_robust.py --saved_model models/nsfw-clean/ResNet_NSFW_with_feature_output.pth --trainning_data_dir ../../datasets/nsfw_attacked_simple/train/ --validation_data_dir ../../datasets/nsfw_attacked_simple/test/ --save_path models/nsfw_robust_2 --safe_clean_distributions ../../distributions_clean_model/NSFW/safe_samples_dist.npy --unsafe_clean_distributions ../../distributions_clean_model/NSFW/unsafe_samples_dist.npy --extract_distributions None --gpu 0  