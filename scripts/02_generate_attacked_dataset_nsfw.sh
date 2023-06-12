#!/bin/bash

cd ../auxiliary_models/nsfw_Working_Model/

python src/generate_attacked_dataset.py --saved_model models/nsfw-clean/ResNet_NSFW_with_feature_output.pth --trainning_data_dir ../../datasets/nsfw/train_balanced/ --validation_data_dir ../../datasets/nsfw/val_balanced/ --save_path models/nsfw_robust --extract_distributions None --gpu 0