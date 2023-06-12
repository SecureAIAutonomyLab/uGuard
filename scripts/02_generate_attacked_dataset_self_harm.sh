#!/bin/bash

cd ../auxiliary_models/self_harm-pytorch/

python src/generate_attacked_dataset.py --saved_model ../../non_robust_classifiers/ResNet_SH_27.pth --trainning_data_dir ../../datasets/self_harm/train/ --validation_data_dir ../../datasets/self_harm/train/ --save_path models/self_harm --extract_distributions None --gpu 0


