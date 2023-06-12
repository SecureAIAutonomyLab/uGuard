#!/bin/bash

cd ../auxiliary_models/cyberbullying-pytorch/

python src/generate_attacked_dataset.py --saved_model ../../non_robust_classifiers/ResNet_CB_24.pth --trainning_data_dir ../../datasets/cyberbullying/train/ --validation_data_dir ../../datasets/cyberbullying/train/ --save_path models/cyberbullying --extract_distributions None --gpu 0


