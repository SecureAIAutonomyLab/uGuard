#!/bin/bash

cd ../auxiliary_models/self_harm-pytorch/

python src/train.py --saved_model models/self_harm/epoch_42.pkl --trainning_data_dir ../../datasets/self_harm/train/ --validation_data_dir ../../datasets/self_harm/val/ --save_path models/self_harm --extract_distributions None --gpu 0

