#!/bin/bash

cd ../auxiliary_models/self_harm-pytorch/

python src/train_robust.py --saved_model models/self_harm/epoch_13.pkl --trainning_data_dir ../../datasets/self_harm_attacked_2/train/ --validation_data_dir ../../datasets/self_harm_attacked_2/test/ --save_path models/self_harm_robust_2 --safe_clean_distributions ../../distributions_clean_model/self_harm/safe_samples_dist.npy --unsafe_clean_distributions ../../distributions_clean_model/self_harm/unsafe_samples_dist.npy --extract_distributions None --gpu 0