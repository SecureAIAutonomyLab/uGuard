#!/bin/bash

cd ../auxiliary_models/cyberbullying-pytorch/

python src/train_robust.py --saved_model models/cyberbullying/epoch_8.pkl --trainning_data_dir ../../datasets/cyberbullying_attacked_2/train/ --validation_data_dir ../../datasets/cyberbullying_attacked_2/test/ --save_path models/cyberbullying_robust_2 --safe_clean_distributions ../../distributions_clean_model/cyberbullying/safe_samples_dist.npy --unsafe_clean_distributions ../../distributions_clean_model/cyberbullying/unsafe_samples_dist.npy --extract_distributions None --gpu 0
