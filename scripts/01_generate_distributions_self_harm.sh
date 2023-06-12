cd ../auxiliary_models/self_harm-pytorch/

python src/train.py --saved_model models/self_harm/epoch_42.pkl --trainning_data_dir ../../datasets/self_harm/train/ --batch_size 1 --extract_distributions True --gpu 0 --save_dist_path '../../distributions_clean_model/self_harm/'


