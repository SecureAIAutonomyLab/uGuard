cd ../auxiliary_models/cyberbullying-pytorch/

python src/train.py --saved_model models/cyberbullying/epoch_54.pkl --trainning_data_dir ../../datasets/cyberbullying/train/ --batch_size 1 --extract_distributions True --gpu 0 --save_dist_path '../../distributions_clean_model/cyberbullying/'


