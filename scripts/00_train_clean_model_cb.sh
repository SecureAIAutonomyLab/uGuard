cd ../auxiliary_models/cyberbullying-pytorch/

python src/train.py --saved_model models/cyberbullying/epoch_54.pkl --trainning_data_dir ../../datasets/cyberbullying/train/ --validation_data_dir ../../datasets/cyberbullying/val/ --save_path models/cyberbullying --extract_distributions None --gpu 0

