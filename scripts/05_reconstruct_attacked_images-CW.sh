#!/bin/bash

cd ../src_files

python purify_attacked_images-NSFW.py --eval_data_dir ../datasets/evaluation_data/NSFW_attacked/CW --distribution_files ../distributions_clean_model/NSFW/ --dictionary_files ../clean_dictionaries/NSFW/ --robust_model ../datasets/evaluation_data/robust_NSFW.pth --save_path ../purified_images/evaluations/NSFW_attacked/CW
