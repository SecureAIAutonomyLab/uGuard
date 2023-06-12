#!/bin/bash

cd ../src_files

python purify_attacked_images.py --eval_data_dir ../datasets/evaluation_data/self_harm_attacked/comb --distribution_files ../distributions_clean_model/self_harm/ --dictionary_files ../clean_dictionaries/self_harm/ --robust_model ../datasets/evaluation_data/robust_SH.pth --save_path ../purified_images/evaluations/self_harm_attacked/comb
