#!/bin/bash

cd ../src_files

python purify_attacked_images.py --eval_data_dir ../datasets/evaluation_data/cyberbullying_attacked/comb --distribution_files ../distributions_clean_model/cyberbullying/ --dictionary_files ../clean_dictionaries/cyberbullying/ --robust_model ../datasets/evaluation_data/robust_CB.pth --save_path ../purified_images/evaluations/cyberbullying_attacked/comb
