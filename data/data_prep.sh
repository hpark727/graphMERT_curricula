#! /bin/bash

conda activate bottom_up_SI

python3 decontamination.py --train_questions_path /curriculum_training_data/curriculum_dataset_hop_3.json --ngram_size 18

python3 tokenization.py --dataset_train_path /curriculum_training_data/curriculum_dataset_hop_3_decontaminated.json 
