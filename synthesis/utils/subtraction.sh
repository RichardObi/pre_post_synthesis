#! /bin/bash

echo "Now subtracting pre-contrast (train_test/test_A) from post-contrast (train_test/test_B)"
python3 subtraction.py --dataset_path_1 train_test/test/test_B --dataset_path_2 train_test/test/test_A --output_folder train_test/test/subtraction_real

echo "Now subtracting pre-contrast (train_test/test_A) from post-contrast (/home/roo/Desktop/software/breastmri/src/synthesis/pix2pixHD/results/pre2postcontrast_512p_train_1to195/test_30/only_synthetic)"
python3 subtraction.py --dataset_path_1 pix2pixHD/results/pre2postcontrast_512p_train_1to195/test_30/only_synthetic --dataset_path_2 train_test/test/test_A --output_folder train_test/test/subtraction_synthetic

echo "FINISHED SUBTRACTION IMAGE GENERATION. OUTPUT=train_test/test/subtraction_real or ../subtraction_synthetic"
