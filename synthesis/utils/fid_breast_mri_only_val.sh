#!/bin/bash

#Preliminaries


echo "1. Creating and activating virtual environment called MMG_env."
python3 -m venv MMG_env
source MMG_env/bin/activate

echo "2. FID computation on VALIDATION DATASET."

echo "====================== NOW CALCULATING FID with Dataset Split ===================="

echo "====================== IMAGENET with Dataset Split: ======================"

echo "postcontrast real - postcontrast real imagenet normalized"
python3 fid.py train_test/validation/validation_B train_test/validation/validation_B --normalize_images  --model imagenet --is_split_per_patient --reverse_split_ds2 --is_only_splitted_loaded --description postcontrast_real-vs-postcontrast_real_imagenet_normalized_SPLIT1_vs_SPLIT2

echo "EPOCH 30: postcontrast synthetic - postcontrast real imagenet normalized - SPLIT 1"
python3 fid.py pix2pixHD/results/pre2postcontrast_512p_train_1to195/validation_30/only_synthetic_val train_test/validation/validation_B --normalize_images  --model imagenet --is_split_per_patient  --reverse_split_ds1 --is_only_splitted_loaded --description EPOCH30-postcontrast_synthetic-vs-postcontrast_real_imagenet_normalized-SPLIT1

echo "EPOCH 30: postcontrast synthetic - postcontrast real imagenet normalized - SPLIT 2"
python3 fid.py pix2pixHD/results/pre2postcontrast_512p_train_1to195/validation_30/only_synthetic_val train_test/validation/validation_B --normalize_images  --model imagenet --is_split_per_patient  --reverse_split_ds2 --is_only_splitted_loaded --description EPOCH30-postcontrast_synthetic-vs-postcontrast_real_imagenet_normalized-SPLIT2

echo "====================== RADIMAGENET with Dataset Split: ======================"

echo "postcontrast real - postcontrast real radimagenet normalized"
python3 fid.py train_test/validation/validation_B train_test/validation/validation_B --normalize_images  --model radimagenet --is_split_per_patient --reverse_split_ds2 --is_only_splitted_loaded --description postcontrast_real-vs-postcontrast_real_radimagenet_normalized_SPLIT1_vs_SPLIT2

echo "EPOCH 30: postcontrast synthetic - postcontrast real radimagenet normalized - SPLIT 1"
python3 fid.py pix2pixHD/results/pre2postcontrast_512p_train_1to195/validation_30/only_synthetic_val train_test/validation/validation_B --normalize_images  --model radimagenet --is_split_per_patient --reverse_split_ds1 --is_only_splitted_loaded --description EPOCH30-postcontrast_synthetic-vs-postcontrast_real_radimagenet_normalized-SPLIT1

echo "EPOCH 30: postcontrast synthetic - postcontrast real radimagenet normalized - SPLIT 2"
python3 fid.py pix2pixHD/results/pre2postcontrast_512p_train_1to195/validation_30/only_synthetic_val train_test/validation/validation_B --normalize_images  --model radimagenet --is_split_per_patient --reverse_split_ds2 --is_only_splitted_loaded --description EPOCH30-postcontrast_synthetic-vs-postcontrast_real_imagenet_normalized-SPLIT2

echo "FINISHED CALCULATING FID with Dataset Split"



echo " ====================== NOW CALCULATING Standard FID without Dataset Split ===================="

echo "==================== Baselines IMAGENET: ===================="

echo "postcontrast real - postcontrast real imagenet normalized"
python3 fid.py train_test/validation/validation_B train_test/validation/validation_B --normalize_images --model imagenet --description postcontrast_real-vs-postcontrast_real_imagenet_normalized_same_cases_compared_wo_split

echo "precontrast real - postcontrast real radimagenet normalized"
python3 fid.py train_test/validation/validation_A train_test/validation/validation_B --normalize_images  --model imagenet --description precontrast_real-vs-postcontrast_real_imagenet_normalized_same_cases_compared_wo_split

echo "==================== Baselines RADIMAGENET: ===================="

echo "postcontrast real - postcontrast real radimagenet normalized"
python3 fid.py train_test/validation/validation_B train_test/validation/validation_B --normalize_images --model radimagenet --description postcontrast_real-vs-postcontrast_real_radimagenet_normalized_same_cases_compared_wo_split

echo "precontrast real - postcontrast real radimagenet normalized"
python3 fid.py train_test/validation/validation_A train_test/validation/validation_B --normalize_images  --model radimagenet --description precontrast_real-vs-postcontrast_real_radimagenet_normalized_same_cases_compared_wo_split



echo "==================== IMAGENET: ===================="

echo "EPOCH 10: postcontrast synthetic - postcontrast real imagenet normalized"
python3 fid.py pix2pixHD/results/pre2postcontrast_512p_train_1to195/validation_10/only_synthetic_val train_test/validation/validation_B --normalize_images  --model imagenet --description EPOCH10-postcontrast_synthetic-vs-postcontrast_real_imagenet_normalized-same_cases_compared_wo_split

echo "EPOCH 20: postcontrast synthetic - postcontrast real imagenet normalized"
python3 fid.py pix2pixHD/results/pre2postcontrast_512p_train_1to195/validation_20/only_synthetic_val train_test/validation/validation_B --normalize_images  --model imagenet --description EPOCH20-postcontrast_synthetic-vs-postcontrast_real_imagenet_normalized-same_cases_compared_wo_split

echo "EPOCH 30: postcontrast synthetic - postcontrast real imagenet normalized"
python3 fid.py pix2pixHD/results/pre2postcontrast_512p_train_1to195/validation_30/only_synthetic_val train_test/validation/validation_B --normalize_images  --model imagenet --description EPOCH30-postcontrast_synthetic-vs-postcontrast_real_imagenet_normalized-same_cases_compared_wo_split

echo "EPOCH 40: postcontrast synthetic - postcontrast real imagenet normalized"
python3 fid.py pix2pixHD/results/pre2postcontrast_512p_train_1to195/validation_40/only_synthetic_val train_test/validation/validation_B --normalize_images  --model imagenet --description EPOCH40-postcontrast_synthetic-vs-postcontrast_real_imagenet_normalized-same_cases_compared_wo_split

echo "EPOCH 50: postcontrast synthetic - postcontrast real imagenet normalized"
python3 fid.py pix2pixHD/results/pre2postcontrast_512p_train_1to195/validation_50/only_synthetic_val train_test/validation/validation_B --normalize_images  --model imagenet --description EPOCH50-postcontrast_synthetic-vs-postcontrast_real_imagenet_normalized-same_cases_compared_wo_split

echo "EPOCH 60: postcontrast synthetic - postcontrast real imagenet normalized"
python3 fid.py pix2pixHD/results/pre2postcontrast_512p_train_1to195/validation_60/only_synthetic_val train_test/validation/validation_B --normalize_images  --model imagenet --description EPOCH60-postcontrast_synthetic-vs-postcontrast_real_imagenet_normalized-same_cases_compared_wo_split

echo "EPOCH 70: postcontrast synthetic - postcontrast real imagenet normalized"
python3 fid.py pix2pixHD/results/pre2postcontrast_512p_train_1to195/validation_70/only_synthetic_val train_test/validation/validation_B --normalize_images  --model imagenet --description EPOCH70-postcontrast_synthetic-vs-postcontrast_real_imagenet_normalized-same_cases_compared_wo_split

echo "EPOCH 80: postcontrast synthetic - postcontrast real imagenet normalized"
python3 fid.py pix2pixHD/results/pre2postcontrast_512p_train_1to195/validation_80/only_synthetic_val train_test/validation/validation_B --normalize_images  --model imagenet --description EPOCH80-postcontrast_synthetic-vs-postcontrast_real_imagenet_normalized-same_cases_compared_wo_split

echo "EPOCH 90: postcontrast synthetic - postcontrast real imagenet normalized"
python3 fid.py pix2pixHD/results/pre2postcontrast_512p_train_1to195/validation_90/only_synthetic_val train_test/validation/validation_B --normalize_images  --model imagenet --description EPOCH90-postcontrast_synthetic-vs-postcontrast_real_imagenet_normalized-same_cases_compared_wo_split

echo "EPOCH 100: postcontrast synthetic - postcontrast real imagenet normalized"
python3 fid.py pix2pixHD/results/pre2postcontrast_512p_train_1to195/validation_100/only_synthetic_val train_test/validation/validation_B --normalize_images  --model imagenet --description EPOCH100-postcontrast_synthetic-vs-postcontrast_real_imagenet_normalized-same_cases_compared_wo_split

echo "EPOCH 110: postcontrast synthetic - postcontrast real imagenet normalized"
python3 fid.py pix2pixHD/results/pre2postcontrast_512p_train_1to195/validation_110/only_synthetic_val train_test/validation/validation_B --normalize_images  --model imagenet --description EPOCH120-postcontrast_synthetic-vs-postcontrast_real_imagenet_normalized-same_cases_compared_wo_split

echo "EPOCH 120: postcontrast synthetic - postcontrast real imagenet normalized"
python3 fid.py pix2pixHD/results/pre2postcontrast_512p_train_1to195/validation_120/only_synthetic_val train_test/validation/validation_B --normalize_images  --model imagenet --description EPOCH130-postcontrast_synthetic-vs-postcontrast_real_imagenet_normalized-same_cases_compared_wo_split

echo "EPOCH 170: postcontrast synthetic - postcontrast real imagenet normalized"
python3 fid.py pix2pixHD/results/pre2postcontrast_512p_train_1to195/validation_170/only_synthetic_val train_test/validation/validation_B --normalize_images  --model imagenet --description EPOCH170-postcontrast_synthetic-vs-postcontrast_real_imagenet_normalized-same_cases_compared_wo_split


echo "==================== RADIMAGENET: ===================="

echo "EPOCH 10: postcontrast synthetic - postcontrast real radimagenet normalized"
python3 fid.py pix2pixHD/results/pre2postcontrast_512p_train_1to195/validation_10/only_synthetic_val train_test/validation/validation_B --normalize_images  --model radimagenet --description EPOCH10-postcontrast_synthetic-vs-postcontrast_real_radimagenet_normalized-same_cases_compared_wo_split

echo "EPOCH 20: postcontrast synthetic - postcontrast real radimagenet normalized"
python3 fid.py pix2pixHD/results/pre2postcontrast_512p_train_1to195/validation_20/only_synthetic_val train_test/validation/validation_B --normalize_images  --model radimagenet --description EPOCH20-postcontrast_synthetic-vs-postcontrast_real_radimagenet_normalized-same_cases_compared_wo_split

echo "EPOCH 30: postcontrast synthetic - postcontrast real radimagenet normalized"
python3 fid.py pix2pixHD/results/pre2postcontrast_512p_train_1to195/validation_30/only_synthetic_val train_test/validation/validation_B --normalize_images  --model radimagenet --description EPOCH30-postcontrast_synthetic-vs-postcontrast_real_radimagenet_normalized-same_cases_compared_wo_split

echo "EPOCH 40: postcontrast synthetic - postcontrast real radimagenet normalized"
python3 fid.py pix2pixHD/results/pre2postcontrast_512p_train_1to195/validation_40/only_synthetic_val train_test/validation/validation_B --normalize_images  --model radimagenet --description EPOCH40-postcontrast_synthetic-vs-postcontrast_real_radimagenet_normalized-same_cases_compared_wo_split

echo "EPOCH 50: postcontrast synthetic - postcontrast real radimagenet normalized"
python3 fid.py pix2pixHD/results/pre2postcontrast_512p_train_1to195/validation_50/only_synthetic_val train_test/validation/validation_B --normalize_images  --model radimagenet --description EPOCH50-postcontrast_synthetic-vs-postcontrast_real_radimagenet_normalized-same_cases_compared_wo_split

echo "EPOCH 60: postcontrast synthetic - postcontrast real radimagenet normalized"
python3 fid.py pix2pixHD/results/pre2postcontrast_512p_train_1to195/validation_60/only_synthetic_val train_test/validation/validation_B --normalize_images  --model radimagenet --description EPOCH60-postcontrast_synthetic-vs-postcontrast_real_radimagenet_normalized-same_cases_compared_wo_split

echo "EPOCH 70: postcontrast synthetic - postcontrast real radimagenet normalized"
python3 fid.py pix2pixHD/results/pre2postcontrast_512p_train_1to195/validation_70/only_synthetic_val train_test/validation/validation_B --normalize_images  --model radimagenet --description EPOCH70-postcontrast_synthetic-vs-postcontrast_real_radimagenet_normalized-same_cases_compared_wo_split

echo "EPOCH 80: postcontrast synthetic - postcontrast real radimagenet normalized"
python3 fid.py pix2pixHD/results/pre2postcontrast_512p_train_1to195/validation_80/only_synthetic_val train_test/validation/validation_B --normalize_images  --model radimagenet --description EPOCH80-postcontrast_synthetic-vs-postcontrast_real_radimagenet_normalized-same_cases_compared_wo_split

echo "EPOCH 90: postcontrast synthetic - postcontrast real radimagenet normalized"
python3 fid.py pix2pixHD/results/pre2postcontrast_512p_train_1to195/validation_90/only_synthetic_val train_test/validation/validation_B --normalize_images  --model radimagenet --description EPOCH90-postcontrast_synthetic-vs-postcontrast_real_radimagenet_normalized-same_cases_compared_wo_split

echo "EPOCH 100: postcontrast synthetic - postcontrast real radimagenet normalized"
python3 fid.py pix2pixHD/results/pre2postcontrast_512p_train_1to195/validation_100/only_synthetic_val train_test/validation/validation_B --normalize_images  --model radimagenet --description EPOCH100-postcontrast_synthetic-vs-postcontrast_real_radimagenet_normalized-same_cases_compared_wo_split

echo "EPOCH 110: postcontrast synthetic - postcontrast real radimagenet normalized"
python3 fid.py pix2pixHD/results/pre2postcontrast_512p_train_1to195/validation_110/only_synthetic_val train_test/validation/validation_B --normalize_images  --model radimagenet --description EPOCH120-postcontrast_synthetic-vs-postcontrast_real_radimagenet_normalized-same_cases_compared_wo_split

echo "EPOCH 120: postcontrast synthetic - postcontrast real radimagenet normalized"
python3 fid.py pix2pixHD/results/pre2postcontrast_512p_train_1to195/validation_120/only_synthetic_val train_test/validation/validation_B --normalize_images  --model radimagenet --description EPOCH130-postcontrast_synthetic-vs-postcontrast_real_radimagenet_normalized-same_cases_compared_wo_split

echo "EPOCH 170: postcontrast synthetic - postcontrast real radimagenet normalized"
python3 fid.py pix2pixHD/results/pre2postcontrast_512p_train_1to195/validation_170/only_synthetic_val train_test/validation/validation_B --normalize_images  --model radimagenet --description EPOCH170-postcontrast_synthetic-vs-postcontrast_real_radimagenet_normalized-same_cases_compared_wo_split

