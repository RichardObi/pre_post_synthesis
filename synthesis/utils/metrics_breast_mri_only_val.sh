#! /bin/bash


echo "1. Creating and activating virtual environment called MMG_env."
python3 -m venv MMG_env
source MMG_env/bin/activate

echo "2. METRIC computation on VALIDATION data."

echo "======================== REAL-REAL Comparisons ========================"

echo "postcontrast real - postcontrast real (both normalized)"
python3 metrics.py train_test/validation/validation_B train_test/validation/validation_B --normalize_images

echo "precontrast real - postcontrast real  normalized"
python3 metrics.py train_test/validation/validation_A train_test/validation/validation_B --normalize_images

echo "======================== REAL-SYNTHETIC Comparisons ========================"

echo "EPOCH 10: postcontrast synthetic - postcontrast real  normalized"
python3 metrics.py pix2pixHD/results/pre2postcontrast_512p_train_1to195/validation_10/only_synthetic_val train_test/validation/validation_B --normalize_images

echo "EPOCH 20: postcontrast synthetic - postcontrast real  normalized"
python3 metrics.py pix2pixHD/results/pre2postcontrast_512p_train_1to195/validation_20/only_synthetic_val train_test/validation/validation_B --normalize_images

echo "EPOCH 30: postcontrast synthetic - postcontrast real  normalized"
python3 metrics.py pix2pixHD/results/pre2postcontrast_512p_train_1to195/validation_30/only_synthetic_val train_test/validation/validation_B --normalize_images

echo "EPOCH 40: postcontrast synthetic - postcontrast real  normalized"
python3 metrics.py pix2pixHD/results/pre2postcontrast_512p_train_1to195/validation_40/only_synthetic_val train_test/validation/validation_B --normalize_images

echo "EPOCH 50: postcontrast synthetic - postcontrast real  normalized"
python3 metrics.py pix2pixHD/results/pre2postcontrast_512p_train_1to195/validation_50/only_synthetic_val train_test/validation/validation_B --normalize_images

echo "EPOCH 60: postcontrast synthetic - postcontrast real  normalized"
python3 metrics.py pix2pixHD/results/pre2postcontrast_512p_train_1to195/validation_60/only_synthetic_val train_test/validation/validation_B --normalize_images

echo "EPOCH 70: postcontrast synthetic - postcontrast real  normalized"
python3 metrics.py pix2pixHD/results/pre2postcontrast_512p_train_1to195/validation_70/only_synthetic_val train_test/validation/validation_B --normalize_images

echo "EPOCH 80: postcontrast synthetic - postcontrast real  normalized"
python3 metrics.py pix2pixHD/results/pre2postcontrast_512p_train_1to195/validation_80/only_synthetic_val train_test/validation/validation_B --normalize_images

echo "EPOCH 90: postcontrast synthetic - postcontrast real  normalized"
python3 metrics.py pix2pixHD/results/pre2postcontrast_512p_train_1to195/validation_90/only_synthetic_val train_test/validation/validation_B --normalize_images

echo "EPOCH 100: postcontrast synthetic - postcontrast real  normalized"
python3 metrics.py pix2pixHD/results/pre2postcontrast_512p_train_1to195/validation_100/only_synthetic_val train_test/validation/validation_B --normalize_images

echo "EPOCH 110: postcontrast synthetic - postcontrast real  normalized"
python3 metrics.py pix2pixHD/results/pre2postcontrast_512p_train_1to195/validation_110/only_synthetic_val train_test/validation/validation_B --normalize_images

echo "EPOCH 120: postcontrast synthetic - postcontrast real  normalized"
python3 metrics.py pix2pixHD/results/pre2postcontrast_512p_train_1to195/validation_120/only_synthetic_val train_test/validation/validation_B --normalize_images

echo "EPOCH 170: postcontrast synthetic - postcontrast real  normalized"
python3 metrics.py pix2pixHD/results/pre2postcontrast_512p_train_1to195/validation_170/only_synthetic_val train_test/validation/validation_B --normalize_images

echo "FINISHED CALCULATING METRICS"
