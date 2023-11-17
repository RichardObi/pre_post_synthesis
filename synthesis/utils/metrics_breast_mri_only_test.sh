#! /bin/bash


echo "1. Creating and activating virtual environment called MMG_env."
python3 -m venv MMG_env
source MMG_env/bin/activate

echo "2. METRIC computation on TEST data."


echo "======================== REAL-REAL Comparisons ========================"

#echo "postcontrast real - postcontrast real (both normalized)"
#python3 metrics.py train_test/test/test_B train_test/test/test_B --normalize_images

echo "precontrast real - postcontrast real  normalized"
python3 metrics.py train_test/test/test_A train_test/test/test_B --normalize_images

echo "======================== REAL-SYNTHETIC Comparisons ========================"

echo "EPOCH 30: postcontrast synthetic - postcontrast real normalized"
python3 metrics.py pix2pixHD/results/pre2postcontrast_512p_train_1to195/test_30/only_synthetic train_test/test/test_B --normalize_images


echo "FINISHED CALCULATING METRICS"
                                                        
