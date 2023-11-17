
echo "1. Creating and activating virtual environment called MMG_env."
python3 -m venv MMG_env
source MMG_env/bin/activate

echo "2. FID computation on TEST DATASET: Start"

echo "====================== TEST DATA: NOW CALCULATING FID with Dataset Split ===================="

echo "====================== IMAGENET with Dataset Split: ======================"

#echo "postcontrast real - postcontrast real imagenet normalized"
#python3 fid.py train_test/test/test_B train_test/test/test_B --limit 2000 --normalize_images  --model imagenet --is_split_per_patient --reverse_split_ds2 --is_only_splitted_loaded --description TESTdata_postcontrast_real-vs-postcontrast_real_imagenet_normalized_SPLIT1_vs_SPLIT2

#echo "EPOCH 30: postcontrast synthetic - postcontrast real imagenet normalized - SPLIT 1"
#python3 fid.py pix2pixHD/results/pre2postcontrast_512p_train_1to195/test_30/only_synthetic train_test/test/test_B --limit 2000 --normalize_images  --model imagenet --is_split_per_patient  --reverse_split_ds1 --is_only_splitted_loaded --description TESTdata_EPOCH30-postcontrast_synthetic-vs-postcontrast_real_imagenet_normalized-SPLIT1


echo "====================== RADIMAGENET with Dataset Split: ======================"

#echo "postcontrast real - postcontrast real radimagenet normalized"
#python3 fid.py train_test/test/test_B train_test/test/test_B --limit 2000 --normalize_images  --model radimagenet --is_split_per_patient --reverse_split_ds2 --is_only_splitted_loaded --description TESTdata_postcontrast_real-vs-postcontrast_real_radimagenet_normalized_SPLIT1_vs_SPLIT2

#echo "EPOCH 30: postcontrast synthetic - postcontrast real radimagenet normalized - SPLIT 1"
#python3 fid.py pix2pixHD/results/pre2postcontrast_512p_train_1to195/test_30/only_synthetic train_test/test/test_B --limit 2000 --normalize_images  --model radimagenet --is_split_per_patient --reverse_split_ds1 --is_only_splitted_loaded --description TESTdata_EPOCH30-postcontrast_synthetic-vs-postcontrast_real_radimagenet_normalized-SPLIT1


echo "==================== IMAGENET: ===================="

echo "EPOCH 30: postcontrast synthetic - postcontrast real imagenet normalized"
python3 fid.py pix2pixHD/results/pre2postcontrast_512p_train_1to195/test_30/only_synthetic train_test/test/test_B --normalize_images  --limit 2000 --model imagenet --description TESTdata_EPOCH30-postcontrast_synthetic-vs-postcontrast_real_imagenet_normalized-same_cases_compared_wo_split

#echo "precontrast real - postcontrast real radimagenet normalized"
#python3 fid.py train_test/test/test_A train_test/test/test_B --limit 2000 --normalize_images  --model imagenet --description precontrast_real-vs-postcontrast_real_imagenet_normalized_same_cases_compared_wo_split

echo "==================== RADIMAGENET: ===================="
echo "EPOCH 30: postcontrast synthetic - postcontrast real radimagenet normalized"
python3 fid.py pix2pixHD/results/pre2postcontrast_512p_train_1to195/test_30/only_synthetic train_test/test/test_B --normalize_images --limit 2000 --model radimagenet --description TESTdata_EPOCH30-postcontrast_synthetic-vs-postcontrast_real_radimagenet_normalized-same_cases_compared_wo_split

#echo "precontrast real - postcontrast real radimagenet normalized"
#python3 fid.py train_test/test/test_A train_test/test/test_B --limit 2000 --normalize_images  --model radimagenet --description precontrast_real-vs-postcontrast_real_radimagenet_normalized_same_cases_compared_wo_split


echo "3. FID computation on TEST DATASET: Done"




