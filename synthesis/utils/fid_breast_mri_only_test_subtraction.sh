
echo "1. Creating and activating virtual environment called MMG_env with preinstalled libs."
python3 -m venv MMG_env
source MMG_env/bin/activate

echo "2. FID computation on SUBTRACTION TEST DATASET: Start"

echo "====================== TEST DATA: NOW CALCULATING FID with Dataset Split ===================="

echo "====================== IMAGENET with Dataset Split: ======================"

echo "SUBTRACTION postcontrast real - postcontrast real imagenet normalized"
python3 fid.py train_test/test/subtraction_real train_test/test/subtraction_real --limit 2000 --normalize_images  --model imagenet --is_split_per_patient --reverse_split_ds2 --is_only_splitted_loaded --description subtraction_TESTdata_postcontrast_real-vs-postcontrast_real_imagenet_normalized_SPLIT1_vs_SPLIT2

echo "SUBTRACTION EPOCH 30: postcontrast synthetic - postcontrast real imagenet normalized - SPLIT 1"
python3 fid.py train_test/test/subtraction_synthetic train_test/test/subtraction_real --limit 2000 --normalize_images  --model imagenet --is_split_per_patient  --reverse_split_ds1 --is_only_splitted_loaded --description subtraction_TESTdata_EPOCH30-postcontrast_synthetic-vs-postcontrast_real_imagenet_normalized-SPLIT1


echo "====================== RADIMAGENET with Dataset Split: ======================"

echo "SUBTRACTION postcontrast real - postcontrast real radimagenet normalized"
python3 fid.py train_test/test/subtraction_real train_test/test/subtraction_real --limit 2000 --normalize_images  --model radimagenet --is_split_per_patient --reverse_split_ds2 --is_only_splitted_loaded --description subtraction_TESTdata_postcontrast_real-vs-postcontrast_real_radimagenet_normalized_SPLIT1_vs_SPLIT2

echo "SUBTRACTION EPOCH 30: postcontrast synthetic - postcontrast real radimagenet normalized - SPLIT 1"
python3 fid.py train_test/test/subtraction_synthetic train_test/test/subtraction_real --limit 2000 --normalize_images  --model radimagenet --is_split_per_patient --reverse_split_ds1 --is_only_splitted_loaded --description subtraction_TESTdata_EPOCH30-postcontrast_synthetic-vs-postcontrast_real_radimagenet_normalized-SPLIT1


echo "==================== IMAGENET: ===================="

echo "SUBTRACTION EPOCH 30: postcontrast synthetic - postcontrast real imagenet normalized"
python3 fid.py train_test/test/subtraction_synthetic train_test/test/subtraction_real --normalize_images  --limit 2000 --model imagenet --description subtraction_TESTdata_EPOCH30-postcontrast_synthetic-vs-postcontrast_real_imagenet_normalized-same_cases_compared_wo_split

echo "==================== RADIMAGENET: ===================="

echo "SUBTRACTION EPOCH 30: postcontrast synthetic - postcontrast real radimagenet normalized"
python3 fid.py train_test/test/subtraction_synthetic train_test/test/subtraction_real --normalize_images --limit 2000 --model radimagenet --description subtraction_TESTdata_EPOCH30-postcontrast_synthetic-vs-postcontrast_real_radimagenet_normalized-same_cases_compared_wo_split


echo "3. FID computation on SUBTRACTION TEST DATASET: Done"




