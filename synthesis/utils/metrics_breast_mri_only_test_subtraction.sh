#! /bin/bash

#sleep 2h

#### Preliminaries

echo "1. METRIC computation on TEST data."

echo "======================== REAL-SYNTHETIC Comparisons ========================"

echo "SUBTRACTION syn - real (both normalized)"
python3 metrics.py train_test/test/subtraction_synthetic train_test/test/subtraction_real --normalize_images

echo "FINISHED CALCULATING METRICS"
                                                        
sh metrics_breast_mri_only_test.sh