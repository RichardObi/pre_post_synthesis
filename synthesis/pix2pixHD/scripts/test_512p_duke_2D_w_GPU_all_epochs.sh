#!/bin/sh
################################ Testing ################################


#for i in {2..13};
#do
#  echo "epoch = $((10 * i))" ;
#  sh scripts/test_512p_duke_2D_w_GPU_1to195.sh $((10 * i));
#done


sh scripts/test_512p_duke_2D_w_GPU_1to195.sh 10
sh scripts/test_512p_duke_2D_w_GPU_1to195.sh 170
sh scripts/test_512p_duke_2D_w_GPU_1to195.sh 130