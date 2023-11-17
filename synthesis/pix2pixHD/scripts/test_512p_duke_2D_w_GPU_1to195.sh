#!/bin/bash
################################ Testing ################################

echo 'First arg:' $1

# Duke Dataset
python test.py \
--name pre2postcontrast_512p_train_1to195 \
--model pix2pixHD \
--loadSize 512 \
--label_nc 0 \
--input_nc 3 \
--output_nc 3 \
--no_instance \
--dataroot train_test/validation \
--checkpoints_dir pix2pixHD/checkpoints \
--how_many 1000000000 \
--which_epoch $1 \
--phase validation \
#--which_epoch 10 \
#--phase test \
#--tf_log \
#--print_freq 100 \
#--nThreads 0 \
#--gpu_ids -1 \
#--fp16 \
#--dataroot PATH_TO_DATA/duke_all_png_slices_1to196/test \
