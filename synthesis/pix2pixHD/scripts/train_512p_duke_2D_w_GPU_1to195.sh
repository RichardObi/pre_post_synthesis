################################ Training ################################

# Duke Dataset
python train.py \
--name pre2postcontrast_512p_train_1to195 \
--model pix2pixHD \
--batchSize 1 \
--loadSize 512 \
--label_nc 0 \
--input_nc 3 \
--output_nc 3 \
--no_instance \
--dataroot  PATH_TO_DATA/duke_all_png_slices_1to196/train \
--continue_train
#--gpu_ids 4 \
#--tf_log \
--save_epoch_freq 5 \
#--print_freq 100 \
#--nThreads 0 \
#--gpu_ids -1 \
#--fp16 \
