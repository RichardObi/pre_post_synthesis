

######## PREPROCESSING GENERAL (MANUAL, outside of nnunet)
# --> breast_mri.py
# bias field correction is done here
# Input dataset is the NIFTI Files from the Duke dataset.

####### PREPROCESSING SPECIAL (MANUAL, outside of nnunet)
# --> convert_data_to_nnunet_207.py
# Crop data (single breast)
# Remove multfiocal cases

####### NNUNET Preprocessing
export nnUNet_preprocessed='/workspace/AutomaticSegmentation/nnUNet/nnunetv2/nnUNet_preprocessed'
export nnUNet_results='/workspace/AutomaticSegmentation/nnUNet/nnunetv2/nnUNet_results'
export nnUNet_raw='/workspace/AutomaticSegmentation/nnUNet/nnunetv2/nnUNet_raw'

export nnUNet_preprocessed='nnunetv2/nnUNet_preprocessed'
export nnUNet_results='nnunetv2/nnUNet_results'
export nnUNet_raw='nnunetv2/nnUNet_raw'

# 100 is Dataset ID / MODEL_ID
nnUNetv2_plan_and_preprocess -d 210 --verify_dataset_integrity


####### NNUNET Training
# splits.json (in nnUNet_preprocessed folder) is a file with random splits that can be adjusted (so that same data splits are used in each iteration).
# Each fold can be run on one GPU here with "export CUDA_VISIBLE_DEVICES=0" then =1, etc (each in separate file)
export CUDA_VISIBLE_DEVICES=0
nnUNetv2_train 208 3d_fullres 0 --npz #-num_gpus 0 # 100 is MODEL_ID # fold 0
nnUNetv2_train 208 3d_fullres 1 --npz #-num_gpus 0 # 100 is MODEL_ID # fold 1
nnUNetv2_train 208 3d_fullres 2 --npz #-num_gpus 0 # 100 is MODEL_ID # fold 2
nnUNetv2_train 208 3d_fullres 3 --npz #-num_gpus 0 # 100 is MODEL_ID # fold 3
nnUNetv2_train 208 3d_fullres 4 --npz #-num_gpus 0 # 100 is MODEL_ID # fold 4


####### NNUNET Inference
# fold-based ensembling is done automatically.
# Different to that, 2d + 3d models are not ensembled by default (run nnunet ensemble rather than nnunet predict for this - see nnunet github for instructions)
# Example: nnUNetv2_predict -i PATH_TO_INPUT -o PATH_TO_OUTPUT -d MODEL_ID -c 3d_fullres
# PATH_TO_INPUT <--nnUNet_preprocessed
# From the ID, the folds are automatically ensembled in -d default mode.
nnUNetv2_predict -i PATH_TO_INPUT -o PATH_TO_OUTPUT -d DATASET_NAME_OR_ID -c 3d_fullres --save_probabilities
#DCE-MRI Phase 1: /workspace/AutomaticSegmentation/nnUNet/nnunetv2/nnUNet_raw/Dataset204_DukePhaseOneHalf/Tests/3d/phase1/imagesTs
#DCE-MRI Phase 2: /workspace/AutomaticSegmentation/nnUNet/nnunetv2/nnUNet_raw/Dataset204_DukePhaseOneHalf/Tests/3d/phase2/imagesTs
#Precontrast Phase: /workspace/AutomaticSegmentation/nnUNet/nnunetv2/nnUNet_raw/Dataset205_DukePreHalf/Tests/3d/prephase/imagesTs

# Example for model trained on pre, post, syn-post
nnUNetv2_predict -i nnunetv2/nnUNet_raw/Dataset205_DukePreHalf/Tests/3d/prephase/imagesTs -o nnunetv2/nnUNet_raw/Dataset205_DukePreHalf/Tests/3d/prephase/predictionTs_prephase_phase1_synphase1 -d 208 -c 3d_fullres
nnUNetv2_predict -i nnunetv2/nnUNet_raw/Dataset204_DukePhaseOneHalf/Tests/3d/phase1/imagesTs -o nnunetv2/nnUNet_raw/Dataset204_DukePhaseOneHalf/Tests/3d/phase1/predictionTs_prephase_phase1_synphase1 -d 208 -c 3d_fullres

# Example for model trained on pre, post
nnUNetv2_predict -i nnunetv2/nnUNet_raw/Dataset205_DukePreHalf/Tests/3d/prephase/imagesTs -o nnunetv2/nnUNet_raw/Dataset205_DukePreHalf/Tests/3d/prephase/predictionTs_prephase_phase1 -d 208 -c 3d_fullres
nnUNetv2_predict -i nnunetv2/nnUNet_raw/Dataset204_DukePhaseOneHalf/Tests/3d/phase1/imagesTs -o nnunetv2/nnUNet_raw/Dataset204_DukePhaseOneHalf/Tests/3d/phase1/predictionTs_prephase_phase1 -d 208 -c 3d_fullres


####### Metrics Calculation
# --> dice_calculation.py (only DICE, needs to be extended for Hausdorff distance)
# PATH_TO_OUTPUT from nnUNetv2_predict is input here.
# e.g., in the file, set
  # path_to_pred = 'nnunetv2/nnUNet_raw/Dataset205_DukePreHalf/Tests/3d/prephase/predictionTs_prephase_phase1_synphase'
  # path_to_gt = 'nnunetv2/nnunetv2/nnUNet_raw/Dataset205_DukePreHalf/labelsTr'
  # OR
  # path_to_pred = 'nnunetv2/nnUNet_raw/Dataset204_DukePhaseOneHalf/Tests/3d/phase1/predictionTs_prephase_phase1_synphase1'
  # path_to_gt = 'nnunetv2/nnUNet_raw/Dataset204_DukePhaseOneHalf/labelsTr'

