
# From Smriti 14.08.2023
import os
from nnunetv2.evaluation.evaluate_predictions import compute_metrics
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO

# TODO Change paths
path_to_pred = 'nnUNet/nnunetv2/nnUNet_raw/Dataset203_Pseudolabels/predictionsTs'
path_to_gt = 'nnUNet/nnunetv2/nnUNet_raw/Dataset203_Pseudolabels/labelsTr'

path_to_pred = 'nnunetv2/nnUNet_raw/Dataset205_DukePreHalf/Tests/3d/prephase/predictionTs_prephase_phase1_synphase'
path_to_gt = 'nnunetv2/nnunetv2/nnUNet_raw/Dataset205_DukePreHalf/labelsTr'

path_to_pred = 'nnunetv2/nnUNet_raw/Dataset204_DukePhaseOneHalf/Tests/3d/phase1/predictionTs_prephase_phase1_synphase1'
path_to_gt = 'nnunetv2/nnUNet_raw/Dataset204_DukePhaseOneHalf/labelsTr'

dice = 0
counter = 0
for pred_id in os.listdir(path_to_pred):
    if pred_id.endswith('.nii.gz'):
        gt_path = os.path.join(path_to_gt, pred_id)
        pred_path = os.path.join(path_to_pred, pred_id)
        results = compute_metrics(gt_path, pred_path, image_reader_writer=SimpleITKIO(), labels_or_regions= [0, 1])

        dice = dice + results['metrics'][1]['Dice']
        counter = counter + 1

av_dice = dice/counter
print(av_dice)