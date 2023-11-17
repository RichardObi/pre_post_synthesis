import os
import SimpleITK as sitk
import numpy as np
from scipy import ndimage
import csv
import random

import shutil

def save_image(save_array, reference_img, save_path):
    image = sitk.GetImageFromArray(save_array)
    image.SetDirection(reference_img.GetDirection())
    image.SetSpacing(reference_img.GetSpacing())
    image.SetOrigin(reference_img.GetOrigin())
    
    sitk.WriteImage(image, os.path.join(save_path))


def save_the_half(paths): #patient, pre_path, post_path

    patient = paths[0]
    post_path = paths[1]

    if os.path.exists(post_path):
        post = sitk.ReadImage(post_path)
        save_name = post_path.split('/')[-1]
        print(save_name)
        
        if not os.path.exists(os.path.join('/workspace/AutomaticSegmentation/nnUNet/nnunetv2/nnUNet_raw/Dataset204_DukePhaseOneHalf/imagesTr', save_name)):

            # create the half image based on where the mask is:
            post_array = sitk.GetArrayFromImage(post)

            # find which side the segmentation is - right or left
            if os.path.exists(os.path.join('/data/Duke-Breast-Cancer-MRI-Nifti-Whole-Preprocessed/masks_preprocessed', patient + '.nii.gz')):
                segmentation = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join('/data/Duke-Breast-Cancer-MRI-Nifti-Whole-Preprocessed/masks_preprocessed', patient + '.nii.gz')))
                segmentation_coordinates = ndimage.center_of_mass(segmentation) 
                
                middle_coordinate = segmentation.shape[-1]/2
                if segmentation_coordinates[-1] < middle_coordinate:
                    new_post = post_array[:, :, :np.int64(middle_coordinate)]
                    new_segmentation = segmentation[:, :, :np.int64(middle_coordinate)]
                else:
                    new_post = post_array[:, :, np.int64(middle_coordinate + 1):]
                    new_segmentation= segmentation[:, :, np.int64(middle_coordinate + 1):]
                save_image(new_post, post, os.path.join('/workspace/AutomaticSegmentation/nnUNet/nnunetv2/nnUNet_raw/Dataset204_DukePhaseOneHalf/imagesTr', patient+'_0000.nii.gz'))
                save_image(new_segmentation, post, os.path.join('/workspace/AutomaticSegmentation/nnUNet/nnunetv2/nnUNet_raw/Dataset204_DukePhaseOneHalf/labelsTr',  patient+'.nii.gz'))

def main():
    processed_nifti_path = '/data/Duke-Breast-Cancer-MRI-Nifti-Whole-Preprocessed'
    pre_post_list = []

    # step 1
    for patient in os.listdir(processed_nifti_path):

        post = os.path.join(processed_nifti_path, patient, patient + '_0001.nii.gz')
        save_the_half([patient, post])
    #     # pre_post_list.append([patient, pre, post])
    
    # # step 2
    # # remove multifocal images
    
    # step 3
    # select test data and remove it from the training set

main()