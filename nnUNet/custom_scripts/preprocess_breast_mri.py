
import os
import SimpleITK as sitk


def preprocess_breast_mri(nifti_mri, output_filepath=None, resample=True, spacing=[1, 1, 1], normalize=True, bias_correction=True, shrink_factor=4):   
    
    image_itk = sitk.ReadImage(nifti_mri, sitk.sitkFloat32)

    if bias_correction:
    # N4BiasFieldCorrectionImageFilter takes too long to run, shrink image
        mask_breast = sitk.OtsuThreshold(image_itk, 0, 1)
        shrinked_image_itk = sitk.Shrink(image_itk, [shrink_factor] * image_itk.GetDimension())
        shrinked_mask_breast = sitk.Shrink(mask_breast, [shrink_factor] * mask_breast.GetDimension())
        # sitk.WriteImage(mask_breast, os.path.join(output_folder_mris, patient_id + '_mask.nii.gz'))
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrected_image = corrector.Execute(shrinked_image_itk, shrinked_mask_breast)
        log_bias_field = corrector.GetLogBiasFieldAsImage(image_itk)
        # sitk.WriteImage(log_bias_field, os.path.join(output_folder_mris, patient_id + '_log_bias_field.nii.gz'))
        corrected_image_itk = image_itk / sitk.Exp(log_bias_field)
        image_itk = corrected_image_itk

    if normalize:
        # NormalizeImageFilter: normalizes an image by setting its mean to zero and variance to one. 
        # RescaleIntensity: sets output to be in range 0-255 (in our case).
        sitk.RescaleIntensity(image_itk, 0, 255)

    if output_filepath:
        sitk.WriteImage(image_itk, output_filepath)

