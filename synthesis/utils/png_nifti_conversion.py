# Partly based on: https://ianmcatee.com/converting-a-nifti-file-to-an-image-sequence-using-python/

import csv
import cv2
import os
import SimpleITK as sitk
import glob
import numpy as np
from tqdm import tqdm
import math

# global variables

# CHECKPOINT to be used
CHECKPOINT = 30

# whether to allow resized synthetic nifti (e.g. avoiding the following error):
# sitk::ERROR: Source image size of [ 448, 448, 160 ] does not match this image's size of [ 512, 512, 159 ]!
ALLOW_RESIZED_SYNTHETIC_NIFTI = False

# This variable defines whether to resize the pngs back so that they match the spacing of the original nifti (e.g. precontrast).
# This is e.g. important to be able to use the existing segmentation masks.
ADJUST_ASPECT_RATIOS_TO_ORIGINAL_NIFTI = True

# whether to transform synthetic nifti to grayscale
TRANSFORM_TO_GRAYSCALE = False

### Local
#PREFIX_PATH = os.path.join('PATH_TO_DATA/', '')
#PREFIX_OUTPUT_PATH = PREFIX_PATH

### Server
PREFIX_PATH = os.path.join('PATH_TO_DATA/')
#PREFIX_OUTPUT_PATH = os.path.join('/PATH_TO_OUTPUT', '')  # PREFIX_PATH
#PREFIX_OUTPUT_PATH = PREFIX_PATH

INPUT_FOLDER_PATH_NIFTI = os.path.join(PREFIX_PATH, 'Duke-Breast-Cancer-MRI-Nifti-Whole')
INPUT_FOLDER_PATH_2DSLICES = os.path.join(f'pix2pixHD/results/pre2postcontrast_512p_train_1to195/test_{CHECKPOINT}/only_synthetic/', '') # synthetic 2d slices
#INPUT_FOLDER_PATH_2DSLICES = os.path.join(PREFIX_PATH, 'Duke-Breast-Cancer-MRI-png-Whole') # real 2d slices
OUTPUT_FOLDER_PATH = os.path.join(PREFIX_PATH, 'Duke-Breast-Cancer-MRI-Nifti-Whole-Synthetic')

CSV_PATH = os.path.join(PREFIX_PATH, 'Duke_Breast_MRI_all_phases.csv')
VERBOSE = False  # verbose logging for debugging & analysis
ARE_ALL_SYNTHETIC_IN_SAME_FOLDER = True
ITERATE_OVER_ALL_PHASES = False

def validate_pngs(slice2d_folder_path, patient_id):
    for file in os.listdir(slice2d_folder_path):
        if not file.endswith('.png'):
            print(f"File '{file}' not a .png file. Please revise why it is in this folder.")
        if not file.startswith(patient_id):
            print(f"File '{file}' does not belong to patient '{patient_id}'. Please revise why it is in this folder.")
        # TODO What else should we test here?

def get_nifti_header(base_nifti_path):
    # Get header info from corresponding NIFTI file
    reader = sitk.ImageFileReader()
    reader.SetFileName(base_nifti_path)
    reader.LoadPrivateTagsOn() # avoid omitting private tags
    reader.ReadImageInformation()

    header_dict = {}
    for k in reader.GetMetaDataKeys():
        v = reader.GetMetaData(k)
        if VERBOSE:
            print(f'{k}=={v}')
        header_dict[k] = v
    return header_dict


def sorting_by_last_k_to_n_elements(elem, k=7, n=4, only_numbers=True):
    # e.g. n = 4 -> .png
    # e.g. k = -7, n=0 -> 123.png
    # e.g. k = -7, n=4 -> 123
    if only_numbers:
        # it might be that we are dealing with 1.png or 12.png instead of 123.png
        while not str(elem[len(elem)-k]).isnumeric():
            if VERBOSE:
                print(f"Digit {len(elem)-k} of {elem} = {elem[len(elem)-k]}")
            k = k-1
    if VERBOSE:
        print(elem[-k:-n])
    return int(elem[-k:-n])

def get_new_scan_dims(nifti_file_array, pix_dim, is_mulitply=False):
    # Calculate new image dimensions based on the aspect ratio
    if is_mulitply:
        new_dims = np.multiply(nifti_file_array.shape, pix_dim)
    else:
        new_dims = np.divide(nifti_file_array.shape, pix_dim)
    new_dims = (round(new_dims[0]), round(new_dims[1]), round(new_dims[2]))
    if VERBOSE: print('The new scan dimensions are: ', new_dims)
    return new_dims

def patient_2dslices_to_nifti(base_nifti_path, target_nifti_path, slice2d_folder_path, patient_id):

    # Get header info from corresponding NIFTI file
    header_dict = get_nifti_header(base_nifti_path=base_nifti_path)

    # Copy spatial information from the base nifti file
    base_nifti_image = sitk.ReadImage(base_nifti_path)

    # glob here assumes that pngs/jps are in the desired order e.g. axial plane starting from the top.
    if ".png" in os.listdir(slice2d_folder_path)[0]:
        file_names = sorted(glob.glob(f'{slice2d_folder_path}/*.png'), key=sorting_by_last_k_to_n_elements)
    elif ".jpg" in os.listdir(slice2d_folder_path)[0]:
        file_names = sorted(glob.glob(f'{slice2d_folder_path}/*.jpg'), key=sorting_by_last_k_to_n_elements)
    else:
        # watch out if you have ".jpeg" files
        raise ValueError(f"No .png or .jpg files found in the input image folder {slice2d_folder_path}. Please check.")

    # Are synthetic data separated by patient into patient folders or all in same folder?
    # If all in same folder, create a list of all png files for folder
    if ARE_ALL_SYNTHETIC_IN_SAME_FOLDER:
        file_names = [filename for filename in file_names if patient_id in filename]

    if TRANSFORM_TO_GRAYSCALE or ADJUST_ASPECT_RATIOS_TO_ORIGINAL_NIFTI:
        for idx, file_name in enumerate(file_names):
            if TRANSFORM_TO_GRAYSCALE:
                img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
            else:
                img = cv2.imread(file_name)

            if ADJUST_ASPECT_RATIOS_TO_ORIGINAL_NIFTI:
                # assumption 1: The png slices have originally been extracted and resized using the same spacing reference (e.g. from the same corresponding precontrast image) as we use here.
                # assumption 2: We are using axial slides, so we only need to adjust the first two dimensions
                img = cv2.resize(img, (int(header_dict["dim[2]"]), int(header_dict["dim[1]"])), interpolation=cv2.INTER_AREA)

            cv2.imwrite(file_name, img)

    if VERBOSE:
        print(file_names)
        print(header_dict["dim[2]"])
        print(header_dict["dim[1]"])

    # Read the series of slices
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(file_names)
    volume = reader.Execute()

    # More info on orientation: http://www.grahamwideman.com/gw/brain/orientation/orientterms.htm
    volume = sitk.DICOMOrient(volume, 'LAS') # 'LAS' is shown as "RPI" in info in ITKSnap

    if ALLOW_RESIZED_SYNTHETIC_NIFTI and not ADJUST_ASPECT_RATIOS_TO_ORIGINAL_NIFTI:
        # see https://stackoverflow.com/a/48065819
        base_nifti_image = sitk.Resample(base_nifti_image, volume.GetSize(), sitk.Transform(), sitk.sitkLinear, base_nifti_image.GetOrigin(), volume.GetSpacing(), base_nifti_image.GetDirection(), 0.0, base_nifti_image.GetPixelID())

    try:
        # The below CopyInformation function should already include what is done in
        # https://nipy.org/nibabel/reference/nibabel.orientations.html#nibabel.orientations.aff2axcodes
        # E.g. orientation should be preserved from the original image (e.g. RPI being the Orientation RAI code as seen in ITKSnap)
        volume.CopyInformation(base_nifti_image)  # copy Origin, Spacing, and Direction from the source image
        # the above CopyInformation function might fail if the base_nifti_image does not have the same dimensions as the synthetic image
        # e.g. due to operations performed for aspect ratio and spacing (1,1,1) instead of original
        # extract the spacing (e.g. x=0.8, y=0.8, z=1.1) as pix_dim of the original nifti file
    except:
        # check if original nifti and synthetic nifti have the same dimensions
        if base_nifti_image.GetSize() != volume.GetSize():
            raise ValueError(f"The original nifti and synthetic nifti have different dimensions ({base_nifti_image.GetSize()}) and {volume.GetSize()}. Please check. Original: {base_nifti_path}.")
        elif base_nifti_image.GetSpacing() != volume.GetSpacing():
            raise ValueError(f"The original nifti and synthetic nifti have the same size () but different spacing ({base_nifti_image.GetSpacing()}) and {volume.GetSpacing()}. Please check. Original: {base_nifti_path}.")

    # Copy header info from the base nifti file
    for k in header_dict.keys():
        volume.SetMetaData(key=k, value=header_dict[k])

    # Create file
    sitk.WriteImage(volume, target_nifti_path)


with open(CSV_PATH) as file_obj:
    reader_obj = csv.reader(file_obj)

    patients = list(reader_obj)
    print(f"Number of patients in CSV file: {len(patients)}")
    # Number of patient in INPUT_FOLDER_PATH_NIFTI
    print(f"Number of files in NIFTI Input folder: {len(os.listdir(INPUT_FOLDER_PATH_NIFTI))}. Example file: {os.listdir(INPUT_FOLDER_PATH_NIFTI)[0]}")
    print(f"Number of files in 2DSLICES folder: {len(os.listdir(INPUT_FOLDER_PATH_2DSLICES))}. Example file: {os.listdir(INPUT_FOLDER_PATH_2DSLICES)[0]}")
    for patient in tqdm(iterable=patients, desc='Patients', total=len(patients)):
        patient_id = patient[0]
        digit = 0
        for indx in range(2, len(patient)):
            # Iterate over DCE-MRI sequences/phases for each patient and create a nifti file for each available sequence/phase
            if patient[indx] != '':
                # Convert the png/jpg file to a NIfTI file and store it in the target folder
                base_nifti_path = os.path.join(INPUT_FOLDER_PATH_NIFTI, patient_id, f'{patient_id}_000{digit}.nii.gz')
                if not os.path.isfile(base_nifti_path):
                    print(f"File '{base_nifti_path}' does not exist. Please revise why not.")
                if ARE_ALL_SYNTHETIC_IN_SAME_FOLDER:
                    slice2d_folder_path = os.path.join(INPUT_FOLDER_PATH_2DSLICES)
                else:
                    slice2d_folder_path = os.path.join(INPUT_FOLDER_PATH_2DSLICES, patient_id, f'{patient_id}_000{digit}')
                if not os.path.isdir(slice2d_folder_path):
                    print(f"Folder '{slice2d_folder_path}' does not exist. Please revise why not.")
                if TRANSFORM_TO_GRAYSCALE:
                    target_nifti_path = os.path.join(OUTPUT_FOLDER_PATH, f'{patient_id}', f'{patient_id}_000{digit if digit != 0 else digit+1}S_grayscale.nii.gz')
                else:
                    target_nifti_path = os.path.join(OUTPUT_FOLDER_PATH, f'{patient_id}', f'{patient_id}_000{digit if digit != 0 else digit+1}S.nii.gz')
                if not os.path.isdir(os.path.dirname(target_nifti_path)):
                    #os.umask(0)
                    os.makedirs(os.path.dirname(target_nifti_path), exist_ok=True, mode=0o777)
                # digit + 1 -> next phase/sequence of the MRI scan
                digit = digit + 1
                patient_2dslices_to_nifti(base_nifti_path, target_nifti_path, slice2d_folder_path, patient_id)
            if not ITERATE_OVER_ALL_PHASES:
                break