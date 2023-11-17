# Based on: https://ianmcatee.com/converting-a-nifti-file-to-an-image-sequence-using-python/

import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import cv2
import csv
import os
from tqdm import tqdm

# global variables

### Local
PREFIX_PATH = os.path.join('PATH_TO_DATA', '')
PREFIX_OUTPUT_PATH = PREFIX_PATH

### Server
PREFIX_PATH = os.path.join('PATH_TO_DATA', '')
PREFIX_PATH2 = os.path.join('PATH_TO_DATA/Duke-Breast-Cancer-MRI-Nifti-Whole-No-Masks','')   #manifest-YOUR_TCIA_ID/', '')
PREFIX_OUTPUT_PATH = os.path.join('PATH_TO_OUTPUT', '')  # PREFIX_PATH

INPUT_FOLDER_PATH = os.path.join(PREFIX_PATH, 'Duke-Breast-Cancer-MRI-Nifti-Whole')
OUTPUT_FOLDER_PATH = os.path.join(PREFIX_OUTPUT_PATH, 'Duke-Breast-Cancer-MRI-png-Whole')
CSV_PATHS = []
CSV_PATHS.append(os.path.join(PREFIX_PATH, 'Duke_Breast_MRI_all_phases.csv'))
CSV_PATHS.append(os.path.join(PREFIX_PATH2, 'Duke_Breast_MRI_all_phases_without_masks.csv'))
VERBOSE = False  # True to display the middle slices of the scan and nifti header information
FOR_PIX2PIX = True  # Set true to store in pix2pixHD desired folder structure (train_A, train_B)
VIEWS = ['axial']  # ['sagittal', 'coronal', 'axial']
#STORE_GRAYSCALE = True
RESIZE_TO = 512  # None

# specific approach for data to be used in pix2pixHD
if FOR_PIX2PIX:
    OUTPUT_FOLDER_PATH = [os.path.join(PREFIX_OUTPUT_PATH, 'train', 'train_A'),
                          os.path.join(PREFIX_OUTPUT_PATH, 'train', 'train_B'),
                          os.path.join(PREFIX_OUTPUT_PATH, 'test', 'test_A'),
                          os.path.join(PREFIX_OUTPUT_PATH, 'test', 'test_B'),
                          os.path.join(PREFIX_OUTPUT_PATH, 'validation', 'validation_A'),
                          os.path.join(PREFIX_OUTPUT_PATH, 'validation', 'validation_B')]

    VIEWS = ['axial']
    # Min slide- previously tried with:  #25 #20
    # 1 is the smallest number of slice indices that contains a tumour in the Duke Dataset (see Annotation_Boxes.xlsx)
    # SLIDE_MIN = 0

    # Max slide- previously tried with:  #120 #125
    # 196 is the largest number of slice indices that contains a tumour in the Duke Dataset (see Annotation_Boxes.xlsx)
    # SLIDE_MAX = 196
    SLIDE_MIN = -1
    SLIDE_MAX = 999999


# The below validation and testset lists are for the Duke dataset based on segmentation masks from Caballo et al
VALIDATIONSET_LIST = ['Breast_MRI_001','Breast_MRI_002','Breast_MRI_005','Breast_MRI_010','Breast_MRI_012','Breast_MRI_019','Breast_MRI_021','Breast_MRI_022','Breast_MRI_028','Breast_MRI_032','Breast_MRI_043','Breast_MRI_044','Breast_MRI_051','Breast_MRI_055','Breast_MRI_057','Breast_MRI_059','Breast_MRI_060','Breast_MRI_061','Breast_MRI_069','Breast_MRI_071','Breast_MRI_077','Breast_MRI_082','Breast_MRI_091','Breast_MRI_099','Breast_MRI_101','Breast_MRI_103','Breast_MRI_104','Breast_MRI_105','Breast_MRI_107','Breast_MRI_114','Breast_MRI_115','Breast_MRI_116','Breast_MRI_117','Breast_MRI_119','Breast_MRI_120','Breast_MRI_123','Breast_MRI_129','Breast_MRI_132','Breast_MRI_134','Breast_MRI_136','Breast_MRI_137','Breast_MRI_141','Breast_MRI_142','Breast_MRI_144','Breast_MRI_150','Breast_MRI_156','Breast_MRI_157','Breast_MRI_160','Breast_MRI_167','Breast_MRI_168','Breast_MRI_176','Breast_MRI_177','Breast_MRI_178','Breast_MRI_180','Breast_MRI_185','Breast_MRI_189','Breast_MRI_192','Breast_MRI_198','Breast_MRI_202','Breast_MRI_205','Breast_MRI_211','Breast_MRI_218','Breast_MRI_225','Breast_MRI_228','Breast_MRI_233','Breast_MRI_234','Breast_MRI_236','Breast_MRI_237','Breast_MRI_239','Breast_MRI_240','Breast_MRI_244','Breast_MRI_253','Breast_MRI_255','Breast_MRI_258','Breast_MRI_265','Breast_MRI_269','Breast_MRI_271','Breast_MRI_275','Breast_MRI_282','Breast_MRI_283','Breast_MRI_290','Breast_MRI_298','Breast_MRI_301','Breast_MRI_303','Breast_MRI_304','Breast_MRI_306','Breast_MRI_313','Breast_MRI_317','Breast_MRI_323','Breast_MRI_328','Breast_MRI_333','Breast_MRI_338','Breast_MRI_345','Breast_MRI_350','Breast_MRI_353','Breast_MRI_360','Breast_MRI_383','Breast_MRI_386','Breast_MRI_395','Breast_MRI_397','Breast_MRI_398','Breast_MRI_399','Breast_MRI_400','Breast_MRI_407','Breast_MRI_408','Breast_MRI_424','Breast_MRI_428','Breast_MRI_429','Breast_MRI_435','Breast_MRI_438','Breast_MRI_441','Breast_MRI_444','Breast_MRI_454','Breast_MRI_457','Breast_MRI_464','Breast_MRI_465','Breast_MRI_468','Breast_MRI_474','Breast_MRI_486','Breast_MRI_489','Breast_MRI_491','Breast_MRI_501','Breast_MRI_506','Breast_MRI_507','Breast_MRI_508','Breast_MRI_512','Breast_MRI_514','Breast_MRI_521','Breast_MRI_525','Breast_MRI_530','Breast_MRI_534','Breast_MRI_539','Breast_MRI_541','Breast_MRI_543','Breast_MRI_546','Breast_MRI_552','Breast_MRI_558','Breast_MRI_559','Breast_MRI_560','Breast_MRI_562','Breast_MRI_567','Breast_MRI_577','Breast_MRI_585','Breast_MRI_590','Breast_MRI_595','Breast_MRI_597','Breast_MRI_605','Breast_MRI_607','Breast_MRI_609','Breast_MRI_610','Breast_MRI_612','Breast_MRI_614','Breast_MRI_615','Breast_MRI_616','Breast_MRI_623','Breast_MRI_636','Breast_MRI_641','Breast_MRI_645','Breast_MRI_650','Breast_MRI_651','Breast_MRI_652','Breast_MRI_656','Breast_MRI_660','Breast_MRI_663','Breast_MRI_666','Breast_MRI_670','Breast_MRI_672','Breast_MRI_677','Breast_MRI_679','Breast_MRI_686','Breast_MRI_687','Breast_MRI_691','Breast_MRI_693','Breast_MRI_694','Breast_MRI_697','Breast_MRI_718','Breast_MRI_724','Breast_MRI_725','Breast_MRI_735','Breast_MRI_746','Breast_MRI_751','Breast_MRI_754','Breast_MRI_757','Breast_MRI_758','Breast_MRI_762','Breast_MRI_765','Breast_MRI_774','Breast_MRI_775','Breast_MRI_780','Breast_MRI_789','Breast_MRI_790','Breast_MRI_792','Breast_MRI_797','Breast_MRI_804','Breast_MRI_805','Breast_MRI_809','Breast_MRI_812','Breast_MRI_816','Breast_MRI_830','Breast_MRI_831','Breast_MRI_832','Breast_MRI_833','Breast_MRI_834','Breast_MRI_836','Breast_MRI_839','Breast_MRI_847','Breast_MRI_850','Breast_MRI_860','Breast_MRI_865','Breast_MRI_869','Breast_MRI_873','Breast_MRI_874','Breast_MRI_879','Breast_MRI_882','Breast_MRI_883','Breast_MRI_884','Breast_MRI_885','Breast_MRI_886','Breast_MRI_891','Breast_MRI_899','Breast_MRI_914','Breast_MRI_915','Breast_MRI_916','Breast_MRI_917']
TESTSET_LIST = ['Breast_MRI_009','Breast_MRI_041','Breast_MRI_045','Breast_MRI_048','Breast_MRI_064','Breast_MRI_097','Breast_MRI_163','Breast_MRI_183','Breast_MRI_260','Breast_MRI_268','Breast_MRI_287','Breast_MRI_307','Breast_MRI_356','Breast_MRI_368','Breast_MRI_377','Breast_MRI_378','Breast_MRI_387','Breast_MRI_412','Breast_MRI_414','Breast_MRI_431','Breast_MRI_568','Breast_MRI_618','Breast_MRI_633','Breast_MRI_640','Breast_MRI_642','Breast_MRI_662','Breast_MRI_684','Breast_MRI_778','Breast_MRI_799','Breast_MRI_907']


def get_aspect_ratios(nifti_file_header):
    # Calculate the aspect ratios based on pixdim values extracted from the header
    pix_dim = nifti_file_header['pixdim'][1:4]
    aspect_ratios = [pix_dim[1] / pix_dim[2], pix_dim[0] / pix_dim[2], pix_dim[0] / pix_dim[1]]
    if VERBOSE: print('The required aspect ratios are: ', aspect_ratios)
    return aspect_ratios, pix_dim


def get_new_scan_dims(nifti_file_array, pix_dim, is_mulitply=True):
    # Calculate new image dimensions based on the aspect ratio
    if is_mulitply:
        new_dims = np.multiply(nifti_file_array.shape, pix_dim)
    else:
        new_dims = np.divide(nifti_file_array.shape, pix_dim)
    new_dims = (round(new_dims[0]), round(new_dims[1]), round(new_dims[2]))
    if VERBOSE: print('The new scan dimensions are: ', new_dims)
    return new_dims


def display_middle_slice(scan_array, plane=['axial'], aspect_ratios=[1, 1, 1]):
    scan_array_shape = scan_array.shape
    # Display scan array's middle slices
    fig, axs = plt.subplots(1, 3)
    fig.suptitle(f'Scan Array (Middle Slices): {plane} \n aspect ratios: {aspect_ratios}')
    if 'sagittal' in plane:
        axs[0].imshow(scan_array[scan_array_shape[0] // 2, :, :], aspect=aspect_ratios[0], cmap='gray')
    if 'coronal' in plane:
        axs[1].imshow(scan_array[:, scan_array_shape[1] // 2, :], aspect=aspect_ratios[1], cmap='gray')
    if 'axial' in plane:
        axs[2].imshow(scan_array[:, :, scan_array_shape[2] // 2], aspect=aspect_ratios[2], cmap='gray')
    fig.tight_layout()
    plt.show()


def nifti_to_png(filepath, target_folder, filename, is_rotated=True, is_normalised=True):
    # Load the NIfTI scan and extract data using nibabel
    nifti_file = nib.load(filepath)
    nifti_file_array = nifti_file.get_fdata()
    aspect_ratios, pix_dim = get_aspect_ratios(nifti_file.header)
    # Explore the data a bit if verbose
    if VERBOSE:
        print('The nifti header is as follows: \n', nifti_file.header)
        display_middle_slice(nifti_file_array)
        display_middle_slice(nifti_file_array, aspect_ratios=aspect_ratios)

    # Get the new scan dimensions based on the aspect ratios
    new_dims = get_new_scan_dims(nifti_file_array=nifti_file_array, pix_dim=pix_dim)
    if is_normalised:
        if "09" in filename and VERBOSE:
            # checking axial view of slide 80  to see voxel value ranges e.g. for scan 0009.
            print(
                f'{filename} axial view pixel values: '
                f'max={np.max(nifti_file_array[:, :, :])} '
                f'min={np.min(nifti_file_array[:, :, :])} '
                f'mean={np.mean(nifti_file_array[:, :, :])}, '
                f'std={np.std(nifti_file_array[:, :, :])}')

        # Normalise image array to range [0, 255]
        nifti_file_array = ((nifti_file_array - np.min(nifti_file_array)) / (
                np.max(nifti_file_array) - np.min(nifti_file_array))) * 255.0

        if "09" in filename and VERBOSE:
            # checking 3D MRI voxel values to see normalised values e.g. for scan 0009.
            print(
                f'{filename} NORMALISED axial view pixel values: '
                f'max={np.max(nifti_file_array[:, :, :])} '
                f'min={np.min(nifti_file_array[:, :, :])} '
                f'mean={np.mean(nifti_file_array[:, :, :])}, '
                f'std={np.std(nifti_file_array[:, :, :])}')

    # Iterate over nifti_file_array in either coronal, axial or sagittal view to extract slices
    for view in VIEWS:
        if view == 'sagittal':
            idx = 0
        elif view == 'coronal':
            idx = 1
        elif view == 'axial':
            idx = 2
        for i in range(nifti_file_array.shape[idx]):
            if FOR_PIX2PIX and (i < SLIDE_MIN or i > SLIDE_MAX):
                continue
            if view == 'sagittal':
                img = nifti_file_array[i, :, :]
                # img = np.flip(img, axis=0)
                # As we are normally shrinking the image, INTER_AREA interpolation is preferred.
                img = cv2.resize(img, (new_dims[2], new_dims[1]), interpolation=cv2.INTER_AREA)
                if is_rotated:
                    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif view == 'coronal':
                img = nifti_file_array[:, i, :]
                # img = np.flip(img, axis=0)
                img = cv2.resize(img, (new_dims[2], new_dims[0]), interpolation=cv2.INTER_AREA)
                if is_rotated:
                    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif view == 'axial':
                img = nifti_file_array[:, :, i]
                img = cv2.resize(img, (new_dims[1], new_dims[0]), interpolation=cv2.INTER_AREA)
                # Rotate the image 90 degrees counter-clockwise to get the correct orientation
                if is_rotated:
                    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            # Create the target folder if it does not exist
            if FOR_PIX2PIX:
                target_folder_w_view = target_folder
            else:
                target_folder_w_view = os.path.join(target_folder, view)
            os.makedirs(target_folder_w_view, exist_ok=True)
            # Save the image as a PNG file
            if RESIZE_TO is not None:
                assert type(RESIZE_TO) == int and RESIZE_TO > 0, "RESIZE_TO must be an integer greater than 0."
                # Note: Here we use bicubic interpolation as this is also used internally in Pix2PixHD (get_transform function in base_dataset.py)
                # For reproducibility, we want to have the same input images (i.e. cv2.INTER_CUBIC resized images) for any generative model that we use
                # In general, bicubic interpolation is a preferred option for upscaling images.
                img = cv2.resize(img, (RESIZE_TO, RESIZE_TO), interpolation=cv2.INTER_CUBIC)

            #if STORE_GRAYSCALE:
                #print(img.shape)
                #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(os.path.join(target_folder_w_view, f'{filename}_slice{i}.png'), img)


def convert_and_save(patient):
    # Set variables for conversion
    patient_id = patient[0]

    # We only want to do the conversion for files that are available in the indicated folder
    if not os.path.exists(os.path.join(INPUT_FOLDER_PATH, patient_id)):
        if VERBOSE:
            print(f"Patient folder '{os.path.join(INPUT_FOLDER_PATH, patient_id)}' does not exist. Skipping patient.")
    else:

        # Iterate over different T1-Weighted Dynamic Contrast-Enhanced MRI phases/sequences for current patient.
        digit = 0
        for indx in range(2, len(patient)):
            if patient[indx] != '':

                # Check if the patient should be in our test set or training set
                # Also, check if the T1-weighted DCE-MRI sequence is relevant for the current task
                # For now, only pre-contrast (=0) and post-contrast 1 (=1) are relevant for pix2pix
                if FOR_PIX2PIX and digit not in [0, 1]:
                    continue
                elif FOR_PIX2PIX and digit == 0:
                    if patient_id in TESTSET_LIST:
                        target_folder = OUTPUT_FOLDER_PATH[2]
                    elif patient_id in VALIDATIONSET_LIST:
                        target_folder = OUTPUT_FOLDER_PATH[4]
                    else:
                        target_folder = OUTPUT_FOLDER_PATH[0]
                elif FOR_PIX2PIX and digit == 1:
                    if patient_id in TESTSET_LIST:
                        target_folder = OUTPUT_FOLDER_PATH[3]
                    elif patient_id in VALIDATIONSET_LIST:
                        target_folder = OUTPUT_FOLDER_PATH[5]
                    else:
                        target_folder = OUTPUT_FOLDER_PATH[1]

                else:
                    target_folder = os.path.join(OUTPUT_FOLDER_PATH, patient_id)

                # Convert the NIfTI file to a PNG file and store it in the target folder
                nifti_to_png(filepath=os.path.join(INPUT_FOLDER_PATH, patient_id, f'{patient_id}_000{digit}.nii.gz'),
                             target_folder=target_folder,
                             filename=patient_id + '_000' + str(digit),
                             )
                # digit + 1 -> next phase/sequence of the MRI scan
                digit = digit + 1


# with open('PATH_TO_DATA/Duke_Breast_MRI_all_phases.csv') as file_obj:
for idx, csv_path in enumerate(CSV_PATHS):
    if idx == 0 and not len(CSV_PATHS) == 1:
        INPUT_FOLDER_PATH = os.path.join(PREFIX_PATH, 'Duke-Breast-Cancer-MRI-Nifti-Whole')
    else:
        INPUT_FOLDER_PATH = os.path.join(PREFIX_PATH2, 'Duke-Breast-Cancer-MRI-Nifti-Whole')
    with open(csv_path) as file_obj:
        reader_obj = csv.reader(file_obj)
        for row in tqdm(reader_obj):
            convert_and_save(row)
