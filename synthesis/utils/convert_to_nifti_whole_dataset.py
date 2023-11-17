import dicom2nifti as d2n
from tqdm import tqdm
import os
import csv
import sys
import warnings

warnings.filterwarnings("ignore")


def convert_and_save(patient):
    input_folder_path = './Duke-Breast-Cancer-MRI' 
    output_folder_path = './Duke-Breast-Cancer-MRI-Nifti-Whole'
    patient_id = patient[0]
    patient_folder = patient[1]

    digit = 0

    if not os.path.exists(os.path.join(output_folder_path, patient_id)):
        os.makedirs(os.path.join(output_folder_path, patient_id), exist_ok=True)

    phases_failed_to_convert = 0
    for indx in range(2, len(patient)):
        if patient[indx] != '':
            original_dicom_directory = os.path.join(input_folder_path, patient_id, patient_folder, patient[indx])
            output_folder = os.path.join(output_folder_path, patient_id)
            filename = patient_id + '_000' + str(digit) + '.nii.gz'

            if not os.path.exists(os.path.join(output_folder, filename)):
                try:
                    # d2n.convert_directory(original_dicom_directory,
                    #                     output_folder,
                    #                     compression=True, reorient=True)
                    d2n.dicom_series_to_nifti(original_dicom_directory,
                                              output_file=os.path.join(output_folder, filename), reorient_nifti=True)
                except Exception as e:
                    print('Error: ', e)
                    print('Patient: ', patient_id)
                    print('Folder: ', original_dicom_directory)
                    print('Phase: ', str(digit))
                    print('Patient: ' + patient_id + ' - ' + patient[indx] + ' - ' + filename)
                    print('----------------------------------------------------')
                    phases_failed_to_convert = phases_failed_to_convert + 1

            digit = digit + 1
    return phases_failed_to_convert


with open('./Duke_Breast_MRI_all_phases_without_masks.csv') as file_obj:
    phases_failed_to_convert = 0
    reader_obj = csv.reader(file_obj)
    print('----------------------------------------------------')
    for row in tqdm(reader_obj):
        phases_failed_to_convert = convert_and_save(row)
        phases_failed_to_convert = phases_failed_to_convert + 1

    print('Phases failed to convert: ', phases_failed_to_convert)
