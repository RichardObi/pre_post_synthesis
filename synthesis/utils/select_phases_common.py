import numpy as np
import os
import csv

class Selector:

    def __init__(self, extract_feature = False, feature_path = None):
        self.extract_feature = extract_feature
        self.feature_path = feature_path
        self.select_images()

    def select_images(self):
        image_data = 'path_to_data'
        image_data = os.path.join('PATH_TO_DATA/Duke-Breast-Cancer-MRI', '')


        self.dataset = []
      
        for patient_id in os.listdir(image_data):
            patient_folder = os.listdir(os.path.join(image_data, patient_id))[0]
            patient_path = os.path.join(image_data, patient_id, patient_folder)
            print(patient_folder)

            dce_phase = sorted([s for s in os.listdir(patient_path) if("dyn" in s or "Ax Vibrant" in s)]) #last one for 775
            dce_phase = sorted(dce_phase, key=lambda item: (float(item.partition('-')[0])
                               if item[0].isdigit() else float('inf'), item))
            if not dce_phase: 
                print(patient_id)
            else:
                dce_phase.insert(0, patient_folder)
                dce_phase.insert(0, patient_id)
                self.dataset.append(dce_phase)

    def get_dataset(self):
        return self.dataset
                       
          

    
selector = Selector()

CSV_PATH = 'Duke_Breast_MRI_all_phases_without_masks.csv'
with open(CSV_PATH, 'w') as f:
    # using csv.writer method from CSV package
    write = csv.writer(f)
    write.writerows(selector.get_dataset())

