
import os
from tqdm import tqdm
import shutil


# Simple script that copies the files that contain the string 'synthetic' in their filename to another folder
epochs = ['120'] #['40', '20', '10', '80', '50']
for epoch in epochs:
	INPUT_FOLDER_PATH = os.path.join(f'./pix2pixHD/results/pre2postcontrast_512p_train_1to195/test_{epoch}/images', '')
	OUTPUT_FOLDER_PATH = os.path.join(f'./pix2pixHD/results/pre2postcontrast_512p_train_1to195/test_{epoch}/only_synthetic', '')
	IDENTIFYING_STRING = 'synth'

	if not os.path.exists(OUTPUT_FOLDER_PATH):
	    os.makedirs(OUTPUT_FOLDER_PATH)

	for filename in tqdm(os.listdir(INPUT_FOLDER_PATH)):
	    shutil.copy2(src=os.path.join(INPUT_FOLDER_PATH, filename), dst=os.path.join(OUTPUT_FOLDER_PATH, filename)) if IDENTIFYING_STRING in filename else None
