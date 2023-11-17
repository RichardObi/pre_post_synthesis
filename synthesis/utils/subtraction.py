
import cv2 
import numpy as np
import argparse
import os
import glob
from tqdm import tqdm

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculates the subtraction image using opencv."
    )
    parser.add_argument(
        "--dataset_path_1",
        type=str,
        help="Path to images from first dataset - the postcontrast base images",
    )
    parser.add_argument(
        "--dataset_path_2",
        type=str,
        help="Path to images from second dataset - the precontrast to_be_subtracted images",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="train_test/test/subtraction_real",
        help="Path to where the subtraction images will be stored.",
    )
    parser.add_argument(
        "--use_case_names",
        action="store_true",
        help="flag to specify that the name_of_cases cases should be used rather than the input datasets",
    )
    parser.add_argument(
        "--name_of_cases",
        type=list,
        default= ['case378','Case228','Case886','case907'],
        help="If only specific cases in folder should be translated. Note: Specific naming convention needed - see below in code.",
    )
    parser.add_argument(
        "--is_contrast_adjusted",
        action="store_true",
        help="Adjust contrast and intensity of the images",
    )
    args = parser.parse_args()
    return args

def adjust_contrast_and_intensity(images:list) -> list:

    adjusted_images = []
    # alpha = 1. #= 1.5 # contrast control
    # beta = 1. #= 10 # intensity control
    for idx, img in enumerate(images):
        for k in range(0, 30):
            # Image intensity scaling
            img[:, :] = np.where(img[:, :] * 1.03 < 255, (img[:, :] * 1.03).astype(np.uint8), img[:, :])
            img = cv2.convertScaleAbs(img, alpha=1.02, beta=1.04)

            # image intensity and contrast scaling
            # img = cv2.convertScaleAbs(img, alpha=alpha*1.03, beta=beta*1.1)
            # img = cv2.convertScaleAbs(img, alpha=1.5, beta=10)

            # show the results
            cv2.imshow(f'Subtraction image {idx} ', img)
            cv2.waitKey(10)
        adjusted_images.append(img)
        # for k in range(0, 2):
        # Image intensity scaling
        # img[:, :] = np.where(img[:, :] * 1.03 < 255, (img[:, :] * 1.03).astype(np.uint8), img[:, :])

        # image intensity and contrast scaling
        # img = cv2.convertScaleAbs(img, alpha=alpha*1.03, beta=beta*1.1)
        # img = cv2.convertScaleAbs(img, alpha=1.5, beta=10)

        # show the results
        # cv2.imshow('Updated', img)
    # cv2.waitKey(4000)
    return adjusted_images

def subtract_specific_cases(name_of_cases, is_contrast_adjusted):
    for name_of_case in name_of_cases:
        # reading the images
        base_image_1 = cv2.imread(f'{name_of_case}_0001.png')
        base_image_2 = cv2.imread(f'{name_of_case}_000d1S.png')
        image_to_subtract = cv2.imread(f'{name_of_case}_0000.png')

        # subtract the images
        subtracted1 = cv2.subtract(base_image_1, image_to_subtract)
        subtracted2 = cv2.subtract(base_image_2, image_to_subtract)

        if is_contrast_adjusted:
            subtracted1, subtracted2 = adjust_contrast_and_intensity([subtracted1, subtracted2])

            # TO show the output
            for idx, img in enumerate([subtracted1, subtracted2]):
                # cv2.imshow(f'{name_of_case} real subtraction images', subtracted1)
                cv2.imwrite(f'{name_of_case}_SUB_REAL.png' if idx == 0 else f'{name_of_case}_SUB_SYN.png', img)
        # To close the window
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # cv2.imshow(f'{name_of_case} fake subtraction image', subtracted2)
        # cv2.imwrite(f'{name_of_case}_SUB_SYN.png', subtracted2)
        # To close the window
        # cv2.waitKey(0)
        cv2.destroyAllWindows()


def subtract_all_cases(folder_path_1, folder_path_2, is_contrast_adjusted, output_folder, exists_ok = True, resize_if_size_conflict = True):
    # get all files from precontrast image folder using glob
    types = (folder_path_2 +'/*.png', folder_path_2 + '/*.jpg')
    files_grabbed = []
    for filetype in types:
        files_grabbed.extend(glob.glob(filetype))

    # for each precontrast image, we want to find its postcontrast counterpart image in folder_path_1
    count = 0
    for precontrast_image in tqdm(files_grabbed):
        filename = os.path.basename(precontrast_image)
        output_file_path = os.path.join(output_folder, filename.replace('0000', f'SUB'))
        if os.path.exists(output_file_path) and exists_ok:
            count = count+1
            if count < 25:
                print(f'WARNING: {output_file_path} already exists. Skipping.')
            elif count == 25:
                print(f'WARNING: Many files in {output_folder} do already exists. Further existing filenames wont be printed.')
            continue
        # get the postcontrast image
        postcontrast_image = None
        for num in range(0,6): # all postcontrast phases
            for filetype in ['png', 'jpg', 'jpeg']:
                if os.path.exists(os.path.join(folder_path_1, filename.replace('0000', f'000{num}').replace('png', filetype))):
                    postcontrast_image = os.path.join(folder_path_1, filename.replace('0000', f'000{num}').replace('png', filetype))
                    break
        if postcontrast_image is None:
            print(f'WARNING: Could not find postcontrast image for {filename} in {folder_path_1}')
            continue
        postcontrast_image_ = cv2.imread(postcontrast_image, cv2.IMREAD_GRAYSCALE)
        precontrast_image_ = cv2.imread(precontrast_image, cv2.IMREAD_GRAYSCALE)
        if postcontrast_image_.shape != precontrast_image_.shape:
            if resize_if_size_conflict:
                # for upscaling when resizing, INTER_CUBIC is better than INTER_LINEAR. However, in metrics.py and fid.py INTER_LINEAR is used.
                # Therefore, we use INTER_LINEAR here as well for better reproducibility.
                postcontrast_image_ = cv2.resize(postcontrast_image_, precontrast_image_.shape, interpolation = cv2.INTER_LINEAR) #cv2.INTER_CUBIC
            else:
                print(f'{postcontrast_image_.shape} <- {postcontrast_image}')
                print(f'{precontrast_image_.shape} <- {precontrast_image}')
                continue
        subtracted_image = cv2.subtract(postcontrast_image_, precontrast_image_)
        if is_contrast_adjusted:
            print("Adjusting contrast of subtracted image.")
            subtracted_image = adjust_contrast_and_intensity([subtracted_image])

        # TO show the output
        # cv2.imshow(f'{filename} subtraction images', subtracted_image)
        cv2.imwrite(output_file_path, subtracted_image)


if __name__ == "__main__":
    args = parse_args()
    print(f"args for subtraction images computation: {args}")
    name_of_cases = args.name_of_cases
    is_contrast_adjusted = args.is_contrast_adjusted
    folder_path_1 = args.dataset_path_1
    folder_path_2 = args.dataset_path_2
    output_folder = args.output_folder
    use_case_names = args.use_case_names

    if use_case_names:
        subtract_specific_cases(name_of_cases, is_contrast_adjusted)
    else:
        os.makedirs(output_folder, exist_ok=True)
        subtract_all_cases(folder_path_1, folder_path_2, is_contrast_adjusted, output_folder)


