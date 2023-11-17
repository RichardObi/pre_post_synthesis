"""
Calculates different metrics such as the MULTI-SCALE STRUCTURAL SIMILARITY INDEX (MS-SSIM) between a pair of images
Related Paper: https://www.cns.nyu.edu/pub/eero/wang03b.pdf

Partially based on https://github.com/Warvito/generative_brain_controlnet/blob/main/src/python/testing/compute_msssim_sample.py
and https://github.com/Warvito/generative_brain_controlnet/blob/main/src/python/testing/compute_controlnet_performance.py

Usage:
    python metrics.py dir1 dir2
"""
import argparse
import os
import cv2
from tqdm import tqdm
import numpy as np
import glob
import torch
from pathlib import Path
from csv import writer
from datetime import datetime

np.seterr('raise') # outcomment after debugging

VERBOSE = False

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculates the mean image-to-image comparison metric between two dataset."
    )
    parser.add_argument(
        "dataset_path_1",
        type=str,
        help="Path to images from first dataset",
    )
    parser.add_argument(
        "dataset_path_2",
        type=str,
        help="Path to images from second dataset",
    )
    parser.add_argument(
        "--metrics",
        type=list,
        default= ['mae', 'mse'], #['mae', 'mse', 'psnr', 'ssim', 'ms-ssim', 'lpips'],  #, 'lpips'], #['mse'],
        help="Select the metric to use. Currently 'ms-ssim', 'ssim', 'mse', 'mae', 'lpips', and 'psnr' are supported.",
    )
    parser.add_argument(
        "--normalize_images",
        action="store_true",
        help="Normalize images from both data sources using min and max of each sample",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default= 4999,
        help="Max number of images to load from each data source",
    )
    args = parser.parse_args()
    return args


def load_images(file_names, normalize=False, resize=True, resize_size=512, split=False, limit=None):
    """
    Loads images from the given directory.
    If split is True, then half of the images is loaded to one array and the other half to another.
    """

    if split:
        subset_1 = []
        subset_2 = []
    else:
        images = []

    # glob here assumes that pngs are in the desired order e.g. axial plane starting from the top.
    #file_names = sorted(glob.glob(f'{directory}/*.png', root_dir=directory), key=sorting_by_last_k_to_n_elements)
    #file_names = sorted(glob.glob(f'{directory}/*.png')) if ".png" in os.listdir(directory)[0] else sorted(glob.glob(f'{directory}/*.jpg'))

    for count, filename in enumerate(file_names):
    #for count, filename in enumerate(os.listdir(directory)):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            #img = cv2.imread(os.path.join(directory, filename))
            img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            #img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
            if normalize:
                img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

            if resize:
                img = cv2.resize(img, (resize_size, resize_size), interpolation=cv2.INTER_LINEAR)
            #if len(img.shape) > 2 and img.shape[2] == 4:
            #    img = img[:, :, :3]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, axis=2)

            if split:
                if count % 2:
                    subset_1.append(img)
                else:
                    subset_2.append(img)
            else:
                images.append(img)
        if limit is not None and count == limit:
            break
    if split:
        subset_1 = preprocess_input(np.array(subset_1))
        subset_2 = preprocess_input(np.array(subset_2))
        return subset_1, subset_2
    else:
        images = preprocess_input(np.array(images))
        return images

def preprocess_input(images_as_np_array):
    """
    Preprocesses the images.
    TODO: Define any necessary preprocessing steps below.
    """
    return images_as_np_array

def get_metric_function(metric):
    if metric =='lpips':
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        # lpips needs values between -1 and 1
        lpips = LearnedPerceptualImagePatchSimilarity() #net_type='squeeze', normalize=False)
        metric_function = lambda a,b: lpips(a, b)
        metric_transform = lambda x: torch.from_numpy(cv2.normalize(x.numpy(), None, -1, 1, cv2.NORM_MINMAX)).unsqueeze(0).permute(0, 3, 1, 2).to(torch.float32)
    elif metric =='ms-ssim':
        from torchmetrics.functional import multiscale_structural_similarity_index_measure
        # TODO Check if kernel size and other hyperparams are fine and may try different ones
        metric_function = lambda a,b: multiscale_structural_similarity_index_measure(a, b)
        metric_transform = lambda x: x.unsqueeze(0).permute(0, 3, 1, 2).to(torch.float32)
    elif metric == 'ssim':
        from torchmetrics.functional import structural_similarity_index_measure
        # TODO Check if kernel size and other hyperparams are fine and may try different ones
        metric_function = lambda a, b: structural_similarity_index_measure(a, b)
        metric_transform = lambda x: x.unsqueeze(0).permute(0, 3, 1, 2).to(torch.float32)
    elif metric == 'kid':
        raise NotImplementedError("Kernel Inception Distance is not yet implemented")
        #from generative.metrics import MMDMetric
        #from torchmetrics.image.kid import KernelInceptionDistance
        #kid = KernelInceptionDistance() # TODO Check if kernel size is fine and may try different ones
        #metric_function = lambda a, b: kid(a, b)
        #metric_transform = lambda x: x
    elif metric == 'mae':
        from torchmetrics import MeanAbsoluteError
        mae = MeanAbsoluteError()
        metric_function = lambda a, b: mae(a, b)
        metric_transform = lambda x: x
    elif metric == 'mse':
        from torchmetrics import MeanSquaredError
        mse = MeanSquaredError()
        metric_function = lambda a, b: mse(a, b)
        metric_transform = lambda x: x
    elif metric == 'psnr':
        from torchmetrics import PeakSignalNoiseRatio
        psnr = PeakSignalNoiseRatio()
        metric_function = lambda a, b: psnr(a, b)
        metric_transform = lambda x: x
    else:
        raise ValueError("Invalid metric name: {}".format(metric))
    return metric_function,  metric_transform

def get_file_transformations(file_name):
    transformed = [file_name, file_name.replace("_0001", "_0000"), file_name.replace("_0000", "_0001")]
    final_transformed = []
    for transformed_filename in transformed:
        final_transformed.extend(
            [transformed_filename.replace("png", "jpg"), transformed_filename.replace("jpg", "png")])
    return list(set(final_transformed)) # remove duplicates

def check_if_files_correspond(directory_1, directory_2, rename=True, enforce_strict_file_correspondence=True):

    if rename and ("_synthesized_image" in os.listdir(directory_1)[0] or "_synthesized_image" in os.listdir(directory_2)[0]):
        directories = [directory_1, directory_2]
        for directory in directories:
            for filename in os.listdir(directory):
                my_dest = os.path.join(directory, filename.replace("_synthesized_image", ""))
                my_source = os.path.join(directory, filename)
                os.rename(my_source, my_dest)

    file_names_1 = sorted(glob.glob(f'{directory_1}/*.png')) if ".png" in os.listdir(directory_1)[0] else sorted(glob.glob(f'{directory_1}/*.jpg'))
    file_names_2 = sorted(glob.glob(f'{directory_2}/*.png')) if ".png" in os.listdir(directory_2)[0] else sorted(glob.glob(f'{directory_2}/*.jpg'))
    file_names_without_path_1 = sorted(os.listdir(directory_1))
    file_names_without_path_2 = sorted(os.listdir(directory_2))

    if enforce_strict_file_correspondence:
        # enforce that only images are used where the same patient case with same slice number is present in both datasets (0001 post- or 0000 pre-contrast are okay)
        file_names_1_new = [file_name for file_name in file_names_1 if any(x in file_names_without_path_2 for x in get_file_transformations(os.path.basename(file_name)))]
        file_names_2_new = [file_name for file_name in file_names_2 if any(x in file_names_without_path_1 for x in get_file_transformations(os.path.basename(file_name)))]
        file_names_1 = file_names_1_new
        file_names_2 = file_names_2_new

    assert len(file_names_1) == len(file_names_2), f"Number of images in both datasets must be equal. {len(file_names_1)}!={len(file_names_2)}"
    assert len(file_names_1) != 0 or len(file_names_2) != 0, f"Number of file_names in a folder cannot be 0. Please revise. From {directory_1}: {len(file_names_1)}. From {directory_2}:{len(file_names_2)}"

    if not len(os.listdir(directory_1)) == len(os.listdir(directory_2)):
        print(f"Number of images in both datasets adjusted to {len(file_names_1)}. Initially number of images in {directory_1} and {directory_2} was not equal. {len(os.listdir(directory_1))}!={len(os.listdir(directory_2))}.")

    idx_for_checks = [0, 10, 30, int(len(file_names_1)/3), int(len(file_names_1)/2), len(file_names_1)-1]
    for idx in idx_for_checks:
        filename_1 = Path(os.fsdecode(file_names_1[idx])).name
        filename_2 = Path(os.fsdecode(file_names_2[idx])).name
        assert filename_1.replace("_synthesized_image", "").replace("0001", "0000").replace("jpg", "png") == filename_2.replace("_synthesized_image", "").replace("0001", "0000").replace("jpg", "png"), f"Files (at idx={idx}) do not correspond: {filename_1} and {filename_2}"

    return file_names_1, file_names_2
def calculate_metrics_for_dataset(
    directory_1,
    directory_2,
    metric_list,
    limit,
    normalize_images=False,
):
    file_names_1, file_names_2 = check_if_files_correspond(directory_1, directory_2)

    if limit is None: limit = min(len(file_names_1), len(file_names_2)) #min(len(os.listdir(directory_1)), len(os.listdir(directory_2)))

    if VERBOSE:
        print(f"Found {len(file_names_1)} files in {directory_1}. Limit={limit}")
        print(f"Found {len(file_names_2)} files in {directory_2}. Limit={limit}")

    images_1 = load_images(file_names_1, resize=True, resize_size=512, normalize=normalize_images, limit=limit,)
    images_2 = load_images(file_names_2, resize=True, resize_size=512, normalize=normalize_images, limit=limit,)


    metric_dict = {}
    for metric in metric_list:
        metric_dict[metric] = []
    if VERBOSE:
        print(f"Now calculating metrics ({metric_list}) for {len(images_1)}-{len(images_2)} image pairs (limit={limit}).")
    pbar = tqdm(enumerate(images_1), total=len(images_1))

    for idx, image in pbar:
        for metric in metric_list:
            metric_function, metric_transform = get_metric_function(metric)
            metric_score = metric_function(metric_transform(torch.from_numpy(image)), metric_transform(torch.from_numpy(images_2[idx])))
            metric_dict[metric].append(metric_score.item())
            if idx % 100 == 0 and VERBOSE:
                print(f"Computed {metric} for {idx} images. Last {metric} score: {metric_score.item()}")
        if VERBOSE:
            pbar.set_description(f"Computing metrics for image {idx}")

        pbar.update()

    return metric_dict

if __name__ == "__main__":
    args = parse_args()

    directory_1 = args.dataset_path_1
    directory_2 = args.dataset_path_2
    normalize_images = args.normalize_images
    if VERBOSE:
        print(f"Computing metrics: {args.metrics}...")
    metric_dict = calculate_metrics_for_dataset(
        directory_1=directory_1,
        directory_2=directory_2,
        normalize_images=normalize_images,
        metric_list=args.metrics,
        limit=args.limit if args.limit is not None else 10000000000,
    )
    metric_strings = []
    for metric in metric_dict:
        metric_score_list = np.array(metric_dict[metric])

        metric_string = f"{metric}: {np.mean(metric_score_list)}+-{np.std(metric_score_list)}"
        metric_strings.append(metric_string)
        # Summary print of results
        if VERBOSE:
            print(
                f"Mean(std) of {metric} based on {'<normalised>' if normalize_images else '<non-normalised>'} images in {directory_1} and {directory_2}: {np.mean(metric_score_list)}+-{np.std(metric_score_list)}")
        else:
            #print(f"metric_score_list: {metric_score_list}")
            print(metric_string)


    metrics_results = [args.limit + 1, f'normalised: {normalize_images}', directory_1, directory_2, str(datetime.now())]
    metrics_results.extend(metric_strings)

    # Open existing CSV file in append mode and add FID info
    with open('metrics.csv', 'a') as f_object:
        writer_object = writer(f_object)
        writer_object.writerow(metrics_results)
        f_object.close()
