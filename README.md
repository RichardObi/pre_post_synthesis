
## [Pre- to Post-Contrast Breast MRI Synthesis for Enhanced Tumour Segmentation]()

In SPIE Medical Imaging 2024.

![examples](docs/examples.png)

## Getting Started
The [Duke Dataset](https://sites.duke.edu/mazurowski/resources/breast-cancer-mri-dataset/) used in this study is available on [The Cancer Imaging Archive (TCIA)](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70226903).

You may find some examples of synthetic nifti files in [synthesis/examples](synthesis/examples).

### Synthesis Code
- [Config](synthesis/pix2pixHD/scripts/train_512p_duke_2D_w_GPU_1to195.sh) to run a training of the image synthesis model.
- [Config](synthesis/pix2pixHD/scripts/test_512p_duke_2D_w_GPU_1to195.sh) to run a test of the image synthesis model.
- [Code](synthesis/utils/convert_to_nifti_whole_dataset.py) to transform Duke DICOM files to NiFti files.
- [Code](synthesis/utils/nifti_png_conversion.py) to extract 2D pngs from 3D NiFti files.
- [Code](synthesis/utils/png_nifti_conversion.py) to create 3D NiFti files from axial 2D pngs.
- [Code](synthesis/utils/get_training_patient_ids.py) to separate synthesis training and test cases.
- [Code](synthesis/utils/metrics.py) to compute the image quality metrics such as SSIM, MSE, LPIPS, and more. 
- [Code](synthesis/utils/fid.py) to compute the Frèchet Inception Distance (FID) based on ImageNet and [RadImageNet](https://github.com/BMEII-AI/RadImageNet).  

### Segmentation Code
- [Code](nnUNet/custom_scripts/convert_data_to_nnunet_204.py) to prepare 3D single breast cases for nnunet segmentation.
- [Train-test-splits](nnUNet/nnunetv2/nnUNet_preprocessed/Dataset208_DukePreSynthetic/splits_final_pre_post_syn.json) of the segmentation dataset.
- [Script](nnUNet/custom_scripts/full_pipeline.sh) to run the full nnunet pipeline on the Duke dataset.


## Run the model
Model weights are stored on on [Zenodo](https://zenodo.org/records/10210945) and made available via the [medigan](https://github.com/RichardObi/medigan) library.

To create your own post-contrast data, simply run:

```command
pip install medigan
```

```python
# import medigan and initialize Generators
from medigan import Generators
generators = Generators()

# generate 10 samples with model 23 (00023_PIX2PIXHD_BREAST_DCEMRI). 
# Also, auto-install required model dependencies.
generators.generate(model_id='00023_PIX2PIXHD_BREAST_DCEMRI', num_samples=10, install_dependencies=True)
```

## Reference
Please consider citing our work if you found it useful for your research:
```bibtex
@article{osuala2023pre,
  title={{Pre-to Post-Contrast Breast MRI Synthesis for Enhanced Tumour Segmentation}},
  author={Osuala, Richard and Joshi, Smriti and Tsirikoglou, Apostolia and Garrucho, Lidia and Pinaya, Walter HL and Diaz, Oliver and Lekadir, Karim},
  journal={arXiv preprint arXiv:2311.10879},
  year={2023}
  }
```

## Acknowledgements
This repository borrows code from the [pix2pixHD](https://github.com/NVIDIA/pix2pixHD) and the [nnUNet](https://github.com/MIC-DKFZ/nnUNet) repositories. The 254 tumour segmentation masks used in this study were provided by [Caballo et al](https://doi.org/10.1002/jmri.28273).
