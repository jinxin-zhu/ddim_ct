import os
from glob import glob
import shutil
from tqdm import tqdm
import dicom2nifti
import numpy as np
import nibabel as nib
from monai.transforms import (
    Compose,
    AddChanneld,
    LoadImaged,
    Resized,
    ToTensord,
    Spacingd,
    Orientationd,
    ScaleIntensityRanged,
    CropForegroundd,
)
from monai.data import DataLoader, Dataset, CacheDataset
from monai.utils import set_determinism
"""
This file is for preporcessing only, it contains all the functions that you need
to make your data ready for training.

You need to install the required libraries if you do not already have them.

pip install os, ...
"""


def create_groups(in_dir, out_dir, Number_slices):
    """
    This function is to get the last part of the path so that we can use it to name the folder.
    `in_dir`: the path to your folders that contain dicom files
    `out_dir`: the path where you want to put the converted nifti files
    `Number_slices`: here you put the number of slices that you need for your project and it will
    create groups with this number.
    """

    for patient in glob(in_dir + "/*"):
        patient_name = os.path.basename(os.path.normpath(patient))

        # Here we need to calculate the number of folders which mean into how many groups we will divide the number of slices
        number_folders = int(len(glob(patient + "/*")) / Number_slices)

        for i in range(number_folders):
            output_path = os.path.join(out_dir, patient_name + "_" + str(i))
            os.mkdir(output_path)

            # Move the slices into a specific folder so that you will save memory in your desk
            for i, file in enumerate(glob(patient + "/*")):
                if i == Number_slices + 1:
                    break

                shutil.move(file, output_path)


def dcm2nifti(in_dir, out_dir):
    """
    This function will be used to convert dicoms into nifti files after creating the groups with
    the number of slices that you want.
    `in_dir`: the path to the folder where you have all the patients (folder of all the groups).
    `out_dir`: the path to the output, which means where you want to save the converted nifties.
    """

    for folder in tqdm(glob(in_dir + "/*")):
        patient_name = os.path.basename(os.path.normpath(folder))
        dicom2nifti.dicom_series_to_nifti(
            folder, os.path.join(out_dir, patient_name + ".nii.gz"))


def find_empy(in_dir):
    """
    This function will help you to find the empty volumes that you may not need for your training
    so instead of opening all the files and search for the empty ones, them use this function to make it quick.
    """

    list_patients = []
    for patient in glob(os.path.join(in_dir, "*")):
        img = nib.load(patient)

        if len(np.unique(img.get_fdata())) > 2:
            print(os.path.basename(os.path.normpath(patient)))
            list_patients.append(os.path.basename(os.path.normpath(patient)))

    return list_patients


def prepare_data(cfg):
    """
    This function is for preprocessing, it contains only the basic transforms, but you can add more operations that you
    find in the Monai documentation.
    https://monai.io/docs.html
    """

    set_determinism(seed=0)
    path_train_volumes = sorted(
        glob(os.path.join(cfg.DATA_PREPARE.TRAIN_IMAGE_DIR, "*.nii.gz"))
    ) if cfg.DATA_PREPARE.TRAIN_IMAGE_DIR is not None else None

    path_train_segmentation = sorted(
        glob(os.path.join(cfg.DATA_PREPARE.TRAIN_MASK_DIR, "*.nii.gz"))
    ) if cfg.DATA_PREPARE.TRAIN_MASK_DIR is not None else None

    path_test_volumes = sorted(
        glob(os.path.join(cfg.DATA_PREPARE.TEST_IMAGE_DIR, "*.nii.gz"))
    ) if cfg.DATA_PREPARE.TEST_IMAGE_DIR is not None else None

    path_test_segmentation = sorted(
        glob(os.path.join(cfg.DATA_PREPARE.TEST_MASK_DIR, "*.nii.gz"))
    ) if cfg.DATA_PREPARE.TEST_MASK_DIR is not None else None

    path_unlabel_volumes = sorted(
        glob(os.path.join(cfg.DATA_PREPARE.UNLABEL_IMAGE_DIR, "*.nii.gz"))
    ) if cfg.DATA_PREPARE.UNLABEL_IMAGE_DIR is not None else None

    train_files = [{
        "vol": image_name,
        "seg": label_name
    } for image_name, label_name in zip(path_train_volumes,
                                        path_train_segmentation)] if path_train_volumes is not None and path_train_segmentation is not None else None
    test_files = [{
        "vol": image_name,
        "seg": label_name
    } for image_name, label_name in zip(path_test_volumes,
                                        path_test_segmentation)] if path_test_volumes is not None and path_test_segmentation is not None else None
    unlabel_files = [{
        "vol": image_name
    } for image_name in path_unlabel_volumes] if path_unlabel_volumes is not None else None

    train_transforms = Compose([
        LoadImaged(keys=["vol", "seg"]),
        AddChanneld(keys=["vol", "seg"]),
        Spacingd(
            keys=["vol", "seg"],
            pixdim=cfg.DATA_PREPARE.SPACINGD.PIXDIM,
            mode=cfg.DATA_PREPARE.SPACINGD.MODE,
        ),
        Orientationd(keys=["vol", "seg"],
                     axcodes=cfg.DATA_PREPARE.ORIENTATIOND.AXCODES),
        ScaleIntensityRanged(
            keys=["vol"],
            a_min=cfg.DATA_PREPARE.SCALEINTENSITYRANGED.A_MIN,
            a_max=cfg.DATA_PREPARE.SCALEINTENSITYRANGED.A_MAX,
            b_min=cfg.DATA_PREPARE.SCALEINTENSITYRANGED.B_MIN,
            b_max=cfg.DATA_PREPARE.SCALEINTENSITYRANGED.B_MAX,
            clip=cfg.DATA_PREPARE.SCALEINTENSITYRANGED.CLIP,
        ),
        CropForegroundd(keys=["vol", "seg"], source_key="vol"),
        Resized(keys=["vol", "seg"],
                spatial_size=cfg.DATA_PREPARE.RESIZED.SPATIAL_SIZE),
        ToTensord(keys=["vol", "seg"]),
    ])

    test_transforms = Compose([
        LoadImaged(keys=["vol", "seg"]),
        AddChanneld(keys=["vol", "seg"]),
        Spacingd(
            keys=["vol", "seg"],
            pixdim=cfg.DATA_PREPARE.SPACINGD.PIXDIM,
            mode=cfg.DATA_PREPARE.SPACINGD.MODE,
        ),
        Orientationd(keys=["vol", "seg"],
                     axcodes=cfg.DATA_PREPARE.ORIENTATIOND.AXCODES),
        ScaleIntensityRanged(
            keys=["vol"],
            a_min=cfg.DATA_PREPARE.SCALEINTENSITYRANGED.A_MIN,
            a_max=cfg.DATA_PREPARE.SCALEINTENSITYRANGED.A_MAX,
            b_min=cfg.DATA_PREPARE.SCALEINTENSITYRANGED.B_MIN,
            b_max=cfg.DATA_PREPARE.SCALEINTENSITYRANGED.B_MAX,
            clip=cfg.DATA_PREPARE.SCALEINTENSITYRANGED.CLIP,
        ),
        CropForegroundd(keys=["vol", "seg"], source_key="vol"),
        Resized(keys=["vol", "seg"],
                spatial_size=cfg.DATA_PREPARE.RESIZED.SPATIAL_SIZE),
        ToTensord(keys=["vol", "seg"]),
    ])

    unlabel_transforms = Compose([
        LoadImaged(keys=["vol"]),
        AddChanneld(keys=["vol"]),
        Spacingd(
            keys=["vol", "seg"],
            pixdim=cfg.DATA_PREPARE.SPACINGD.PIXDIM,
            mode=cfg.DATA_PREPARE.SPACINGD.MODE,
            allow_missing_keys=True,
        ),
        Orientationd(keys=["vol"],
                     axcodes=cfg.DATA_PREPARE.ORIENTATIOND.AXCODES),
        ScaleIntensityRanged(
            keys=["vol"],
            a_min=cfg.DATA_PREPARE.SCALEINTENSITYRANGED.A_MIN,
            a_max=cfg.DATA_PREPARE.SCALEINTENSITYRANGED.A_MAX,
            b_min=cfg.DATA_PREPARE.SCALEINTENSITYRANGED.B_MIN,
            b_max=cfg.DATA_PREPARE.SCALEINTENSITYRANGED.B_MAX,
            clip=cfg.DATA_PREPARE.SCALEINTENSITYRANGED.CLIP,
        ),
        CropForegroundd(keys=["vol"], source_key="vol"),
        Resized(keys=["vol"],
                spatial_size=cfg.DATA_PREPARE.RESIZED.SPATIAL_SIZE),
        ToTensord(keys=["vol"]),
    ])

    if cfg.DATA_PREPARE.IS_CACHE:
        train_ds = CacheDataset(data=train_files,
                                transform=train_transforms,
                                cache_rate=1.0)
        # train_loader = DataLoader(train_ds, batch_size=1)

        test_ds = CacheDataset(data=test_files,
                               transform=test_transforms,
                               cache_rate=1.0)
        # test_loader = DataLoader(test_ds, batch_size=1)

        unlabel_ds = CacheDataset(data=unlabel_files,
                                  transform=unlabel_transforms,
                                  cache_rate=1.0)
        # unlabel_loader = DataLoader(unlabel_ds, batch_size=1)

        return train_ds, test_ds, unlabel_ds

    else:
        train_ds = Dataset(data=train_files, transform=train_transforms)
        # train_loader = DataLoader(train_ds, batch_size=1)

        test_ds = Dataset(data=test_files, transform=test_transforms)
        # test_loader = DataLoader(test_ds, batch_size=1)

        unlabel_ds = Dataset(data=unlabel_files, transform=unlabel_transforms)
        # unlabel_loader = DataLoader(unlabel_ds, batch_size=1)

        return train_ds, test_ds, unlabel_ds
