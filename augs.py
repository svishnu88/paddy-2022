import albumentations as A
from torchvision import transforms as T
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2


def get_train_augs(img_sz=448):
    # tfms = A.Compose([
    #     A.Transpose(p=0.5),
    #     A.VerticalFlip(p=0.5),
    #     A.HorizontalFlip(p=0.5),
    #     A.RandomBrightness(limit=0.2, p=0.75),
    #     A.RandomContrast(limit=0.2, p=0.75),
    #     A.OneOf([
    #         A.MotionBlur(blur_limit=5),
    #         A.MedianBlur(blur_limit=5),
    #         A.GaussianBlur(blur_limit=5),
    #         A.GaussNoise(var_limit=(5.0, 30.0)),
    #     ], p=0.7),

    #     A.OneOf([
    #         A.OpticalDistortion(distort_limit=1.0),
    #         A.GridDistortion(num_steps=5, distort_limit=1.),
    #         A.ElasticTransform(alpha=3),
    #     ], p=0.7),

    #     A.CLAHE(clip_limit=4.0, p=0.7),
    #     A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
    #     A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
    #     A.Resize(img_sz, img_sz),
    #     A.Cutout(max_h_size=int(img_sz * 0.375), max_w_size=int(img_sz * 0.375), num_holes=1, p=0.7),    
    #     A.Normalize(),
    #     ToTensorV2()
    # ])
    tfms = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ImageCompression(quality_lower=99, quality_upper=100),
        A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2,
                           rotate_limit=10, border_mode=0, p=0.7),
        A.RandomResizedCrop(img_sz, img_sz),
        A.Cutout(max_h_size=int(img_sz * 0.4),
                 max_w_size=int(img_sz * 0.4), num_holes=1, p=0.5),
        A.Normalize(),
        ToTensorV2()
    ])
    return tfms


def get_valid_augs(img_sz=448):
    tfms = A.Compose([
        A.Resize(img_sz, img_sz),
        A.Normalize(),
        ToTensorV2()
    ])
    return tfms
