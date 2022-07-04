import numpy as np
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from augs import get_train_augs, get_valid_augs
from datasets import PaddyDataset

@dataclass
class Config:
    epochs:int=4,
    path:Path=None
    classes:int=10,
    batch_size:int=32,
    lr:float=0.05,
    weight_decay:float=1e-4,
    momentum:float=0.9,
    image_size:int=224,
    model_name:str='resnet26d',
    num_workers:int=6

def get_split(items=None,split=0.2,seed=None):
    np.random.seed(seed)
    np.random.shuffle(items)
    n = len(items)
    split_idx = int(n * (1-split))
    return items[:split_idx],items[split_idx:]

def prepare_data(config:Config):
    train_images = config.path/'train_images'
    test_images = config.path/'test_images'
    train_meta = pd.read_csv(config.path/'train.csv')
    img_files = list(train_images.glob('*/*.jpg'))
    labels = [p.parts[-2] for p in img_files]
    label_idx = {label:i for i,label in enumerate(set(labels))}
    train_images, valid_images = get_split(items=img_files,split=0.2,seed=42)
    train_transforms,valid_transforms = get_train_augs(img_sz=config.image_size), get_valid_augs(config.image_size)
    train_ds = PaddyDataset(files=train_images, label_idx=label_idx, transform=train_transforms)
    valid_ds = PaddyDataset(files=valid_images, label_idx=label_idx, transform=valid_transforms)
    return train_ds, valid_ds, label_idx    




