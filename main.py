from pathlib import Path
import pandas as pd
from utils import get_split
from datasets import PaddyDataset
from augs import train_transforms, valid_transforms
from learner import Learner
from torchvision import models
import torch.nn as nn
import torch

data_path = Path('../data')
train_images = data_path/'train_images'
test_images = data_path/'test_images'
train_meta = pd.read_csv(data_path/'train.csv')

img_files = list(train_images.glob('*/*.jpg'))
labels = [p.parts[-2] for p in img_files]
label_idx = {label:i for i,label in enumerate(set(labels))}
train_images, valid_images = get_split(items=img_files,split=0.2,seed=42)


train_ds = PaddyDataset(files=train_images, label_idx=label_idx, transform=train_transforms)
valid_ds = PaddyDataset(files=valid_images, label_idx=label_idx, transform=valid_transforms)
model = models.resnet34(pretrained=True)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
learn = Learner(train_ds, valid_ds, model, loss_fn, optimizer)
learn.fit(1)
