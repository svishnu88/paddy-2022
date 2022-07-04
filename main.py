from pathlib import Path
import pandas as pd
from utils import get_split, prepare_data
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR


from learner import Learner
from torchvision import models
import torch.nn as nn
import torch
import timm
import torch.backends.cudnn as cudnn
import wandb
from utils import Config

wandb.login()
cudnn.benchmark = True

config = Config(epochs=6,
                path = Path('../data'),
                classes=10,
                batch_size=32,
                lr=1e-4,
                weight_decay=1e-2,
                momentum=0.9,
                image_size=224,
                model_name='resnet26d')


train_ds, valid_ds, label_idx = prepare_data(config)
train_dl = DataLoader(dataset=train_ds,batch_size=config.batch_size,shuffle=True,num_workers=config.num_workers,drop_last=True)
valid_dl = DataLoader(dataset=valid_ds,batch_size=config.batch_size,shuffle=False,num_workers=config.num_workers)

model = timm.create_model(model_name=config.model_name,pretrained=True,num_classes=config.classes)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr/25, weight_decay=config.weight_decay)
lr_scheduler = OneCycleLR(optimizer=optimizer, max_lr=config.lr, epochs=config.epochs, steps_per_epoch=len(train_dl))

learn = Learner(train_dl, valid_dl, model, loss_fn, optimizer, lr_scheduler, config)
learn.fit(config.epochs,freeze_until=1)


# https://twitter.com/karpathy/status/1528808361558306817