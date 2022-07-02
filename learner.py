from torch.utils.data import DataLoader
from accelerate import Accelerator
import torch
from fastprogress.fastprogress import master_bar, progress_bar

accelerate = Accelerator()

class Learner():
    def __init__(self,train_ds,valid_ds,model, loss_fn, optimizer,batch_size=32):
        self.train_ds = train_ds
        self.valid_ds = valid_ds
        self.device = accelerate.device
        self.loss_fn = loss_fn
        train_dl = DataLoader(dataset=train_ds,batch_size=batch_size,shuffle=True,num_workers=6)
        valid_dl = DataLoader(dataset=valid_ds,batch_size=batch_size,shuffle=False,num_workers=6)
        self.model, self.train_dl, self.valid_dl, self.opt = accelerate.prepare(model,
                                                                        train_dl,
                                                                        valid_dl,
                                                                        optimizer)
        self.train_loss = AverageMeter('train_loss')
        self.valid_loss = AverageMeter('valid_loss')
        


    def fit(self,epochs=1):
        self.mb = master_bar(range(epochs))
        for epoch in self.mb:
            self.do_train()
            self.do_validate()

    def do_train(self):
        self.model.train()
        for i,batch in enumerate(progress_bar(self.train_dl, parent=self.mb)):
            loss = self.one_batch(i,batch,train=True)
            self.train_loss.update(loss,batch[0].shape[0])
            self.mb.child.comment = f"{self.train_loss}"
    
    def do_validate(self):
        self.model.eval()
        for i,batch in enumerate(progress_bar(self.valid_dl, parent=self.mb)):
            with torch.no_grad():
                loss = self.one_batch(i,batch,train=False)
                self.valid_loss.update(loss,batch[0].shape[0])
                self.mb.child.comment = f"{self.valid_loss}"

    def one_batch(self,i,batch,train=True):
        inp, targs = batch
        outputs = self.model(inp)
        loss = self.loss_fn(outputs,targs)
        if train:
            accelerate.backward(loss)
            self.opt.step()
        return loss.item()

class AverageMeter(object):
    """Computes and stores the average and current value"""
    # Code from https://github.com/pytorch/examples/blob/78acb79062189bd937f17edf7c97571e6ec59083/imagenet/main.py#L397
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


        

    
