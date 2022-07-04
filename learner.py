from accelerate import Accelerator
import torch
from fastprogress.fastprogress import master_bar, progress_bar
import wandb
from utils import Config

accelerate = Accelerator()

class Learner():
    def __init__(self, train_dl, valid_dl, model, loss_fn, optimizer, lr_scheduler, config:Config=None):
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.device = accelerate.device
        self.loss_fn = loss_fn
        self.config = config
        self.opt = optimizer
        self.lr_scheduler = lr_scheduler
        self.model, self.train_dl, self.valid_dl, self.opt= accelerate.prepare(model,
                                                                        train_dl,
                                                                        valid_dl,
                                                                        self.opt)
        
        self.train_loss = AverageMeter('train_loss')
        self.valid_loss = AverageMeter('valid_loss')
        self.valid_acc = AverageMeter('valid_loss')
        
    def freeze(self,last_idx=2):
        print(f"Froze model {self.epoch}")
        for param in list(self.model.parameters())[:-last_idx]:
            param.requires_grad = False
        # for name, param in self.model.named_parameters():
        #     if name.find('bn') != -1:
        #         param.requires_grad = True

    def unfreeze(self):
        print(f"UnFreeze model {self.epoch}")
        for param in list(self.model.parameters()):
            param.requires_grad = True

    def fit(self,epochs=1, freeze_until=1):
        with wandb.init(project="paddy-temp",config=self.config):
            # wandb.watch(self.model,self.loss_fn,log='all',log_freq=100)
            self.mb = master_bar(range(epochs))
            print(['train_loss', 'valid_loss', 'err_rate'])
            for self.epoch in self.mb:
                if self.epoch < freeze_until:
                    self.freeze()
                elif self.epoch == freeze_until:
                    self.unfreeze()
                self.do_train()
                self.do_validate()
                print([f"{self.train_loss.avg:.3f}",f"{self.valid_loss.avg:.3f}",f"{1-self.valid_acc.avg:.3f}"])
            

    def do_train(self):
        self.model.train()
        for i,batch in enumerate(progress_bar(self.train_dl, parent=self.mb)):
            loss,_ = self.one_batch(i,batch,train=True)
            self.train_loss.update(loss,batch[0].shape[0])
            self.mb.child.comment = f"{self.train_loss}"
            if i % 25 == 0:
                # print(self.lr_scheduler.get_last_lr())
                wandb.log({"epoch":self.epoch, "train_loss":self.train_loss.avg,"lr":self.opt.optimizer.param_groups[0]['lr']})
                # self.opt.param_groups[0]['lr']
    
    def do_validate(self):
        self.model.eval()
        for i,batch in enumerate(progress_bar(self.valid_dl, parent=self.mb)):
            with torch.no_grad():
                loss,acc = self.one_batch(i,batch,train=False)
                self.valid_loss.update(loss,batch[0].shape[0])
                self.valid_acc.update(acc,batch[0].shape[0])
                self.mb.child.comment = f"{self.valid_loss}"
                if i % 25 == 0:
                    wandb.log({"epoch":self.epoch, "valid_loss":self.valid_loss.avg})
                    wandb.log({"epoch":self.epoch, "valid_acc":self.valid_acc.avg})

    def one_batch(self,i,batch,train=True):
        inp, targs = batch
        outputs = self.model(inp)
        loss = self.loss_fn(outputs,targs)
        if train:
            accelerate.backward(loss)
            self.opt.step()
            self.lr_scheduler.step()
            for param in self.model.parameters():
                param.grad = None
        acc = accuracy(outputs, targs)
        return loss.item(),acc

def accuracy(out,targ):
    acc = targ.eq(out.max(dim=1)[1]).float().sum()/len(targ)
    return acc.item()

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


        

    
