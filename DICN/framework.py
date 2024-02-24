import torch
from torch.autograd import Variable
import os
from tqdm import tqdm
from utils.metrics import IoU
from loss import dice_bce_loss
import math
from torch.optim.lr_scheduler import _LRScheduler

class SGDRScheduler(_LRScheduler):
    """ Consine annealing with warm up and restarts.
    Proposed in `SGDR: Stochastic Gradient Descent with Warm Restarts`.
    """
    def __init__(self, optimizer, total_epoch=150, start_cyclical=100, cyclical_base_lr=7e-4, cyclical_epoch=10, eta_min=0, warmup_epoch=10, last_epoch=-1):
        self.total_epoch = total_epoch
        self.start_cyclical = start_cyclical
        self.cyclical_epoch = cyclical_epoch
        self.cyclical_base_lr = cyclical_base_lr
        self.eta_min = eta_min
        self.warmup_epoch = warmup_epoch
        super(SGDRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epoch:
            return [self.eta_min + self.last_epoch*(base_lr - self.eta_min)/self.warmup_epoch for base_lr in self.base_lrs]
        elif self.last_epoch < self.start_cyclical:
            return [self.eta_min + (base_lr-self.eta_min)*(1+math.cos(math.pi*(self.last_epoch-self.warmup_epoch)/(self.start_cyclical-self.warmup_epoch))) / 2 for base_lr in self.base_lrs]
        else:
            return [self.eta_min + (self.cyclical_base_lr-self.eta_min)*(1+math.cos(math.pi* ((self.last_epoch-self.start_cyclical)% self.cyclical_epoch)/self.cyclical_epoch)) / 2 for base_lr in self.base_lrs]



class Solver:
    def __init__(self, net, optimizer, local_rank=-1, lr=2e-4, dist=False, args=None, loss=dice_bce_loss, metrics=IoU):
        # self.net = net.cuda()
        self.net = net.cpu()
        if dist == False:
            if torch.cuda.device_count() > 1:
                self.net = torch.nn.DataParallel(
                    self.net, device_ids=list(range(torch.cuda.device_count()))
                )
            self.optimizer = torch.optim.AdamW(params=self.net.parameters(), lr=lr)
            self.lr_scheduler = SGDRScheduler(self.optimizer, total_epoch=args.epochs, eta_min=args.lr/100, warmup_epoch=3,
                                             start_cyclical=100, cyclical_base_lr=args.lr/2, cyclical_epoch=20)
        else:
            self.net = torch.nn.parallel.DistributedDataParallel(self.net, device_ids=[local_rank], find_unused_parameters=True)
            self.optimizer = torch.optim.AdamW(params=self.net.parameters(), lr=lr)
            self.lr_scheduler = SGDRScheduler(self.optimizer, total_epoch=args.epochs, eta_min=args.lr / 100,
                                              warmup_epoch=3,
                                              start_cyclical=100, cyclical_base_lr=args.lr / 2, cyclical_epoch=20)
        self.loss = loss()
        self.metrics = metrics()

    def set_input(self, img_batch, mask_batch=None, img_id=None):
        self.img = img_batch
        self.mask = mask_batch
        self.img_id = img_id

    def optimize(self, epoch):
        self.net.train()
        self.img = Variable(self.img.cuda())
        if self.mask is not None:
            self.mask = Variable(self.mask.cuda())

        pred = self.net.forward(self.img)
        loss = self.loss(self.mask, pred, epoch)

        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()
        if isinstance(pred, tuple):
            pred = pred[0]
        if isinstance(pred, list):
            pred = pred[0]
        metrics = self.metrics(self.mask, pred)
        return loss.item(), metrics

    def save_weights(self, path):
        torch.save(self.net.state_dict(), path)

    def load_weights(self, path):
        load_weight = torch.load(path)
        self.net.load_state_dict(load_weight, strict=False)

    def test_batch(self):
        self.net.eval()
        with torch.no_grad():
            self.img = Variable(self.img.cuda())
            if self.mask is not None:
                self.mask = Variable(self.mask.cuda())
            pred = self.net.forward(self.img)
            loss = self.loss(self.mask, pred)
            if isinstance(pred, list):
                pred = pred[0]
            metrics = self.metrics(self.mask, pred)
            pred = pred.cpu().data.numpy().squeeze(1)
        return pred, loss.item(), metrics

    def pred_one_image(self, image):
        self.net.eval()
        image = image.cuda().unsqueeze(0)
        pred = self.net.forward(image)
        return pred

class Trainer:
    def __init__(self, *args, **kwargs):
        self.solver = Solver(*args, **kwargs)
        self.local_rank_ = args[2]
        self.dist = args[4]
    def set_train_dl(self, dataloader):
        self.train_dl = dataloader

    def set_validation_dl(self, dataloader):
        self.validation_dl = dataloader

    def set_test_dl(self, dataloader):
        self.test_dl = dataloader

    def set_save_path(self, save_path):
        self.save_path = save_path

    def fit_one_epoch(self, dataloader, epoch, eval=False):
        dataloader_iter = iter(dataloader)
        epoch_loss = 0
        epoch_metrics = 0
        iter_num = len(dataloader_iter)
        progress_bar = tqdm(enumerate(dataloader_iter), total=iter_num)
        num_all = 0
        for i, (img, mask) in progress_bar:
            num_all += img.shape[0]
            self.solver.set_input(img, mask)
            if eval:
                _, iter_loss, iter_metrics = self.solver.test_batch()
            else:
                iter_loss, iter_metrics = self.solver.optimize(epoch)
            epoch_loss += iter_loss
            epoch_metrics += iter_metrics
            progress_bar.set_description(
                f'iter: {i} loss: {iter_loss:.4f} metrics: {iter_metrics[3] / img.shape[0]:.4f}'
            )
        epoch_loss /= iter_num
        epoch_metrics /= num_all
        return epoch_loss, epoch_metrics

    def fit(self, epochs):
        val_best_metrics = [0, 0, 0, 0, 0, 0]
        val_best_loss = float("+inf")
        no_optim = 0
        for epoch in range(1, epochs + 1):
            self.solver.lr_scheduler.step(epoch=epoch)
            lr_epoch = self.solver.lr_scheduler.get_lr()[0]
            print(f"epoch {epoch}/{epochs}    lr = {lr_epoch}")
            print(f"training")
            train_loss, train_metrics = self.fit_one_epoch(self.train_dl, epoch, eval=False)

            print(f"validating")
            val_loss, val_metrics = self.fit_one_epoch(self.validation_dl, epoch, eval=True)

            if self.local_rank_ == 0 or self.dist == False:
                print('epoch finished')
                print(f'train_loss: {train_loss:.4f} train_metrics: {train_metrics}')
                print(f'val_loss: {val_loss:.4f} val_metrics: {val_metrics}')
                if val_metrics[3] > val_best_metrics[3]:
                    val_best_metrics = val_metrics
                    self.solver.save_weights(os.path.join(self.save_path, f"epoch{epoch}_val{val_metrics[3]:.4f}.pth"))
                    print(f"val best metric = {val_best_metrics}")
                print(f"epoch {epoch}/{epochs}, best_A_IoU = {val_best_metrics[3]:.4f}, G_IoU = {val_best_metrics[-2] / val_best_metrics[-1]:.4f}")

            if val_loss < val_best_loss:
                no_optim = 0
                val_best_loss = val_loss
            else:
                no_optim += 1

class Tester:
    def __init__(self, *args, **kwargs):
        self.solver = Solver(*args, **kwargs)

    def set_validation_dl(self, dataloader):
        self.validation_dl = dataloader

    def predict(self):
        pass


