import os
import argparse
import multiprocessing
from pathlib import Path
from PIL import Image

import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl

from byol_pytorch import BYOL_RGB_Flow
from dataloader import VideoFlowDataset

# test model, a resnet 50

rgb_resnet = models.video.r3d_18(pretrained=False)
flow_resnet= models.video.r3d_18(pretrained=False)

# arguments

parser = argparse.ArgumentParser(description='byol-lightning-test')

args = parser.parse_args()

# constants

BATCH_SIZE = 16
EPOCHS     = 128
NUM_GPUS   = 2
IMAGE_SIZE = 224
LR         = 1e-3 * (BATCH_SIZE * NUM_GPUS / 256.0)
IMAGE_EXTS = ['.jpg', '.png', '.jpeg']
NUM_WORKERS = 32#multiprocessing.cpu_count()

# pytorch lightning module

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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

class SelfSupervisedLearner(pl.LightningModule):
    def __init__(self, rgb_net, flow_net, **kwargs):
        super().__init__()
        self.learner = BYOL_RGB_Flow(rgb_net, flow_net, **kwargs)
        self.rgb_loss_meter= AverageMeter()
        self.flow_loss_meter = AverageMeter()

    def forward(self, sample):
        return self.learner(sample)

    def training_step(self, sample, _):
        loss_rgb, loss_flow = self.forward(sample)
        #print(loss_rgb, loss_flow)
        #print(loss_rgb.shape, loss_flow.shape)
        loss = ( loss_rgb + loss_flow ) / 2
        self.log('training_rgb_loss', loss_rgb) 
        self.log('training_flow_loss', loss_flow)
        self.log('training_loss', loss)
        self.rgb_loss_meter.update(loss_rgb)
        self.flow_loss_meter.update(loss_flow)
        return loss
    
    def training_epoch_end(self, _):
        self.log('epoch_rgb_loss', self.rgb_loss_meter.avg)
        self.log('epoch_flow_loss', self.flow_loss_meter.avg)
        self.log('epoch_loss', (self.rgb_loss_meter.avg + self.rgb_loss_meter.avg)/2)
        self.rgb_loss_meter.reset()
        self.flow_loss_meter.reset()
    
    def validation_step(self, sample, _):
        loss_rgb, loss_flow = self.forward(sample)
        loss = ( loss_rgb + loss_flow ) / 2
        self.log('validation_rgb_loss', loss_rgb) 
        self.log('validation_flow_loss', loss_flow)
        self.log('validation_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=LR)

    def on_before_zero_grad(self, _):
        if self.learner.use_momentum:
            self.learner.update_moving_average()

if __name__ == '__main__':
    ds = VideoFlowDataset()
    train_loader = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)

    model = SelfSupervisedLearner(
        rgb_resnet,
        flow_resnet,
        image_size = IMAGE_SIZE,
        hidden_layer = 'avgpool',
        projection_size = 512,
        projection_hidden_size = 8192,
        moving_average_decay = 0.99
    )

#     PATH = 'lightning_logs/rgb_flow_byol_test/checkpoints/epoch=21-step=40242.ckpt'
    
#     model = SelfSupervisedLearner.load_from_checkpoint(PATH, rgb_net=rgb_resnet, flow_net=flow_resnet, image_size = 224,
#         hidden_layer = 'avgpool',
#         projection_size = 512,
#         projection_hidden_size = 8192,
#         moving_average_decay = 0.99)

    trainer = pl.Trainer(
        accelerator = 'ddp',
        gpus = NUM_GPUS,
        max_epochs = EPOCHS,
        sync_batchnorm = True,
    )

    trainer.fit(model, train_loader)
