from collections import OrderedDict

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision import transforms
from PIL import Image

from model.models import Refiner, Discriminator
from datasets.NYUDataset import NYUDataset
from datasets.NYUSynthDataset import NYUSynthDataset
from datasets.ConcatDataset import ConcatDataset


class SimGAN(pl.LightningModule):
    def __init__(self, hparams):
        super(SimGAN, self).__init__()
        self.hparams = hparams

        self.refiner = Refiner(10)
        self.discriminator = Discriminator()

        self.img_buffer = None

    def forward(self, x):
        return self.refiner(x)

    def refiner_loss(self, x, y):
        return self.hparams.ref_lambda * F.l1_loss(x, y, reduction='sum')

    def adversarial_loss(self, x, y):
        return F.cross_entropy(x, y)

    def training_step(self, batch, batch_idx, optimizer_idx):
        real_imgs, synth_imgs = batch

        # train generator
        if optimizer_idx == 0:
            imgs_ref = self.refiner(synth_imgs)
            ref_loss = self.refiner_loss(synth_imgs, imgs_ref)

            tqdm_dict = {'ref_loss': ref_loss}
            output = OrderedDict({
                'loss': ref_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

        if optimizer_idx == 1:
            # Process real images
            real_pred = self.discriminator(real_imgs).view(-1, 2)

            valid = torch.ones(real_pred.size(0), dtype=torch.long)
            if self.on_gpu:
                valid = valid.cuda(real_imgs.device.index)

            real_loss = self.adversarial_loss(real_pred, valid)

            # Process synthetic images
            ref_imgs = self.refiner(synth_imgs)
            synth_pred = self.discriminator(ref_imgs).view(-1, 2)

            fake = torch.zeros(synth_pred.size(0), dtype=torch.long)
            if self.on_gpu:
                fake = fake.cuda(ref_imgs.device.index)

            fake_loss = self.adversarial_loss(synth_pred, fake)

            d_loss = real_loss + fake_loss

            tqdm_dict = {'d_loss': d_loss}
            output = OrderedDict({
                'loss': d_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

    def validation_step(self, batch, batch_idx):
        real_imgs, synth_imgs = batch

        # Process real images
        real_pred = self.discriminator(real_imgs).view(-1, 2)

        valid = torch.ones(real_pred.size(0), dtype=torch.long)
        if self.on_gpu:
            valid = valid.cuda(real_imgs.device.index)

        real_loss = self.adversarial_loss(real_pred, valid)

        # Process synthetic images
        ref_imgs = self.refiner(synth_imgs)
        ref_loss = self.refiner_loss(synth_imgs, ref_imgs)
        synth_pred = self.discriminator(synth_imgs).view(-1, 2)

        fake = torch.zeros(synth_pred.size(0), dtype=torch.long)
        if self.on_gpu:
            fake = fake.cuda(ref_imgs.device.index)

        fake_loss = self.adversarial_loss(synth_pred, fake)

        d_loss = real_loss + fake_loss

        tqdm_dict = {'val_d_loss': d_loss}
        output = OrderedDict({
            'val_loss': d_loss,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })
        return output

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_log = {'val_loss': avg_loss}
        output = OrderedDict({
            'val_loss': avg_loss,
            'log': tensorboard_log
        })
        return output

    def configure_optimizers(self):
        lr = self.hparams.learning_rate

        opt_r = torch.optim.Adam(self.refiner.parameters(), lr=lr)
        opt_d = torch.optim.SGD(self.discriminator.parameters(), lr=lr)

        return (
            {'optimizer': opt_r, 'frequency': self.hparams.steps_g},
            {'optimizer': opt_d, 'frequency': self.hparams.steps_d}
        )

    def train_dataloader(self):
        sample_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(224, Image.NEAREST),
            transforms.ToTensor()
        ])

        # Init NYU dataset
        nyu_dataset = NYUDataset(self.hparams.nyu_train_path,
                                 sample_transforms, True)

        # Init synth dataset
        synth_dataset = NYUSynthDataset(self.hparams.synth_train_path,
                                        sample_transforms)

        concat_dataset = ConcatDataset(nyu_dataset, synth_dataset)

        loader = torch.utils.data.DataLoader(
            concat_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True
        )

        return loader

    def val_dataloader(self):
        sample_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(224, Image.NEAREST),
            transforms.ToTensor()
        ])

        # Init NYU dataset
        nyu_dataset = NYUDataset(self.hparams.nyu_val_path, sample_transforms,
                                 False)

        # Init synth dataset
        synth_dataset = NYUSynthDataset(self.hparams.synth_val_path,
                                        sample_transforms)

        concat_dataset = ConcatDataset(nyu_dataset, synth_dataset)

        loader = torch.utils.data.DataLoader(
            concat_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True
        )

        return loader
