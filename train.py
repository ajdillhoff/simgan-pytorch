import json
import argparse
from argparse import Namespace

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from model.SimGAN import SimGAN


def main(args):
    hparams = Namespace(**json.load(open((args.config))))

    if args.resume:
        print("LOADING FROM CHECKPOINT")
        model = SimGAN.load_from_checkpoint(args.resume)
        model.hparams = hparams
    else:
        model = SimGAN(hparams)

    max_steps = None
    if hparams.max_steps > 0:
        max_steps = hparams.max_steps

    checkpoint_callback = ModelCheckpoint(filepath=hparams.save_path,
                                          monitor='val_loss', mode='min')
    early_stop_callback = EarlyStopping('val_loss', patience=hparams.patience)
    trainer = Trainer(gpus=hparams.gpus,
                      max_steps=max_steps,
                      val_check_interval=hparams.val_check_interval,
                      checkpoint_callback=checkpoint_callback,
                      early_stop_callback=early_stop_callback,
                      progress_bar_refresh_rate=1)
    trainer.fit(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--resume", type=str, default="",
                        help="resume from checkpoint (default: none)")
    parser.add_argument("-c", "--config",
                        default="configs/config.json", type=str,
                        help="config file path (default: configs/config.json)")
    args = parser.parse_args()
    main(args)
