import json
import argparse
from argparse import Namespace

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from model.SimGAN import SimGAN


def main(args):
    hparams = Namespace(**json.load(open((args.config))))

    model = SimGAN(hparams)

    trainer = Trainer(gpus=hparams.gpus,
                      progress_bar_refresh_rate=1)
    trainer.fit(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config",
                        default="configs/config.json", type=str,
                        help="config file path (default: configs/config.json)")
    args = parser.parse_args()
    main(args)
