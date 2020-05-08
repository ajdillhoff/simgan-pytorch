import math
import random
import argparse

import torch
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
from PIL import Image

from model.SimGAN import SimGAN
from datasets.NYUSynthDataset import NYUSynthDataset


def plot_images(imgs):
    num_rows = int(math.ceil(math.sqrt(imgs.size(0))))
    num_cols = int(imgs.size(0) / num_rows)

    fig, axes = plt.subplots(num_rows, num_cols)
    for i, ax in enumerate(axes.flatten()):
        if i >= imgs.size(0):
            break

        img = imgs[i].detach().squeeze()
        ax.imshow(img)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        ax.axis('off')

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()
    plt.show()


def main(args):
    # Prepare dataset
    sample_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224, Image.NEAREST),
        transforms.ToTensor()
    ])

    dataset = NYUSynthDataset(args.data_path, sample_transforms)

    # Load model checkpoint
    model = SimGAN.load_from_checkpoint(args.checkpoint)
    model.eval()

    # Refine images
    batch_size = 16
    idxs = random.sample(range(len(dataset)), batch_size)
    imgs = torch.stack([dataset[i] for i in idxs])
    ref_imgs = model(imgs)

    # Show results
    plot_images(ref_imgs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", default="", type=str,
                        help="path to checkpoint")
    parser.add_argument("-d", "--data_path", default="", type=str,
                        help="path to synthetic data")
    args = parser.parse_args()
    main(args)
