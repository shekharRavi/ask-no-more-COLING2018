import os
import io
import h5py
import json
import torch
import argparse
import numpy as np
from multiprocessing import cpu_count
from torchvision.models import vgg16
from torch.utils.data import DataLoader

from dataset import GuessWhatDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.no_grad()

def main(args):

    h5_file = h5py.File(os.path.join(args.data_dir, "vgg_fc8.hdf5"), "w")
    dset = None

    vgg = vgg16(pretrained=True)
    vgg.eval()
    vgg.to(device)

    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        vgg = torch.nn.DataParallel(vgg)

    splits = ['train', 'valid', 'test']
    datasets = dict()
    for split in splits:
        datasets[split] = GuessWhatDataset(
            split=split,
            data_dir=args.data_dir,
            model='image_only',
            coco_dir=args.coco_dir)

    id2file = list()

    for split in splits:

        data_loader = DataLoader(datasets[split], args.batch_size, num_workers=args.num_workers)

        for i, sample in enumerate(data_loader):

            id2file += sample['file_name']

            fc8 = vgg(sample['image'].to(device)).cpu().detach().numpy()

            if dset is not None:
                idx = dset.shape[0]
                dset.resize((idx+fc8.shape[0], 1000))
            else:
                idx=0
                dset = h5_file.create_dataset("vgg_fc8", (fc8.shape[0], 1000), maxshape=(None, 1000), dtype='f')

            dset[idx:] = fc8

    mapping = dict()
    for file in id2file:
        mapping[file] = len(mapping)

    json.dump(mapping, open(os.path.join(args.data_dir, 'imagefile2id.json'), 'w'))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', type=str, default='data')
    parser.add_argument('-cd', '--coco_dir', type=str, default='/Users/timbaumgartner/MSCOCO')
    parser.add_argument('-nw', '--num_workers', type=int, default=2)
    parser.add_argument('-bs', '--batch_size', type=int, default=32)

    args = parser.parse_args()
    args.num_workers = min(cpu_count(), args.num_workers)

    main(args)
