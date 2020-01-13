import os
import argparse
import torch
from torchvision.models import vgg16
from multiprocessing import cpu_count
from torch.utils.data import DataLoader
from collections import OrderedDict, defaultdict

from dataset import GuessWhatDataset
from models import QGen

torch.manual_seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):

    splits = ['train', 'valid', 'test']
    datasets = OrderedDict()
    for split in splits:
        datasets[split] = GuessWhatDataset(
            split=split,
            data_dir=args.data_dir,
            model='qgen',
            coco_dir=args.coco_dir,
            min_occ=args.min_occ,
            max_sequence_length=args.max_sequence_length,
            h5File='vgg_fc8.hdf5',
            mapping_file='imagefile2id.json')

    # uncomment for on the fly vgg
    # vgg = vgg16(pretrained=True)
    # vgg.eval()
    # vgg.to(device)


    model = QGen(
        num_embeddings=datasets['train'].vocab_size,
        embedding_dim=args.embedding_dim,
        hidden_size=args.hidden_size,
        visual_embedding_dim=args.visual_embedding_dim,
        padding_idx=datasets['train'].pad
        )
    model.to(device)

    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        # vgg = torch.nn.DataParallel(vgg)
        model = torch.nn.DataParallel(model)

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=datasets['train'].pad)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    logs = defaultdict(lambda: defaultdict(float))

    for epoch in range(args.epochs):

        for split in splits:

            data_loader = DataLoader(
                dataset=datasets[split],
                batch_size=args.batch_size,
                shuffle=split=='train',
                num_workers=args.num_workers
                )

            logs[split]['running_loss'] = 0

            if split == 'train':
                torch.enable_grad()
            else:
                torch.no_grad()

            for iteration, sample in enumerate(data_loader):

                for k, v in sample.items():
                    if torch.is_tensor(v):
                        sample[k] = v.to(device)


                # uncomment for on the fly vgg
                # fc8 = vgg(sample['image'])
                # fc8.detach_()

                fc8 = sample['image']
                if torch.cuda.is_available() and torch.cuda.device_count() > 1:
                    max_length = sample['length'].new_tensor(torch.max(sample['length']))\
                                                 .repeat(torch.cuda.device_count()).view(-1)
                else:
                    max_length = None
                logits = model(sample['input'], sample['length'], fc8, max_length)

                target = sample['target'][:, :logits.size(1)].contiguous()
                loss = loss_fn(logits.view(-1, logits.size(2)), target.view(-1))

                logs[split]['running_loss'] += 1/(iteration+1) * (loss.item() - logs[split]['running_loss'])

                if split == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # bookkeeping
                if iteration % args.print_every == 0 or iteration+1 == len(data_loader):
                    print("%s Batch %04d/%04d Batch-Loss %.3f Running-Loss %.3f"
                        %(split.upper(), iteration, len(data_loader)-1, loss.item(), logs[split]['running_loss']))

            print("%s Epoch %02d/%02d Epoch-Loss %.3f"%(split.upper(), epoch, args.epochs-1, logs[split]['running_loss']))

        if logs['valid']['running_loss'] + args.tolerance < logs['valid']['best_loss'] or epoch == 0:
            logs['valid']['best_epoch'] = epoch
            logs['valid']['best_loss'] = logs['valid']['running_loss']
            if torch.cuda.is_available() and torch.cuda.device_count() > 1:
                logs['valid']['model'] = model.module.state_dict()
            else:
                logs['valid']['model'] = model.state_dict()

        if (logs['valid']['best_epoch'] + args.patience == epoch ) or (epoch == args.epochs-1):
            if not os.path.exists('bin'):
                os.mkdir('bin')
            torch.save(logs['valid']['model'], 'bin/qgenX.pt')
            print("Training stopped. Model saved with best validation loss of %.3f from epoch %02d."
                %(logs['valid']['best_loss'], logs['valid']['best_epoch']))
            break


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Dataset Settings
    parser.add_argument('-d', '--data_dir', type=str, default='data')
    parser.add_argument('-cd', '--coco_dir', type=str, default='/Users/timbaumgartner/MSCOCO')
    parser.add_argument('-mo', '--min_occ', type=int, default=3)
    parser.add_argument('-ms', '--max_sequence_length', type=int, default=100)

    # Experiment Settings
    parser.add_argument('-e', '--epochs', type=int, default=40)
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.0001)
    parser.add_argument('-p', '--patience', type=int, default=5, help="Number of epochs to wait for validation loss improvement.")
    parser.add_argument('-tol', '--tolerance', type=float, default=0.005, help="Minimal validation loss improvement.")
    parser.add_argument('-nw', '--num_workers', type=int, default=2)
    parser.add_argument('-v', '--print_every', type=int, default=100)

    # Hyperparameter
    parser.add_argument('-ed', '--embedding_dim', type=int, default=512)
    parser.add_argument('-ved', '--visual_embedding_dim', type=int, default=512)
    parser.add_argument('-hs', '--hidden_size', type=int, default=1024)

    args = parser.parse_args()
    args.num_workers = min(cpu_count(), args.num_workers)

    main(args)
