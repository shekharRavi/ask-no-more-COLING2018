import os
import argparse
import torch
from torchvision.models import vgg16
from multiprocessing import cpu_count
from torch.utils.data import DataLoader
from collections import OrderedDict, defaultdict

import utils
from dataset import GuessWhatDataset
from models import QGen, DM1

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
            model='dm1',
            coco_dir=args.coco_dir,
            min_occ=args.min_occ,
            max_sequence_length=args.max_sequence_length,
            h5File='vgg_fc8.hdf5',
            mapping_file='imagefile2id.json')


    qgen = QGen(
        num_embeddings=datasets['train'].vocab_size,
        embedding_dim=args.embedding_dim,
        hidden_size=args.hidden_size,
        visual_embedding_dim=args.visual_embedding_dim,
        padding_idx=datasets['train'].pad
        )

    qgen.to(device)
    qgen.load_state_dict(torch.load('bin/qgenX.pt', map_location=lambda storage, loc: storage))

    # vgg = vgg16(pretrained=True)
    # vgg.eval()
    # vgg.to(device)

    model = DM1(
        rnn_hidden_size=args.hidden_size,
        attention_hidden=512
        )
    model.to(device)

    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        # vgg = torch.nn.DataParallel(vgg)
        qgen = torch.nn.DataParallel(qgen)
        model = torch.nn.DataParallel(model)


    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    logs = defaultdict(lambda: defaultdict(float))

    for epoch in range(args.epochs):

        for split in splits:

            data_loader = DataLoader(
                dataset=datasets[split],
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                sampler=torch.utils.data.sampler.WeightedRandomSampler(
                    weights=datasets[split].sample_weights,
                    num_samples=datasets[split].sample_weights.count(0.8)*2,
                    replacement=True),
                drop_last=True
                )

            logs[split]['running_loss'] = 0

            if split == 'train':
                torch.enable_grad()
                model.train()
            else:
                torch.no_grad()
                model.eval()

            for iteration, sample in enumerate(data_loader):

                for k, v in sample.items():
                    if torch.is_tensor(v):
                        sample[k] = v.to(device)

                # fc8 = vgg(sample['image'])
                # fc8.detach_()
                fc8 = sample['image']

                if torch.cuda.is_available() and torch.cuda.device_count() > 1:
                    max_length = sample['length'].new_tensor(torch.max(sample['length']))\
                                                 .repeat(torch.cuda.device_count()).view(-1)
                else:
                    max_length = None

                hidden_states = qgen(
                    input=sample['input'],
                    length=sample['length'],
                    fc8=fc8,
                    max_length=max_length,
                    return_hidden=True
                )

                hidden_states.detach_()

                logits = model(
                    hidden_states=hidden_states,
                    lengths=sample['length'],
                    fc8=fc8,
                    masking=False
                )

                loss = loss_fn(logits, sample['decision_label'])
                logs[split]['running_loss'] += 1/(iteration+1) * (loss.item() - logs[split]['running_loss'])

                acc, acc0, acc1 = utils.two_class_accuracy(predictions=logits, targets=sample['decision_label'])
                logs[split]['running_acc'] += 1/(iteration+1) * (acc - logs[split]['running_acc'])
                logs[split]['running_acc0'] += 1/(iteration+1) * (acc0 - logs[split]['running_acc0'])
                logs[split]['running_acc1'] += 1/(iteration+1) * (acc1 - logs[split]['running_acc1'])

                if split == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # bookkeeping
                if iteration % args.print_every == 0 or iteration+1 == len(data_loader):
                    print("%s Batch %04d/%04d Batch-Loss %.3f Running-Mean-Loss %.3f Batch-Acc %.3f Running-Mean-Acc %.3f Ask-Acc %.3f Gue-Acc %.3f"
                    %(split.upper(), iteration, len(data_loader)-1, loss.item(), logs[split]['running_loss'], acc, logs[split]['running_acc'], logs[split]['running_acc0'], logs[split]['running_acc1']))

            print("%s Epoch %02d/%02d Epoch-Loss %.3f Epoch-Acc %.3f"
                %(split.upper(), epoch, args.epochs-1, logs[split]['running_loss'], logs[split]['running_acc']))

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
            torch.save(logs['valid']['model'], 'bin/dm1_nomask128.pt')
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
    parser.add_argument('-e', '--epochs', type=int, default=10)
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.0001)
    parser.add_argument('-p', '--patience', type=int, default=3, help="Number of epochs to wait for validation loss improvement.")
    parser.add_argument('-tol', '--tolerance', type=float, default=0.01, help="Minimal validation loss improvement.")
    parser.add_argument('-nw', '--num_workers', type=int, default=2)
    parser.add_argument('-v', '--print_every', type=int, default=100)

    # Hyperparameter
    parser.add_argument('-ed', '--embedding_dim', type=int, default=512)
    parser.add_argument('-hs', '--hidden_size', type=int, default=1024)
    parser.add_argument('-ved', '--visual_embedding_dim', type=int, default=512)

    args = parser.parse_args()
    args.num_workers = min(cpu_count(), args.num_workers)

    main(args)
