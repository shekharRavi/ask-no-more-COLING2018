import os
import argparse
import torch
from multiprocessing import cpu_count
from torch.utils.data import DataLoader
from collections import OrderedDict, defaultdict

import utils
from dataset import GuessWhatDataset
from models import Guesser

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
            model='guesser',
            min_occ=args.min_occ,
            max_sequence_length=args.max_sequence_length)

    model = Guesser(
        num_word_embeddings=datasets['train'].vocab_size,
        word_embedding_dim=args.word_embedding_dim,
        hidden_size=args.hidden_size,
        num_cat_embeddings=datasets['train'].num_categories,
        cat_embedding_dim=args.cat_embedding_dim,
        mlp_hidden=args.mlp_hidden
        )
    model.to(device)

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)
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

                logits = model(
                    sequence=sample['input'],
                    sequence_length=sample['length'],
                    objects=sample['categories'],
                    spatial=sample['bboxes']
                )

                loss = loss_fn(logits.view(-1, 20), sample['target'].view(-1))
                acc = utils.accuracy(logits.view(-1, 20), sample['target'].view(-1))

                logs[split]['running_loss'] += 1/(iteration+1) * (loss.item() - logs[split]['running_loss'])
                logs[split]['running_acc'] += 1/(iteration+1) * (acc - logs[split]['running_acc'])

                if split == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # bookkeeping
                if iteration % args.print_every == 0 or iteration+1 == len(data_loader):
                    print("%s Batch %04d/%04d Batch-Loss %.3f Running-Mean-Loss %.3f Batch-Acc %.3f Running-Mean-Acc %.3f"
                    %(split.upper(), iteration, len(data_loader)-1, loss.item(), logs[split]['running_loss'], acc, logs[split]['running_acc']))

            print("%s Epoch %02d/%02d Epoch-Loss %.3f Epoch-Acc %.3f"
                %(split.upper(), epoch, args.epochs-1, logs[split]['running_loss'], logs[split]['running_acc']))

        if logs['valid']['running_loss'] < logs['valid']['best_loss'] or epoch == 0:
            logs['valid']['best_epoch'] = epoch
            logs['valid']['best_loss'] = logs['valid']['running_loss']
            logs['valid']['model'] = model.state_dict()

        if (logs['valid']['best_epoch'] + args.patience == epoch ) or (epoch == args.epochs-1):
            if not os.path.exists('bin'):
                os.mkdir('bin')
            torch.save(logs['valid']['model'], 'bin/guesser.pt')
            print("Training stopped. Model saved with best validation loss of %.3f from epoch %02d."
                %(logs['valid']['best_loss'], logs['valid']['best_epoch']))
            break

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Dataset Settings
    parser.add_argument('-d', '--data_dir', type=str, default='data')
    parser.add_argument('-mo', '--min_occ', type=int, default=3)
    parser.add_argument('-ms', '--max_sequence_length', type=int, default=100)

    # Experiment Settings
    parser.add_argument('-e', '--epochs', type=int, default=20)
    parser.add_argument('-b', '--batch_size', type=int, default=54)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.0001)
    parser.add_argument('-p', '--patience', type=int, default=3, help="Number of epochs to wait for validation loss improvement.")
    parser.add_argument('-nw', '--num_workers', type=int, default=2)
    parser.add_argument('-v', '--print_every', type=int, default=100)

    # Hyperparameter
    parser.add_argument('-wed', '--word_embedding_dim', type=int, default=512)
    parser.add_argument('-ced', '--cat_embedding_dim', type=int, default=256)
    parser.add_argument('-hs', '--hidden_size', type=int, default=512)
    parser.add_argument('-mh', '--mlp_hidden', type=int, default=512)

    args = parser.parse_args()
    args.num_workers = min(cpu_count(), args.num_workers)

    main(args)
